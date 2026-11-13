"""Command-line simulation of the full system.

    face image -> face model    -> known member?            no -> ACCESS DENIED
               -> product model runs for that member's customer profile (result held back)
    voice clip -> voice model   -> same member, confident?  no -> ACCESS DENIED
               -> show the product recommendation

Run from the repo root:
    python src/app.py --face data/images/towi/towi_neutral.jpg --voice data/audio/towi/towi_yes_approve.wav
    python src/app.py            (asks for the two paths)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import config                    # MEMBER_TO_CUSTOMER: which customer profile each member uses
import product_model as pm       # predict_for_customer(customer_id) -> (category, probabilities)
import audio_pipeline as ap      # verify_speaker(path, expected_member) -> (accepted, predicted, confidence)

try:
    import image_features as fp  # recognise_face(path) -> (member or "no_face", confidence)
except ImportError as err:
    fp, FACE_IMPORT_ERROR = None, err


def check_face(path: Path):
    """Return (member, confidence), or (None, confidence) when access must be denied.

    The face model always picks one of the four members, even for a stranger, so a
    low-confidence answer is treated as "not recognised" (threshold in config.py).
    """
    if fp is None:
        raise SystemExit(f"Face model could not be loaded (src/image_features.py): {FACE_IMPORT_ERROR}")
    member, conf = fp.recognise_face(path)
    if member == "no_face" or member not in config.MEMBER_TO_CUSTOMER or conf < config.FACE_THRESHOLD:
        return None, conf
    return member, conf


def run(face_path: Path, voice_path: Path) -> bool:
    """Run one full transaction. Returns True if a recommendation was shown."""
    for p in (face_path, voice_path):
        if not Path(p).exists():
            print(f"File not found: {p}")
            return False

    print("\n=== Multimodal authentication ===")

    # Step 1: who is this? The face model must recognise a member with enough confidence.
    print(f"[1/3] Face recognition: {Path(face_path).name}")
    member, face_conf = check_face(face_path)
    if member is None:
        why = "no face found" if face_conf == 0 else f"confidence {face_conf:.2f} < threshold {config.FACE_THRESHOLD}"
        print(f"      ACCESS DENIED: face not recognised ({why})")
        return False
    print(f"      Recognised {member} (confidence {face_conf:.2f})")

    # Step 2: run the product model now, but hold the result back until the voice confirms.
    customer_id = config.MEMBER_TO_CUSTOMER[member]
    category, probs = pm.predict_for_customer(customer_id)
    print(f"[2/3] Product model run for customer {customer_id}. Result held until voice approval.")

    # Step 3: the voice must belong to the SAME member, with enough confidence.
    print(f"[3/3] Voice verification: {Path(voice_path).name}")
    accepted, speaker, voice_conf = ap.verify_speaker(voice_path, member)
    if not accepted:
        reason = (f"voice matched {speaker}, not {member}" if speaker != member
                  else f"voice confidence {voice_conf:.2f} is below the threshold")
        print(f"      ACCESS DENIED: {reason}")
        return False
    print(f"      Voice confirmed as {member} (confidence {voice_conf:.2f})")

    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    print(f"\n>>> Recommended for {member}: {category}")
    print("    Top categories: " + ", ".join(f"{c} {p:.0%}" for c, p in top))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Face + voice authentication, then product recommendation")
    parser.add_argument("--face", type=Path, help="path to a face image")
    parser.add_argument("--voice", type=Path, help="path to a voice clip (.wav)")
    args = parser.parse_args()
    face = args.face or Path(input("Path to face image: ").strip().strip('"'))
    voice = args.voice or Path(input("Path to voice clip: ").strip().strip('"'))
    sys.exit(0 if run(face, voice) else 1)


if __name__ == "__main__":
    main()
