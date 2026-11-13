"""Runs the demo scenarios the assignment asks for, one after another (use this for the video).

Run from the repo root:
    python src/demo.py
    python src/demo.py --member towi --other harmony --outsider ama
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from app import run

IMG = ROOT / "data" / "images"
AUD = ROOT / "data" / "audio"
UNAUTH = ROOT / "data" / "unauthorized"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", default="towi", help="member for the approved transaction")
    parser.add_argument("--other", default="kethia", help="a different member, for the wrong-voice case")
    parser.add_argument("--outsider", default=None, help="outsider name, e.g. ama for outsider_ama_*")
    args = parser.parse_args()
    m, o = args.member, args.other

    def first(folder: Path, pattern: str):
        hits = sorted(folder.glob(pattern))
        return hits[0] if hits else None

    out = args.outsider or "*"
    scenarios = [
        ("1. Approved transaction", first(IMG / m, f"{m}_neutral.*"), first(AUD / m, f"{m}_yes_approve.wav"), True),
        ("2. Unauthorized face", first(UNAUTH / "images", f"outsider_{out}_*"), first(AUD / m, f"{m}_yes_approve.wav"), False),
        ("3. Right face, outsider voice", first(IMG / m, f"{m}_neutral.*"), first(UNAUTH / "audio", f"outsider_{out}_*.wav"), False),
        ("4. Right face, another member's voice", first(IMG / m, f"{m}_neutral.*"), first(AUD / o, f"{o}_yes_approve.wav"), False),
    ]

    results = []
    for title, face, voice, expected in scenarios:
        print("\n" + "#" * 60 + f"\n# {title}\n" + "#" * 60)
        if face is None or voice is None:
            print("Skipped: missing file (face or voice) for this scenario.")
            results.append((title, "skipped", expected))
            continue
        got = run(face, voice)
        results.append((title, "approved" if got else "denied", expected))

    print("\n" + "=" * 60 + "\nSummary")
    for title, got, expected in results:
        want = "approved" if expected else "denied"
        mark = "OK " if got == want else ("-- " if got == "skipped" else "!! ")
        print(f"{mark}{title}: {got} (expected {want})")


if __name__ == "__main__":
    main()
