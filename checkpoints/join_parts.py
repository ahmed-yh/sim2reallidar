"""Restore the trained checkpoints into outputs/, where the training, evaluation and
inference scripts expect them.

GitHub refuses files over 100 MB, so each SalsaNext checkpoint is stored here as
90 MB parts (<name>.pt.part00, .part01, ...). This script joins the parts, copies
the other files, and checks every checkpoint against SHA256SUMS.

    python checkpoints/join_parts.py
"""
import hashlib
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    expected = {}
    for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            expected[rel.strip()] = digest

    for src in sorted(HERE.rglob("*")):
        if not src.is_file() or src.parent == HERE:
            continue
        rel = src.relative_to(HERE)
        if src.suffix.startswith(".part"):
            if src.suffix != ".part00":
                continue
            target = OUT / rel.parent / src.stem          # strip ".part00"
            target.parent.mkdir(parents=True, exist_ok=True)
            parts = sorted(src.parent.glob(src.stem + ".part*"))
            with open(target, "wb") as out:
                for part in parts:
                    out.write(part.read_bytes())
        else:
            target = OUT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)

    bad = [rel for rel, digest in expected.items() if sha256(OUT / rel) != digest]
    if bad:
        raise SystemExit("checksum mismatch: " + ", ".join(bad))
    print(f"{len(expected)} checkpoints restored into {OUT} and verified")


if __name__ == "__main__":
    main()
