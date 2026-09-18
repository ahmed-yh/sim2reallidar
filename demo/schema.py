"""Payload validation. Standard library only -- no torch, no numpy -- so
`build_demo.py --check` and the pytest suite run anywhere, including in CI and
on a laptop with none of the project's data.

The central check is telemetry consistency. The real_bags/*.html pages shipped
broken because the page named `m.x` and `m.class_counts` while the generator
emitted meta entries of only {"t", "mse"}: render(0) threw before play() was
ever called, so autoplay, the telemetry panel and the legend were all silently
dead and only manual scrubbing moved the image. Asserting a payload's own
contract turns that from "a bug someone fixes once" into "a bug that cannot
reach the page".
"""
from __future__ import annotations

import json
import re
from pathlib import Path

KNOWN_SECTION_TYPES = {
    "hero", "prose", "comparison", "table", "tabs",
    "gallery", "video", "player", "pointcloud", "tunnel",
}

# Keys the legend consumes; never telemetry rows.
COUNT_KEYS = {"class_counts", "pred_class_counts"}


def _load_registered(path: Path):
    """Read a DEMO.register("kind", "id", {...}); payload file back into Python.

    The payloads are JS wrappers around JSON (see build_demo.write_payload for
    why), so parsing means peeling the wrapper off.
    """
    text = path.read_text(encoding="utf-8")
    start = text.index("{")
    end = text.rindex("}")
    return json.loads(text[start:end + 1])


def check_player_payload(path: Path) -> list[str]:
    problems = []
    try:
        p = _load_registered(path)
    except Exception as e:  # noqa: BLE001
        return [f"{path.name}: not parseable ({type(e).__name__}: {e})"]

    meta = p.get("meta") or []
    if not meta:
        return [f"{path.name}: no meta entries"]

    # 1. every declared telemetry field must be readable from EVERY frame
    for field in p.get("telemetryFields", []):
        keys = field.get("from", [field["key"]])
        missing = [i for i, m in enumerate(meta) if any(k not in m for k in keys)]
        if missing:
            problems.append(
                f"{path.name}: telemetry field {field['key']!r} reads {keys} but "
                f"{len(missing)} frame(s) lack it (first: index {missing[0]})")

    # 2. a declared legend needs its counts key present everywhere
    legend = p.get("legend")
    if legend and legend.get("source"):
        src = legend["source"]
        missing = [i for i, m in enumerate(meta) if src not in m]
        if missing:
            problems.append(
                f"{path.name}: legend source {src!r} missing from {len(missing)} frame(s)")

    # 3. frame count must match meta, and file-mode frames must exist
    frames = p.get("frames") or {}
    count = frames.get("count")
    if count is not None and count != len(meta):
        problems.append(f"{path.name}: frames.count={count} but {len(meta)} meta entries")
    if frames.get("mode") == "files":
        base = path.parent.parent / frames["base"]
        pad, ext = frames.get("pad", 4), frames.get("format", "png")
        for i in (0, len(meta) - 1):
            f = base / f"{i:0{pad}d}.{ext}"
            if not f.exists():
                problems.append(f"{path.name}: frame file missing: {f}")

    # 4. timestamps must be non-decreasing, or the real-time scheduler computes
    #    a negative delay and playback stalls
    ts = [m.get("t") for m in meta if isinstance(m.get("t"), (int, float))]
    for i in range(1, len(ts)):
        if ts[i] < ts[i - 1]:
            problems.append(f"{path.name}: meta[{i}].t goes backwards ({ts[i-1]} -> {ts[i]})")
            break

    return problems


def check_manifest(demo_dir: Path) -> list[str]:
    problems = []
    manifest_path = demo_dir / "payloads" / "manifest.js"
    if not manifest_path.exists():
        return ["payloads/manifest.js is missing"]
    try:
        m = _load_registered(manifest_path)
    except Exception as e:  # noqa: BLE001
        return [f"manifest.js not parseable ({type(e).__name__}: {e})"]

    for s in m.get("sections", []):
        if s.get("type") not in KNOWN_SECTION_TYPES:
            problems.append(f"manifest: section {s.get('id')!r} has unknown type {s.get('type')!r}")
        for key in ("src",):
            if s.get(key):
                target = demo_dir / "payloads" / s[key]
                if not target.exists():
                    problems.append(f"manifest: section {s.get('id')!r} -> missing {s[key]}")
        for tab in s.get("tabs", []):
            if tab.get("src") and not (demo_dir / "payloads" / tab["src"]).exists():
                problems.append(f"manifest: tab {tab.get('label')!r} -> missing {tab['src']}")

    if not m.get("classes"):
        problems.append("manifest: no classes[] -- swatches and legends will be empty")
    return problems


def check_no_remote_subresources(demo_dir: Path) -> list[str]:
    """Nothing may load from a remote origin.

    One forgotten Google Fonts @import is the difference between "works on the
    examiner's train ride" and a page that silently loses its typography. Links
    the user clicks (<a href>) are fine; subresources are not.
    """
    problems = []
    targets = [demo_dir / "index.html"]
    targets += list((demo_dir / "assets").rglob("*.css"))
    targets += list((demo_dir / "assets").rglob("*.js"))

    pattern = re.compile(r"""(src|href)\s*=\s*["']https?://|@import\s+url\(["']?https?://|url\(["']?https?://""")
    for path in targets:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("*", "//", "/*", "#")):
                continue
            m = pattern.search(line)
            if not m:
                continue
            # an <a href> to the repo is a link, not a subresource
            if "<a " in line or 'rel="noopener"' in line:
                continue
            problems.append(f"{path.relative_to(demo_dir)}:{lineno}: remote subresource: {stripped[:80]}")
    return problems


def check_all(demo_dir: Path) -> list[str]:
    problems = []
    problems += check_manifest(demo_dir)
    problems += check_no_remote_subresources(demo_dir)
    players = sorted((demo_dir / "payloads" / "players").glob("*.js")) \
        if (demo_dir / "payloads" / "players").exists() else []
    for p in players:
        problems += check_player_payload(p)
    return problems


if __name__ == "__main__":
    import sys
    found = check_all(Path(__file__).resolve().parent)
    for f in found:
        print("FAIL " + f)
    print("OK" if not found else f"{len(found)} problem(s)")
    sys.exit(1 if found else 0)
