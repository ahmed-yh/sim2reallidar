"""Charts for the thesis, generated from project data only.

    python thesis_figures/make_thesis_figures.py

Each figure function reads its numbers from the source named in thesis_figures/README.md, and
returns without writing when that source is not on the machine. Figures 1, 2, 3 and 5 appear in
the thesis; 6 and 7 chart the threshold sweep and the classifier comparison, which the thesis
gives as tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
DPI = 300

# ---------------------------------------------------------------- style ----
# Warm = the "before" / problem condition, cool = the "after" / fixed one.
C_BEFORE = "#d94f2b"   # warm orange-red
C_MID    = "#e8a33d"   # amber, for the intermediate "one fix only" condition
C_AFTER  = "#2a7fbf"   # blue
C_ACCENT = "#1f9e6e"   # green, second cool tone
INK      = "#1a1a1a"
MUTED    = "#8a8a8a"
GRID     = "#e3e3e3"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.size": 9,
})


def clean(ax, xgrid=False, ygrid=True):
    """No top/right spines, thin light gridlines on one axis only."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cccccc")
    # matplotlib enables the grid anyway if line properties are passed with
    # False, so the two cases have to be kept apart.
    for axis, on in (("y", ygrid), ("x", xgrid)):
        if on:
            ax.grid(True, axis=axis, color=GRID, linewidth=0.6)
        else:
            ax.grid(False, axis=axis)
    ax.tick_params(length=0)


def titles(fig, title, subtitle, x=0.035, y=0.975, sub_dy=0.038):
    fig.text(x, y, title, fontsize=14, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(x, y - sub_dy, subtitle, fontsize=9, color=MUTED,
             ha="left", va="top")


def summary_panel(ax, rows, header=None):
    """A borderless label/value list. Headline values in bold."""
    ax.axis("off")
    y = 1.0
    if header:
        ax.text(0, y, header, fontsize=9.5, fontweight="bold", color=INK,
                transform=ax.transAxes, va="top")
        y -= 0.085
    for item in rows:
        if item is None:                      # spacer
            y -= 0.045
            continue
        label, value, strong, color = item
        ax.text(0, y, label, fontsize=8.2, color=MUTED,
                transform=ax.transAxes, va="top")
        y -= 0.052
        ax.text(0, y, value, fontsize=11.5 if strong else 9.5,
                fontweight="bold" if strong else "normal",
                color=color or INK, transform=ax.transAxes, va="top")
        # A multi-line value occupies more than one row; without this the next
        # label is drawn straight through it.
        # Line height in axes fractions, derived rather than guessed: a line is
        # fontsize points tall, the axis is (fig height * axis fraction) inches.
        fs = 11.5 if strong else 9.5
        ax_in = ax.get_window_extent().height / ax.figure.dpi
        line_frac = (fs * 1.32 / 72.0) / max(ax_in, 1e-6)
        y -= 0.075 + value.count("\n") * line_frac
    return y


def metres(v):
    """Human-readable distance with the unit chosen by magnitude."""
    if v >= 1000:
        return f"{v/1000:.1f} km"
    if v >= 1:
        return f"{v:.2f} m".rstrip("0").rstrip(".") + (" m" if False else "")
    if v >= 0.001:
        return f"{v*1000:.1f} mm"
    return f"{v*1e6:.0f} um"


def fmt_m(v):
    """Value labels: precise, unit chosen by magnitude."""
    if v >= 1000:
        return f"{v/1000:.1f} km"
    if v >= 1:
        return f"{v:.2f} m"
    return f"{v*1000:.1f} mm"


def fmt_tick(v, _=None):
    """Axis ticks: as short as possible, or decade labels collide on a log
    axis this wide (5.5 mm to 25.9 km is seven orders of magnitude)."""
    if v >= 1000:
        return f"{v/1000:g} km"
    if v >= 1:
        return f"{v:g} m"
    return f"{v*1000:g} mm"


# ============================================================ FIGURE 1 =====

# Values transcribed from the thesis PDF's Chapter 5. Each carries the exact
# sentence it came from so it can be checked against the document.
DRIFT = {
    "Original": {
        "jumps": 285,
        "largest_m": 12_200.0,
        "extent_m": 24_000.0,
        "median_m": None,
        "colour": C_BEFORE,
        "src": "S5.2.5: '285 jumps larger than one metre, the largest reaching "
               "12.2 km, and a total trajectory extent of 24.0 km'",
    },
    "Timestamp fix only": {
        "jumps": 3_932,
        "largest_m": 691.83,
        "extent_m": 25_900.0,
        "median_m": None,
        "colour": C_MID,
        "src": "S5.2.5: '3,932 jumps, the largest reaching 691.83 m'; "
               "S5.3.4: '25.9 km with the timestamp fix alone'",
    },
    "Both fixes": {
        "jumps": 0,
        "largest_m": 0.0055,
        "extent_m": 34.62,
        "median_m": 0.0034,
        "colour": C_AFTER,
        "src": "S5.3.4: 'zero jumps larger than one metre, a largest single "
               "jump of 5.5 mm, a median jump of 3.4 mm, and a trajectory "
               "extent of 34.62 m'",
    },
}

MISSING_PANELS = [
    "per-step position-jump magnitude vs time",
    "before/after X-Y trajectory plots",
    "jump magnitude vs angular rate scatter",
]


def figure_1_drift():
    """Horizontal bars: three conditions with long names cannot share a narrow
    vertical-bar axis without the tick labels colliding."""
    fig = plt.figure(figsize=(7.6, 4.3))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 1.15],
                          left=0.135, right=0.985, top=0.735, bottom=0.165,
                          wspace=0.30)
    names = list(DRIFT)                       # original -> timestamp -> both
    ypos = np.arange(len(names))[::-1]        # first condition at the top
    cols = [DRIFT[n]["colour"] for n in names]

    def hbar(ax, values, log, title, floor):
        ax.barh(ypos, [max(v, floor) for v in values], color=cols, height=0.55)
        if log:
            ax.set_xscale("log")
        ax.set_yticks(ypos)
        ax.set_ylim(-0.6, len(names) - 0.4)
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=7)
        clean(ax, xgrid=True, ygrid=False)
        return ax

    # -- A: jumps > 1 m ----------------------------------------------------
    axA = fig.add_subplot(gs[0, 0])
    vals = [DRIFT[n]["jumps"] for n in names]
    hbar(axA, vals, False, "Jumps > 1 m", 0)
    axA.set_yticklabels(names, fontsize=8.5, color=INK)
    axA.set_xlim(0, max(vals) * 1.32)
    for y, v, c in zip(ypos, vals, cols):
        axA.text(v + max(vals) * 0.03, y, f"{v:,}", va="center", ha="left",
                 fontsize=9.5, fontweight="bold", color=c)

    # -- B: largest single jump (12.2 km .. 5.5 mm -> log) -----------------
    axB = fig.add_subplot(gs[0, 1])
    vals = [DRIFT[n]["largest_m"] for n in names]
    hbar(axB, vals, True, "Largest single jump", 1e-3)
    axB.set_yticklabels([])
    axB.set_xlim(1e-3, 3e6)
    axB.set_xticks([1e-3, 1e0, 1e3])
    axB.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
    axB.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    for y, v, c in zip(ypos, vals, cols):
        axB.text(v * 2.2, y, fmt_m(v), va="center", ha="left",
                 fontsize=9.5, fontweight="bold", color=c)

    # -- C: trajectory extent vs the physical corridor ---------------------
    axC = fig.add_subplot(gs[0, 2])
    vals = [DRIFT[n]["extent_m"] for n in names]
    hbar(axC, vals, True, "Trajectory extent", 1)
    axC.set_yticklabels([])
    axC.set_xlim(1, 3e6)
    axC.set_xticks([1e0, 1e2, 1e4])
    axC.xaxis.set_major_formatter(FuncFormatter(fmt_tick))
    axC.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    axC.axvline(125, color=C_ACCENT, linewidth=1.1, linestyle=(0, (4, 3)))
    axC.text(125, len(names) - 0.42, " corridor\n ~125 m", fontsize=7,
             color=C_ACCENT, va="top", ha="left")
    for y, v, c in zip(ypos, vals, cols):
        axC.text(v * 2.2, y, fmt_m(v), va="center", ha="left",
                 fontsize=9.5, fontweight="bold", color=c)

    # -- D: per-condition summary ------------------------------------------
    axD = fig.add_subplot(gs[0, 3])
    axD.axis("off")
    y = 1.0
    for n in names:
        d = DRIFT[n]
        axD.text(0, y, n, fontsize=8.5, fontweight="bold", color=d["colour"],
                 transform=axD.transAxes, va="top")
        y -= 0.075
        med = fmt_m(d["median_m"]) if d["median_m"] is not None else "not reported"
        for k, v in (("jumps > 1 m", f"{d['jumps']:,}"),
                     ("largest", fmt_m(d["largest_m"])),
                     ("median", med),
                     ("extent", fmt_m(d["extent_m"]))):
            axD.text(0.02, y, k, fontsize=7.4, color=MUTED,
                     transform=axD.transAxes, va="top")
            axD.text(0.62, y, v, fontsize=7.8,
                     fontweight="bold" if k == "extent" else "normal",
                     color=INK, transform=axD.transAxes, va="top")
            y -= 0.062
        y -= 0.045

    titles(fig,
           "Two defects, two fixes: trajectory drift in the tunnel SLAM run",
           "The timestamp fix alone made the jump count worse — it removed the error large enough to dominate the trajectory\n"
           "and exposed a second, independent clock-drift defect. Only both together give a usable result. 150 s window.",
           y=0.975, sub_dy=0.062)

    fig.text(0.02, 0.035,
             "Source: thesis Ch. 5 (§5.2.5, §5.3.2, §5.3.4 and its Fig. 5.4) — already-aggregated summary statistics, not re-derived from raw logs.\n"
             "The jackal-tunnel-slam repository and its per-message diagnostic CSVs are not present on this machine, so the per-step time series,\n"
             "the X–Y trajectory plots and the jump-vs-rotation-rate scatter could not be produced. Median jump is reported in the thesis only for the fixed run.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig1_drift_fix_comparison.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================ FIGURE 2 =====

def read_ply_xyz_intensity(path: Path):
    """Minimal binary-LE PLY reader for the vertex-only CloudCompare export."""
    with open(path, "rb") as fh:
        header = b""
        while b"end_header\n" not in header:
            chunk = fh.read(1)
            if not chunk:
                raise ValueError("no end_header")
            header += chunk
        lines = header.decode("ascii").splitlines()
        if not any(l.startswith("format binary_little_endian") for l in lines):
            raise ValueError(f"{path.name}: not binary_little_endian")
        n = next(int(l.split()[2]) for l in lines if l.startswith("element vertex"))
        props = [l.split()[1:] for l in lines if l.startswith("property ")]
        np_types = {"float": "<f4", "float32": "<f4", "double": "<f8",
                    "uchar": "u1", "int": "<i4", "uint": "<u4"}
        dtype = np.dtype([(name, np_types[t]) for t, name in props])
        raw = np.frombuffer(fh.read(n * dtype.itemsize), dtype=dtype, count=n)
    xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=1).astype(np.float64)
    inten = raw["scalar_intensity"].astype(np.float64) if "scalar_intensity" in dtype.names else None
    return xyz, inten, n


def voxel_downsample(xyz, voxel):
    """One point per occupied voxel -- the standard way to state a cloud's size
    independently of how densely the sensor happened to revisit each surface."""
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return xyz[np.sort(idx)]


def figure_2_cloud():
    src = REPO / "noslil.ply"
    if not src.exists():
        print(f"  SKIP figure 2: {src} not found")
        return None
    xyz, inten, n_raw = read_ply_xyz_intensity(src)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]

    # 20 cm, not 5 cm. Measured on this file: 2, 5 and 10 cm voxels remove
    # essentially nothing (482,757 -> 482,750 at 10 cm), because the cloud has
    # already been decimated to roughly 10 cm spacing before export. Quoting a
    # "downsampled" count that is one point smaller would imply a reduction
    # that did not happen.
    VOXEL = 0.20
    ds = voxel_downsample(xyz, VOXEL)
    n_10cm = len(voxel_downsample(xyz, 0.10))

    # Two extents, because they differ by a factor of 2.5 and only one of them
    # describes the structure. 966 stray points -- SLAM outliers far off the
    # corridor -- stretch the raw bounding box to 315.9 m in X while 99% of the
    # cloud lies within 125 m. Reporting only the bounding box would overstate
    # the reconstruction; reporting only the trimmed one would hide the outliers.
    bbox = xyz.max(axis=0) - xyz.min(axis=0)
    p_lo, p_hi = np.percentile(xyz, [1, 99], axis=0)
    robust = p_hi - p_lo
    v_lo, v_hi = np.percentile(xyz, [0.1, 99.9], axis=0)   # plot window
    n_stray = int((~((xyz >= v_lo) & (xyz <= v_hi)).all(axis=1)).sum())

    floor_z = float(np.percentile(xyz[:, 2], 1.0))
    height = xyz[:, 2] - floor_z
    hi = float(np.percentile(height, 99.0))
    lo = 0.0

    fig = plt.figure(figsize=(7.5, 4.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1.0],
                          height_ratios=[float(robust[1]), float(robust[2])],
                          left=0.075, right=0.985, top=0.735, bottom=0.145,
                          hspace=0.55, wspace=0.10)

    order = np.argsort(height)          # draw high points last
    x, y, z, h = xyz[order, 0], xyz[order, 1], xyz[order, 2], height[order]

    axT = fig.add_subplot(gs[0, 0])
    axT.scatter(x, y, c=h, cmap="viridis", vmin=lo, vmax=hi, s=0.09,
                linewidths=0, rasterized=True)
    axT.set_xlim(v_lo[0], v_hi[0]); axT.set_ylim(v_lo[1], v_hi[1])
    axT.set_ylabel("Y (m)", fontsize=8.5)
    axT.set_title("Top-down (X–Y)", fontsize=9.5, color=INK, loc="left", pad=5)
    axT.set_aspect("equal", adjustable="box")
    clean(axT, xgrid=True)

    axS = fig.add_subplot(gs[1, 0], sharex=axT)
    sc = axS.scatter(x, z, c=h, cmap="viridis", vmin=lo, vmax=hi, s=0.09,
                     linewidths=0, rasterized=True)
    axS.set_ylim(v_lo[2], v_hi[2])
    axS.set_xlabel("X (m)", fontsize=8.5)
    axS.set_ylabel("Z (m)", fontsize=8.5)
    axS.set_title("Side (X–Z)", fontsize=9.5, color=INK, loc="left", pad=5)
    axS.set_aspect("equal", adjustable="box")
    clean(axS, xgrid=True)

    cax = fig.add_axes([0.075, 0.072, 0.24, 0.016])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_label("height above floor (m)", fontsize=7.2, color=MUTED, labelpad=2)
    cb.ax.tick_params(labelsize=6.8, length=0, colors=MUTED, pad=1)
    cb.outline.set_visible(False)

    axN = fig.add_subplot(gs[:, 1])
    summary_panel(axN, [
        ("Points in file", f"{n_raw:,}", True, C_AFTER),
        (f"After {int(VOXEL*100)} cm voxel downsample", f"{len(ds):,}", True, C_ACCENT),
        ("At 10 cm (already ~this spacing)", f"{n_10cm:,}", False, None),
        None,
        ("Extent, 1st–99th pct.", f"{robust[0]:.1f} × {robust[1]:.1f} × {robust[2]:.1f} m",
         True, None),
        ("Raw bounding box", f"{bbox[0]:.1f} × {bbox[1]:.1f} × {bbox[2]:.1f} m", False, None),
        ("Stray points outside window", f"{n_stray:,} ({n_stray/n_raw*100:.2f}%)", False, C_BEFORE),
        None,
        ("Floor reference (1st pct. Z)", f"{floor_z:.2f} m", False, None),
        ("Colour range", f"0 – {hi:.2f} m", False, None),
    ], header="Cloud")

    titles(fig,
           "Reconstructed tunnel point cloud",
           f"KISS-Matcher SLAM output, exported from CloudCompare. {n_raw:,} points; "
           f"{len(ds):,} remain after a {int(VOXEL*100)} cm voxel downsample.\n"
           f"99% of the cloud spans {robust[0]:.1f} × {robust[1]:.1f} × {robust[2]:.1f} m — "
           f"the raw bounding box is {bbox[0]:.0f} m in X only because of a handful of SLAM outliers.",
           y=0.975, sub_dy=0.055)

    fig.text(0.075, 0.005,
             f"Source: {src.name} — {n_raw:,} vertices (x, y, z, scalar_intensity), CloudCompare v2.14.alpha, created 2026-09-19. "
             f"Views cropped to the 0.1–99.9th percentile; both panels are to equal scale.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig2_reconstructed_pointcloud.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================ FIGURE 3 =====

def figure_3_encoders():
    fair_p = REPO / "outputs" / "measured" / "fair_comparison.json"
    eval_p = REPO / "outputs" / "measured" / "evaluate_pointnet2.json"
    if not fair_p.exists():
        print(f"  SKIP figure 3: {fair_p} not found "
              "(run kevin_pipeline/fair_comparison.py --json-out)")
        return None
    fair = json.loads(fair_p.read_text())
    ev = json.loads(eval_p.read_text()) if eval_p.exists() else None

    order = ["pointnet2", "kevin_cnn", "salsanext"]
    label = {"pointnet2": "PointNet++", "kevin_cnn": "CNN autoencoder",
             "salsanext": "SalsaNext"}
    # PointNet++ is the selected candidate -> cool tone; the others warm.
    colour = {"pointnet2": C_AFTER, "kevin_cnn": C_MID, "salsanext": C_BEFORE}
    m = fair["models"]
    order = [k for k in order if k in m]

    # Horizontal, for the same reason as figure 1: "CNN autoencoder" cannot
    # share a narrow vertical-bar axis with two other names without colliding.
    fig = plt.figure(figsize=(7.6, 4.25))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.0, 1.0, 1.2],
                          left=0.135, right=0.985, top=0.695, bottom=0.185,
                          wspace=0.30)
    names = [label[k] for k in order]
    cols = [colour[k] for k in order]
    ypos = np.arange(len(order))[::-1]

    def hb(ax, title):
        ax.set_yticks(ypos)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.set_title(title, fontsize=9, color=INK, loc="left", pad=7)
        clean(ax, xgrid=True, ygrid=False)

    axA = fig.add_subplot(gs[0, 0])
    vals = [m[k]["mse"] for k in order]
    axA.barh(ypos, vals, color=cols, height=0.55)
    hb(axA, "Test MSE (normalized)\nlower is better")
    axA.set_yticklabels(names, fontsize=8.5, color=INK)
    axA.set_xlim(0, max(vals) * 1.42)
    axA.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    for y, v, c in zip(ypos, vals, cols):
        axA.text(v + max(vals) * 0.04, y, f"{v:.6f}", va="center", ha="left",
                 fontsize=8.5, fontweight="bold", color=c)

    axB = fig.add_subplot(gs[0, 1])
    vals = [m[k]["ssim"] for k in order]
    # Zero-based would render three identical full bars; the axis is clipped to
    # just under the observed range and the clipping is stated in the footnote.
    axB.barh(ypos, vals, color=cols, height=0.55)
    hb(axB, "SSIM\nhigher is better")
    axB.set_yticklabels([])
    axB.set_xlim(0.94, 0.9862)
    axB.set_xticks([0.95, 0.97])
    for y, v, c in zip(ypos, vals, cols):
        axB.text(v + 0.0012, y, f"{v:.4f}", va="center", ha="left",
                 fontsize=8.5, fontweight="bold", color=c)

    axC = fig.add_subplot(gs[0, 2])
    hb(axC, "Per-point class accuracy\nhigher is better")
    axC.set_yticklabels([])
    axC.set_xlim(98.5, 99.14)
    axC.set_xticks([98.6, 99.0])
    for y, k in zip(ypos, order):
        v, c = m[k]["pointAcc"], colour[k]
        if v is None:
            # Not a zero bar: this candidate has no classification head at all,
            # and a zero would read as "got everything wrong".
            axC.text(98.52, y, "no classifier", va="center", ha="left",
                     fontsize=8, style="italic", color=MUTED)
            continue
        axC.barh([y], [v * 100], color=c, height=0.55)
        axC.text(v * 100 + 0.015, y, f"{v*100:.2f}%", va="center", ha="left",
                 fontsize=8.5, fontweight="bold", color=c)

    rows = [
        ("Shared held-out test set", f"{fair['testSamples']:,} samples", True, None),
        ("Shared max_range", f"{fair['sharedMaxRange']:.3f} m", False, None),
        None,
        ("Best reconstruction", "PointNet++", True, C_AFTER),
        ("MSE advantage over next best", f"{m['kevin_cnn']['mse']/m['pointnet2']['mse']:.1f}x", True, C_AFTER),
    ]
    if ev:
        rows += [
            None,
            ("PointNet++ macro-F1 @ 0.97", f"{ev['macroF1']*100:.1f}%", True, C_ACCENT),
            ("Per-class F1 (sphere/cyl/box2/box1)",
             " / ".join(f"{r['f1']*100:.0f}" for r in ev["rows"] if r["classId"] > 0),
             False, None),
        ]
    axD = fig.add_subplot(gs[0, 3])
    summary_panel(axD, rows, header="Shared protocol")

    titles(fig,
           "Three observation encoders on one shared test set",
           f"Final test-set metrics under an identical protocol — the same {fair['testSamples']:,} held-out samples and the same\n"
           "normalization constant, with PointNet++'s free-form output reprojected onto the ray grid to put all three on one scale.",
           x=0.02, y=0.975, sub_dy=0.068)

    fig.text(0.02, 0.028,
             "Source: outputs/measured/fair_comparison.json (kevin_pipeline/fair_comparison.py --json-out) and outputs/measured/evaluate_pointnet2.json (evaluate.py --json-out).\n"
             "Per-epoch training curves are NOT plotted because none were ever saved: train_utils.fit() hands each epoch to a log_fn that only prints to stdout, and every\n"
             "outputs/*/train_metrics.json stores final scalars alone. SSIM axis starts at 0.940 — a zero-based axis would render three visually identical bars.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig3_encoder_comparison.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================ FIGURE 5 =====
# §5.6.4 Decimation Validation. Thesis Table 5.1, as a figure -- four classes
# across four configurations is hard to read as numbers in a row.
DECIMATION = [
    # label, points, sphere, cylinder, box2, box1, is_chosen
    ("Full-resolution baseline",      65_536, 581, 410, 521, 600, False),
    ("Every 2nd ring, every 2nd col", 16_384, 481, 271, 259, 542, False),
    ("Every ring, every 4th col",     16_384, 424, 222, 413, 550, False),
    ("Every ring, every 2nd col",     32_768, 528, 316, 501, 598, True),
]
N_FRAMES = 600


def figure_5_decimation():
    classes = ["Sphere", "Cylinder", "Box 2", "Box 1"]
    fig = plt.figure(figsize=(7.6, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.0],
                          left=0.075, right=0.985, top=0.695, bottom=0.185,
                          wspace=0.22)

    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(classes))
    width = 0.2
    for i, (lab, pts, sp, cy, b2, b1, chosen) in enumerate(DECIMATION):
        vals = np.array([sp, cy, b2, b1]) / N_FRAMES * 100
        if i == 0:
            colour, alpha, edge = MUTED, 0.35, "none"      # baseline = reference
        elif chosen:
            colour, alpha, edge = C_AFTER, 1.0, "none"
        else:
            colour, alpha, edge = C_BEFORE if i == 1 else C_MID, 0.9, "none"
        ax.bar(x + (i - 1.5) * width, vals, width * 0.92, color=colour, alpha=alpha,
               edgecolor=edge, label=lab)
        if chosen:
            for xi, v in zip(x + (i - 1.5) * width, vals):
                ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color=C_AFTER)
    ax.set_xticks(x); ax.set_xticklabels(classes, fontsize=9)
    ax.set_ylim(0, 132)
    ax.set_ylabel(f"frames containing the class (% of {N_FRAMES})", fontsize=8.5)
    ax.set_title("Per-class frame coverage by decimation pattern", fontsize=9,
                 color=INK, loc="left", pad=7)
    clean(ax)

    # Labelled directly beneath the group rather than in a boxed legend.
    handles = [
        ("Full-resolution baseline (65,536 pts)", MUTED, 0.35),
        ("Every 2nd ring, every 2nd col (16,384)", C_BEFORE, 0.9),
        ("Every ring, every 4th col (16,384)", C_MID, 0.9),
        ("Every ring, every 2nd col (32,768) — chosen", C_AFTER, 1.0),
    ]
    for i, (lab, c, a) in enumerate(handles):
        ax.add_patch(plt.Rectangle((0.008, 0.968 - i * 0.052), 0.020, 0.032,
                                   transform=ax.transAxes, color=c, alpha=a, clip_on=False))
        ax.text(0.036, 0.984 - i * 0.052, lab, transform=ax.transAxes,
                fontsize=7.3, color=INK, va="top")

    axS = fig.add_subplot(gs[0, 1])
    summary_panel(axS, [
        ("Chosen", "Every ring,\nevery 2nd column", True, C_AFTER),
        ("Points per scan", "32,768", False, None),
        None,
        ("Weakest class, chosen pattern", "Cylinder 53%", True, C_BEFORE),
        ("Cylinder at full resolution", "68%", False, None),
        None,
        ("Why not decimate rings", "only 64 exist;\na small object spans\nonly a handful", False, None),
    ], header="Decision")

    titles(fig,
           "Decimating azimuth, never rings",
           "At an equal or smaller point budget, thinning only the azimuth direction keeps far more of the object classes than\n"
           "dropping rings. Cylinder stays the weakest class throughout — a limitation of the recording, not of the decimation.",
           x=0.02, y=0.975, sub_dy=0.066)

    fig.text(0.02, 0.028,
             "Source: thesis §5.6.4 / Table 5.1, measured against the full recording by the decimation-validation tooling; also in docs/data.md. "
             "Percentages are the table's\nframe counts out of 600 recorded frames. The baseline is no decimation at all, not a competing strategy.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig5_decimation_validation.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================ FIGURE 6 =====

def figure_6_threshold():
    """§5.7.3 -- the confidence threshold, and why 0.97.

    The thesis describes the threshold as a display decision but reports only
    the un-thresholded numbers (macro-F1 68.1%, precision 33.6-71.4%). This is
    the sweep that justifies the value actually shipped.
    """
    sweep_p = REPO / "outputs" / "measured" / "evaluate_pointnet2_sweep.json"
    if not sweep_p.exists():
        print(f"  SKIP figure 6: {sweep_p} not found "
              "(run evaluate.py --sweep '0,0.3,...' --json-out ...)")
        return None
    data = json.loads(sweep_p.read_text())["sweep"]
    data.sort(key=lambda d: d["threshold"])

    th = np.array([d["threshold"] for d in data])
    macro = np.array([d["macroF1"] for d in data]) * 100
    best_i = int(np.argmax(macro))

    names = {4: "Sphere", 5: "Cylinder", 6: "Box 2", 7: "Box 1"}
    fig = plt.figure(figsize=(7.6, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.3, 1.05],
                          left=0.08, right=0.985, top=0.695, bottom=0.175,
                          wspace=0.40)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(th, macro, "-o", color=C_AFTER, linewidth=1.9, markersize=4)
    axA.scatter([th[best_i]], [macro[best_i]], s=70, facecolor="none",
                edgecolor=C_ACCENT, linewidth=1.6, zorder=5)
    axA.text(th[best_i], macro[best_i] + 2.0, f"peak {macro[best_i]:.1f}%\n@ {th[best_i]:g}",
             fontsize=7.6, fontweight="bold", color=C_ACCENT, ha="center")
    axA.text(th[0], macro[0] - 3.0, f"argmax\n{macro[0]:.1f}%", fontsize=7.6,
             color=C_BEFORE, ha="left", va="top", fontweight="bold")
    axA.set_xlabel("confidence threshold", fontsize=8.5)
    axA.set_ylabel("macro-F1 over object classes (%)", fontsize=8.5)
    axA.set_title("Choosing the threshold", fontsize=9, color=INK, loc="left", pad=7)
    axA.set_ylim(min(macro) - 8, max(macro) + 9)
    clean(axA)

    axB = fig.add_subplot(gs[0, 1])
    palette = {4: C_MID, 5: C_BEFORE, 6: C_ACCENT, 7: C_AFTER}
    ends = []
    for cid, nm in names.items():
        prec = [next((r["precision"] for r in d["rows"] if r["classId"] == cid), np.nan) * 100
                for d in data]
        axB.plot(th, prec, "-", linewidth=1.7, color=palette[cid])
        ends.append([prec[-1], nm, palette[cid]])

    # Direct end-of-line labels instead of a boxed legend -- but Box 1 and
    # Sphere converge to within a point of each other at high thresholds, so
    # their labels overlap unless pushed apart. Nudge only the label; the line
    # still ends at its true value.
    ends.sort(key=lambda e: e[0])
    span = axB.get_ylim()[1] - axB.get_ylim()[0]
    min_gap = span * 0.055
    for i in range(1, len(ends)):
        if ends[i][0] - ends[i - 1][0] < min_gap:
            ends[i][0] = ends[i - 1][0] + min_gap
    for y_lab, nm, col in ends:
        axB.text(th[-1] + 0.012, y_lab, nm, fontsize=7.6, fontweight="bold",
                 color=col, va="center")
    axB.set_xlabel("confidence threshold", fontsize=8.5)
    axB.set_ylabel("precision (%)", fontsize=8.5)
    axB.set_title("Precision is what the threshold buys", fontsize=9, color=INK,
                  loc="left", pad=7)
    axB.set_xlim(th[0] - 0.02, th[-1] + 0.20)
    clean(axB)

    best = data[best_i]
    objs = [r for r in best["rows"] if r["classId"] > 0]
    base = data[0]
    base_objs = [r for r in base["rows"] if r["classId"] > 0]
    axC = fig.add_subplot(gs[0, 2])
    summary_panel(axC, [
        ("Shipped threshold", f"{best['threshold']:g}", True, C_AFTER),
        ("Macro-F1 at argmax", f"{macro[0]:.1f}%", False, C_BEFORE),
        ("Macro-F1 at threshold", f"{macro[best_i]:.1f}%", True, C_ACCENT),
        None,
        ("Worst precision, argmax",
         f"{min(r['precision'] for r in base_objs)*100:.1f}%", False, C_BEFORE),
        ("Worst precision, threshold",
         f"{min(r['precision'] for r in objs)*100:.1f}%", True, C_ACCENT),
        None,
        ("Recall cost", f"{(np.mean([r['recall'] for r in base_objs]) - np.mean([r['recall'] for r in objs]))*100:.1f} pts",
         False, None),
    ], header="Display threshold")

    titles(fig,
           "The confidence threshold is a display decision, measured not guessed",
           "Arg-max always picks some class, so on 99%-environment scans a small per-point error rate paints visible speckle.\n"
           "Suppressing low-confidence OBJECT predictions trades a little recall for a large gain in precision.",
           x=0.02, y=0.975, sub_dy=0.066)

    fig.text(0.02, 0.028,
             f"Source: outputs/measured/evaluate_pointnet2_sweep.json — evaluate.py --sweep, {data[0].get('rows') and json.loads(sweep_p.read_text())['testSamples']:,} held-out SIMULATED test samples, one shared forward pass.\n"
             "The threshold is selected on labelled simulated data because real data has no ground truth; it changes what is displayed, never the model's weights.",
             fontsize=6.4, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig6_confidence_threshold.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ============================================================ FIGURE 7 =====

def figure_7_classifiers():
    """The classification comparison the thesis is missing.

    Section 5.6.6 calls classification "a near-tie" on 98.96% vs 98.93% overall
    point accuracy -- while the same section argues, correctly, that overall
    accuracy is meaningless here because ~99% of points are environment. This
    is the same two models scored on the four classes that carry the signal.
    """
    fp = REPO / "outputs" / "measured" / "fair_comparison.json"
    if not fp.exists():
        print(f"  SKIP figure 7: {fp} not found")
        return None
    data = json.loads(fp.read_text())
    cls = data.get("classification")
    if not cls:
        print("  SKIP figure 7: fair_comparison.json has no classification block "
              "(re-run kevin_pipeline/fair_comparison.py --json-out)")
        return None

    def curve(key):
        sw = sorted(cls[key], key=lambda d: d["threshold"])
        return (np.array([d["threshold"] for d in sw]),
                np.array([d["macroF1"] for d in sw]) * 100, sw)

    th, pn2, pn2_sw = curve("pointnet2")
    _, salsa, salsa_sw = curve("salsanext")
    matched = cls.get("salsanext_column_matched")
    mt = np.array([d["macroF1"] for d in sorted(matched, key=lambda d: d["threshold"])]) * 100         if matched else None

    fig = plt.figure(figsize=(7.6, 4.15))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.35, 1.05],
                          left=0.08, right=0.985, top=0.695, bottom=0.175, wspace=0.40)

    axA = fig.add_subplot(gs[0, 0])
    axA.plot(th, pn2, "-o", color=C_AFTER, linewidth=1.9, markersize=3.5)
    axA.plot(th, salsa, "-o", color=C_BEFORE, linewidth=1.9, markersize=3.5)
    axA.text(th[-1] + 0.012, pn2[-1] - 1.4, "PointNet++", fontsize=7.8, fontweight="bold",
             color=C_AFTER, va="center")
    axA.text(th[-1] + 0.012, salsa[-1] + 1.4, "SalsaNext", fontsize=7.8, fontweight="bold",
             color=C_BEFORE, va="center")
    # The column-matched curve is deliberately NOT drawn: it lands within 0.001
    # of the unmatched one at every threshold, so plotting it adds a line that
    # is indistinguishable from its neighbour while colliding with the labels.
    # The number is reported in the panel instead, which is what carries it.
    axA.set_xlabel("confidence threshold", fontsize=8.5)
    axA.set_ylabel("macro-F1 over object classes (%)", fontsize=8.5)
    axA.set_title("Both classifiers, swept", fontsize=9, color=INK, loc="left", pad=7)
    axA.set_xlim(th[0] - 0.02, th[-1] + 0.22)
    clean(axA)

    # per-class at the shipped threshold
    def at(sw, t):
        return next(d for d in sw if abs(d["threshold"] - t) < 1e-9)
    a, b = at(pn2_sw, 0.97), at(salsa_sw, 0.97)
    names = {4: "Sphere", 5: "Cylinder", 6: "Box 2", 7: "Box 1"}
    ids = [4, 5, 6, 7]
    fa = [next(r["f1"] for r in a["rows"] if r["classId"] == c) * 100 for c in ids]
    fb = [next(r["f1"] for r in b["rows"] if r["classId"] == c) * 100 for c in ids]

    axB = fig.add_subplot(gs[0, 1])
    y = np.arange(len(ids))[::-1]
    axB.barh(y + 0.19, fa, 0.36, color=C_AFTER)
    axB.barh(y - 0.19, fb, 0.36, color=C_BEFORE)
    for yy, va, vb in zip(y, fa, fb):
        axB.text(va + 1.0, yy + 0.19, f"{va:.0f}", va="center", fontsize=7.4,
                 fontweight="bold", color=C_AFTER)
        axB.text(vb + 1.0, yy - 0.19, f"{vb:.0f}", va="center", fontsize=7.4,
                 fontweight="bold", color=C_BEFORE)
    axB.set_yticks(y); axB.set_yticklabels([names[c] for c in ids], fontsize=8.5, color=INK)
    axB.set_xlim(0, 108); axB.set_xlabel("F1 at threshold 0.97 (%)", fontsize=8.5)
    axB.set_title("Per class, where it matters", fontsize=9, color=INK, loc="left", pad=7)
    clean(axB, xgrid=True, ygrid=False)

    gap = b["macroF1"] - a["macroF1"]
    cyl_a = next(r["f1"] for r in a["rows"] if r["classId"] == 5)
    cyl_b = next(r["f1"] for r in b["rows"] if r["classId"] == 5)
    rows = [
        ("Overall point accuracy", "98.96% vs 98.93%", False, None),
        ("...but ~99% of points are environment", "", False, None),
        None,
        ("Macro-F1 at arg-max", f"{pn2[0]:.1f}% vs {salsa[0]:.1f}%", False, None),
        ("Macro-F1 at 0.97", f"{a['macroF1']*100:.1f}% vs {b['macroF1']*100:.1f}%", True, C_BEFORE),
        ("Cylinder F1 at 0.97", f"{cyl_a*100:.1f}% vs {cyl_b*100:.1f}%", True, C_BEFORE),
    ]
    if mt is not None:
        rows += [None, ("Same, on PointNet++'s own rays", f"{mt[list(th).index(0.97)]:.1f}% (identical)", False, None)]
    axC = fig.add_subplot(gs[0, 2])
    summary_panel(axC, rows, header="PointNet++ vs SalsaNext")

    titles(fig,
           "Classification is not a near-tie once environment is excluded",
           f"On overall point accuracy the two are within 0.03 points. On the four object classes at the shipped threshold,\n"
           f"SalsaNext leads by {gap*100:.1f} F1 points, and by {(cyl_b-cyl_a)*100:.1f} on cylinder - the weakest class for both.",
           x=0.02, y=0.975, sub_dy=0.066)

    fig.text(0.02, 0.028,
             "Source: outputs/measured/fair_comparison.json (kevin_pipeline/fair_comparison.py --json-out), 1,140 held-out SIMULATED test samples, one shared forward pass per model.\n"
             "Each model is scored in its own native domain: PointNet++ on its 32,768 decimated points, SalsaNext on the valid cells of its 64x1024 grid. The dashed curve restricts\n"
             "SalsaNext to the same every-2nd-azimuth rays PointNet++ is given, so the comparison is not confounded by sampling density. Reconstruction is unaffected: PointNet++ still leads 7.5x.",
             fontsize=6.3, color=MUTED, ha="left", va="bottom")

    p = OUT / "fig7_classifier_head_to_head.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    made = []
    for fn in (figure_1_drift, figure_2_cloud, figure_3_encoders,
               figure_5_decimation, figure_6_threshold,
               figure_7_classifiers):
        p = fn()
        if p:
            made.append(p)
            print(f"  wrote {p.relative_to(REPO)}  "
                  f"({p.stat().st_size/1024:.0f} KB)")
    print(f"\n{len(made)} figure(s) in {OUT}")
