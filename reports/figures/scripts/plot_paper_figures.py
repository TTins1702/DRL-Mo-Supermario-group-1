"""Tạo toàn bộ hình vector (.pdf) chuẩn xuất bản khoa học cho báo cáo LaTeX (Nhóm 1).

Tuân thủ nguyên tắc thiết kế hình khoa học:
- 100% vector PDF (pdf.fonttype = 42), kích thước in thực tính bằng inch (width = 6.8 in).
- Bảng màu an toàn mù màu Okabe-Ito nhất quán toàn bài + phân biệt kiểu nét/marker.
- Loại bỏ khung trên/phải, lưới mờ alpha=0.25, nhãn chữ >= 8 pt, không chồng lấp.
- Đọc trực tiếp từ runs/*/progress.csv, evals.csv và ket-qua-*.json.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

# Cấu hình typography & vector PDF chuẩn xuất bản
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "DejaVu Sans",
    "font.size": 8.0,
    "axes.titlesize": 8.6,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.4,
    "ytick.labelsize": 7.4,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.7,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.4,
})

# Bảng màu Okabe-Ito nhất quán toàn bài
C_BLUE = "#0072B2"       # Chính / Nhóm (Ours, gamma=0.90, Pha 2)
C_VERM = "#D55E00"       # Đối chứng / Pha 1 / gamma=0.99
C_GREEN = "#009E73"      # Đích / Thành công / Mốc đạt
C_PURPLE = "#CC79A7"     # Entropy
C_ORANGE = "#E69F00"     # Approx KL
C_SKY = "#56B4E9"        # Clipfrac
C_GRAY = "#555555"       # Tham chiếu / nền trung tính

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "runs"
OUT = ROOT / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def read_progress(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    data = {
        k: np.array([float(r[k]) if r[k] not in (None, "") else np.nan for r in rows], dtype=float)
        for k in cols
    }
    order = np.argsort(data["global_step"], kind="mergesort")
    return {k: v[order] for k, v in data.items()}


def ema(arr, alpha=0.15):
    out = np.full_like(arr, np.nan, dtype=float)
    run = None
    for i, v in enumerate(arr):
        if np.isfinite(v):
            run = float(v) if run is None else alpha * float(v) + (1.0 - alpha) * run
        if run is not None:
            out[i] = run
    return out


def fmt_steps(x, _pos):
    if abs(x) >= 1_000_000:
        return f"{x / 1e6:.0f}M" if abs(x / 1e6 - round(x / 1e6)) < 1e-6 else f"{x / 1e6:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1e3:.0f}k"
    return f"{int(x)}"


def style_ax(ax, title=None, xlabel=None, ylabel=None, fmt_x=True):
    if title:
        ax.set_title(title, pad=4)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=2)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2)
    if fmt_x:
        ax.xaxis.set_major_formatter(FuncFormatter(fmt_steps))
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_pdf(fig, name: str):
    p = OUT / name
    fig.savefig(p, format="pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {p.relative_to(ROOT)} ({p.stat().st_size / 1024:.1f} KB)")


# ==============================================================================
# FIGURE 1: TEASER — (a) Chẩn đoán & Khắc phục 7-1 | (b) Pareto Đánh đổi
# ==============================================================================
def make_fig1_teaser():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.8, 2.55), gridspec_kw={"width_ratios": [1.15, 1.0]}, constrained_layout=True)

    # Panel (a): Grouped comparison on World 7-1 (Stage 1 8M vs Stage 2 9.51M)
    cats = ["Eval chuẩn\n(--seed 2026)", "Đa seed\n(100 lượt, 5 seed)", "Tỉ lệ vượt\nHammer Bros"]
    s1_vals = [0.70, 0.82, 17 / 20]  # Stage 1 8M: 3/20 died at Hammer Bros (2168,2181,2201) -> 17/20 = 0.85
    s1_errs = [0.0, 0.112, 0.0]
    s2_vals = [0.80, 0.86, 20 / 20]  # Stage 2 9.51M: 0/20 died at Hammer Bros -> 1.00
    s2_errs = [0.0, 0.058, 0.0]

    x = np.arange(len(cats))
    w = 0.34
    b1 = ax_a.bar(x - w / 2, s1_vals, width=w, color=C_VERM, alpha=0.88, edgecolor="white",
                  label="Pha 1 (8M, ent=0.01, H=0.75)")
    b2 = ax_a.bar(x + w / 2, s2_vals, width=w, color=C_BLUE, alpha=0.95, edgecolor="white",
                  label="Pha 2 (9.51M, ent=0.002, H=0.52)")
    ax_a.errorbar(x[1] - w / 2, s1_vals[1], yerr=s1_errs[1], fmt="none", ecolor="#222", elinewidth=1.0, capsize=3)
    ax_a.errorbar(x[1] + w / 2, s2_vals[1], yerr=s2_errs[1], fmt="none", ecolor="#222", elinewidth=1.0, capsize=3)

    ax_a.axhline(0.80, color=C_GREEN, linestyle=":", linewidth=1.2)
    ax_a.text(-0.46, 0.812, "Mốc 0.80", color=C_GREEN, fontsize=6.8, ha="left", va="bottom",
              fontweight="bold", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    for rect, val, err in zip(b1, s1_vals, s1_errs):
        y_off = err + 0.022 if err > 0 else 0.018
        ax_a.text(rect.get_x() + rect.get_width() / 2, val + y_off, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=6.9, color=C_VERM, fontweight="bold")
    for rect, val, err in zip(b2, s2_vals, s2_errs):
        y_off = err + 0.022 if err > 0 else 0.018
        ax_a.text(rect.get_x() + rect.get_width() / 2, val + y_off, f"{val:.2f}",
                  ha="center", va="bottom", fontsize=6.9, color=C_BLUE, fontweight="bold")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(cats, fontsize=7.2)
    ax_a.set_ylim(0.45, 1.14)
    style_ax(ax_a, title="(a) Khắc phục phình Entropy trên màn 7-1", ylabel="Tỉ lệ thành công (↑)", fmt_x=False)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ddd", fontsize=6.5)

    # Panel (b): Trade-off Official Flag Rate vs Cumulative Steps (M)
    ax_b.axhspan(0.80, 1.07, xmin=0.0, xmax=10.0 / 10.8, color=C_GREEN, alpha=0.09)
    ax_b.axhline(0.80, color=C_GREEN, linestyle=":", linewidth=1.1)
    ax_b.axvline(10.0, color=C_GRAY, linestyle="--", linewidth=0.9)
    ax_b.text(10.12, 0.615, "Giới hạn 10M", rotation=90, fontsize=6.5, color=C_GRAY, ha="left", va="bottom")

    pts = [
        (4.00, 1.00, "1-1 (từ đầu)\n1.00 @ 4.0M", C_BLUE, "o", (-0.55, -0.092)),
        (6.00, 1.00, "4-1 (Transfer)\n1.00 @ 6.0M", C_GREEN, "s", (0.28, -0.068)),
        (8.00, 0.70, "7-1 Pha 1\n0.70 @ 8.0M", C_VERM, "X", (-2.65, -0.035)),
        (9.51, 0.80, "7-1 Pha 2 (Nộp)\n0.80 @ 9.51M", C_BLUE, "*", (-3.15, 0.042)),
    ]
    # Arrow showing Stage 1 -> Stage 2 jump
    ax_b.annotate("", xy=(9.42, 0.793), xytext=(8.09, 0.708),
                  arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.35, ls="--"))
    ax_b.text(8.05, 0.768, "+0.10", fontsize=7.0, color=C_BLUE, fontweight="bold")

    for sx, fy, lbl, col, mk, (dx, dy) in pts:
        ms = 115 if mk == "*" else 52
        ax_b.scatter([sx], [fy], color=col, marker=mk, s=ms, zorder=5, edgecolors="white", linewidths=0.7)
        ax_b.text(sx + dx, fy + dy, lbl, fontsize=6.5, color="#1f2937",
                  fontweight="bold" if "Nộp" in lbl or "1.00" in lbl else "normal")

    ax_b.annotate("Tốt hơn", xy=(1.0, 1.02), xytext=(1.5, 0.88),
                  arrowprops=dict(arrowstyle="->", color=C_GREEN, lw=1.2),
                  fontsize=7.0, fontweight="bold", color=C_GREEN)

    ax_b.set_xlim(0.5, 10.8)
    ax_b.set_ylim(0.60, 1.07)
    style_ax(ax_b, title="(b) Đánh đổi Tỉ lệ chạm cờ vs. Ngân sách bước",
             xlabel="Tổng bước huấn luyện (triệu bước, ↓)", ylabel="Flag Rate chuẩn (--seed 2026, ↑)", fmt_x=False)

    save_pdf(fig, "fig1_teaser.pdf")


# ==============================================================================
# FIGURE 2: ARCHITECTURE & PIPELINE DIAGRAM (Vector)
# ==============================================================================
def make_fig2_architecture():
    fig, ax = plt.subplots(figsize=(6.8, 2.35))
    ax.set_xlim(0, 130)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f8fafc", ec="#334155", lw=0.9, ls="-", title_c="#0f172a"):
        r = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
            facecolor=fc, edgecolor=ec, linewidth=lw, linestyle=ls, zorder=3
        )
        ax.add_patch(r)
        ax.text(x + w / 2, y + h - 1.3, title, ha="center", va="top", fontsize=5.8, fontweight="bold", color=title_c, zorder=4)
        ax.plot([x + 1.0, x + w - 1.0], [y + h - 4.8, y + h - 4.8], color=ec, lw=0.45, alpha=0.45, zorder=4)
        ax.text(x + w / 2, y + (h - 4.8) / 2, body, ha="center", va="center", fontsize=5.2, color="#1e293b", linespacing=1.35, zorder=4)

    def arrow(x1, y1, x2, y2, lbl=None, dy=1.0, col="#334155"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", color=col, lw=1.0), zorder=5)
        if lbl:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, lbl, ha="center", va="bottom", fontsize=5.1, color="#0f172a")

    # Column 1: Environment
    box(0.5, 16.5, 19.5, 22.5, "1. Môi trường NES",
        "mo-supermario-v0\nRGB 240×256×3\nDeathPenaltyFix\nXPosWrapFix\nButtonActions (7)",
        fc="#f1f5f9", ec="#475569")

    # Column 2: Shared Preprocessing + Train-only dashed block
    box(23.0, 22.0, 22.5, 17.0, "2. Tiền xử lý chung",
        "FrameSkip(4, max-2)\nScalarizeReward(w)\nGrayResize(84×84)\nFrameStack(4)",
        fc="#eff6ff", ec=C_BLUE, title_c=C_BLUE)

    box(23.0, 4.2, 22.5, 14.5, "Chỉ lúc học (not eval)",
        "NoProgressTruncation\nMarioRewardShaping\nRandomNoopStart(30)\nNormalizeReward(γ)",
        fc="#fffbeb", ec=C_VERM, ls="--", lw=1.0, title_c=C_VERM)

    ax.annotate("", xy=(34.25, 21.7), xytext=(34.25, 19.0),
                arrowprops=dict(arrowstyle="<->", color=C_VERM, lw=0.95, ls="--"), zorder=5)

    # Column 3: Shared Nature-CNN Backbone
    box(54.5, 9.0, 35.0, 29.0, "3. Thân Nature-CNN (1,684,128 params)",
        "Chuẩn hóa: obs / 255.0 ∈ [0, 1]\n"
        "Conv 4→32 (8×8, s=4) + ReLU → 32×20×20\n"
        "Conv 32→64 (4×4, s=2) + ReLU → 64×9×9\n"
        "Conv 64→64 (3×3, s=1) + ReLU → 64×7×7\n"
        "Flatten(3136) → FC(3136→512) + ReLU\n"
        "Khởi tạo trực giao: gain = √2",
        fc="#e0f2fe", ec=C_BLUE, lw=1.1, title_c="#0c4a6e")

    # Column 4: Actor & Critic Heads
    box(97.5, 24.5, 32.0, 14.5, "4a. Đầu Actor (3,591 params)",
        "Linear(512 → 7), gain = 0.01\n"
        "π_θ(a_t | s_t) = Categorical(logits)\n"
        "L^CLIP(ε=0.2) − c₂ H(π_θ)",
        fc="#ecfdf5", ec=C_GREEN, lw=1.0, title_c="#065f46")

    box(97.5, 6.5, 32.0, 14.5, "4b. Đầu Critic (513 params)",
        "Linear(512 → 1), gain = 1.0 → V_θ(s_t)\n"
        "GAE(γ, λ=0.95): tách term vs. trunc\n"
        "L^VF = mean((V_θ − R_t)²), c₁ = 0.25",
        fc="#fdf2f8", ec=C_PURPLE, lw=1.0, title_c="#831843")

    # Connecting arrows with clean tensor labels
    arrow(20.2, 30.5, 22.8, 30.5)
    arrow(45.7, 29.5, 54.3, 29.5, lbl="s_t\n(B,4,84,84)", dy=1.0)
    arrow(89.7, 28.0, 97.3, 31.5, lbl="h_t\n(B,512)", dy=1.0, col=C_GREEN)
    arrow(89.7, 18.0, 97.3, 14.0, lbl="h_t\n(B,512)", dy=-4.8, col=C_PURPLE)

    ax.text(65, 0.5,
            "Tổng ActorCritic(7, 4): 1,688,232 tham số (FP32 ≈ 6.44 MB)  |  "
            "Lô thu: B = T×N = 4,096  |  Cập nhật: 4 epoch × 8 mini-batch (32 bước/lô)",
            ha="center", va="bottom", fontsize=5.8, color="#334155", fontweight="bold")

    save_pdf(fig, "fig2_architecture.pdf")


# ==============================================================================
# FIGURES 3, 4, 5: 6 LEARNING CURVES (1-1, 4-1, 7-1 2-Stage)
# ==============================================================================
def make_learning_curves(data, out_name: str, vline_step=None, vline_label=None):
    metrics = [
        ("ep_return", "Lợi tức thô (Episode Return ↑)", C_BLUE),
        ("x_pos", "Tọa độ trung bình x_pos (↑)", C_GREEN),
        ("flag_rate", "Tỉ lệ chạm cờ 100 lượt (↑)", C_VERM),
        ("entropy", "Entropy chính sách H(π)", C_PURPLE),
        ("approx_kl", "Approx KL (↓ ổn định)", C_ORANGE),
        ("clipfrac", "Tỉ lệ cắt Clipfrac", C_SKY),
    ]
    steps = data["global_step"]
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 2.82), constrained_layout=True)

    for ax, (col, title, color) in zip(axes.flat, metrics):
        vals = data[col]
        mask = np.isfinite(steps) & np.isfinite(vals)
        ax.plot(steps[mask], vals[mask], color=color, alpha=0.22, linewidth=0.7)
        ax.plot(steps[mask], ema(vals[mask], 0.15), color=color, linewidth=1.45, label="EMA (0.15)")
        if col == "flag_rate":
            ax.axhline(0.80, color="#222222", linestyle=":", linewidth=0.95, label="Mốc 0.8")
            ax.set_ylim(-0.04, 1.06)
            ax.legend(loc="lower right", frameon=False, fontsize=6.5)
        if vline_step is not None:
            ax.axvline(vline_step, color="#1f2937", linestyle="--", linewidth=0.85, alpha=0.85)
            if col == "entropy" and vline_label:
                yl = ax.get_ylim()
                ax.text(vline_step - 1.5e5, yl[0] + 0.18 * (yl[1] - yl[0]), vline_label,
                        fontsize=6.3, color="#1f2937", ha="right", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#cbd5e1", alpha=0.9))
        style_ax(ax, title=title, xlabel="Bước tác tử" if ax in axes[1] else None)

    save_pdf(fig, out_name)


def build_71_combined():
    d1 = read_progress(RUNS / "7-1_seed1" / "progress.csv")
    d2 = read_progress(RUNS / "7-1_stage2_finetune" / "progress.csv")
    offset = float(d1["global_step"][-1])
    d2 = dict(d2)
    d2["global_step"] = d2["global_step"] + offset
    comb = {k: np.concatenate([d1[k], d2[k]]) for k in d1 if k in d2}
    return comb, offset


# ==============================================================================
# FIGURE 6: ABLATION STUDY (gamma=0.90 vs gamma=0.99, 2 seeds)
# ==============================================================================
def make_fig6_ablation():
    a = [read_progress(RUNS / f"ablation/gamma090_seed{s}/progress.csv") for s in (1, 2)]
    b = [read_progress(RUNS / f"ablation/gamma099_seed{s}/progress.csv") for s in (1, 2)]
    grid = np.linspace(4096, 3_002_368, 220)

    def agg(ds, col):
        arr = np.stack([np.interp(grid, d["global_step"], d[col]) for d in ds], axis=0)
        return arr.mean(axis=0), arr.std(axis=0)

    metrics = [
        ("ep_return", "Lợi tức thô (Episode Return ↑)"),
        ("x_pos", "Tọa độ trung bình x_pos (↑)"),
        ("flag_rate", "Tỉ lệ chạm cờ 100 lượt (↑)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.22), constrained_layout=True)

    for ax, (col, title) in zip(axes, metrics):
        ma, sa = agg(a, col)
        mb, sb = agg(b, col)
        ax.plot(grid, ma, color=C_BLUE, linestyle="-", linewidth=1.6, label="γ = 0.90 (mặc định, n=2)")
        ax.fill_between(grid, ma - sa, ma + sa, color=C_BLUE, alpha=0.18)
        ax.plot(grid, mb, color=C_VERM, linestyle="--", linewidth=1.6, label="γ = 0.99 (n=2)")
        ax.fill_between(grid, mb - sb, mb + sb, color=C_VERM, alpha=0.18)

        if col == "flag_rate":
            ax.axhline(0.80, color="#222222", linestyle=":", linewidth=0.95)
            ax.set_ylim(-0.04, 1.06)
            ax.annotate("Sớm hơn ~31%\n(0.78M vs 1.13M)", xy=(780_000, 0.80), xytext=(1_150_000, 0.42),
                        arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.0),
                        fontsize=6.6, color=C_BLUE, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C_BLUE, alpha=0.9))

        style_ax(ax, title=title, xlabel="Bước tác tử")
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e2e8f0", fontsize=6.4)

    save_pdf(fig, "fig6_ablation_gamma.pdf")


# ==============================================================================
# FIGURE 7: FAILURE ANALYSIS (1-1 & 7-1 Side-by-Side)
# ==============================================================================
def collect_x(json_paths, eval_csvs):
    xs, frs = [], []
    for jp in json_paths:
        d = json.loads(jp.read_text(encoding="utf-8"))
        vals = [float(v) for v in d["x_pos"]]
        xs.extend(vals)
        frs.append((float(d["flag_rate"]), len(vals)))
    for cp in eval_csvs:
        with cp.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                vals = [float(t) for t in row["x_pos"].split()]
                xs.extend(vals)
                frs.append((float(row["flag_rate"]), len(vals)))
    w_fr = sum(f * w for f, w in frs) / sum(w for _, w in frs)
    return np.array(xs), w_fr


def make_fig7_failures():
    x11, fr11 = collect_x([RUNS / "1-1_seed1/ket-qua-1-1.json"], [RUNS / "1-1_seed1/evals.csv"])
    x71, fr71 = collect_x(
        [RUNS / "7-1_stage2_finetune/ket-qua-7-1.json"],
        [RUNS / "7-1_seed1/evals.csv", RUNS / "7-1_stage2_finetune/evals.csv"],
    )

    fig = plt.figure(figsize=(6.8, 2.55), constrained_layout=True)
    outer = fig.add_gridspec(1, 2, wspace=0.06)

    panels = [
        (outer[0], x11, fr11, "(a) Màn 1-1 (N=60)",
         [(950, "Ống cống\n(950)"), (1790, "Hố/thang\n(1790)"), (2475, "Thang cuối\n(2475)"), (3161, "Cờ\n(3161)")]),
        (outer[1], x71, fr71, "(b) Màn 7-1 (N=120)",
         [(867, "Đạn/quái\n(867)"), (1388, "Bệ pháo\n(1388)"), (2180, "Hammer Bros\n(2180)"), (2857, "Cờ\n(2857)")]),
    ]

    for gs_cell, xvals, fr, title, landmarks in panels:
        inner = gs_cell.subgridspec(2, 1, height_ratios=[3.6, 1.0], hspace=0.06)
        ax_h = fig.add_subplot(inner[0])
        ax_s = fig.add_subplot(inner[1], sharex=ax_h)

        bins = 20
        ax_h.hist(xvals, bins=bins, color=C_BLUE, alpha=0.82, edgecolor="white", linewidth=0.5)
        mx, md = float(xvals.mean()), float(np.median(xvals))
        ax_h.axvline(mx, color=C_VERM, linestyle="-", linewidth=1.3, label=f"Mean={mx:.0f}")
        ax_h.axvline(md, color=C_ORANGE, linestyle="--", linewidth=1.2, label=f"Median={md:.0f}")

        ymax = ax_h.get_ylim()[1]
        ax_h.set_ylim(0, ymax * 1.28)
        xmin, xmax = ax_h.get_xlim()
        xspan = max(xmax - xmin, 1.0)
        for idx, (lx, lbl) in enumerate(landmarks):
            ax_h.axvline(lx, color=C_GRAY, linestyle=":", linewidth=0.85)
            near_r = (lx - xmin) / xspan > 0.80
            ha = "right" if near_r else "left"
            y_frac = 0.78 - 0.19 * (idx % 2)
            ax_h.text(lx + (-25 if near_r else 25), ymax * y_frac, lbl,
                      fontsize=6.3, color="#1f2937", ha=ha, va="top",
                      bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.82))

        ax_h.legend(loc="upper left", ncol=2, fontsize=6.4,
                    frameon=True, facecolor="white", edgecolor="#cbd5e1", handlelength=1.4, columnspacing=0.8)
        style_ax(ax_h, title=f"{title} (flag={fr:.1%})", ylabel="Số lượt", fmt_x=False)
        plt.setp(ax_h.get_xticklabels(), visible=False)

        rng = np.random.default_rng(0)
        ax_s.scatter(xvals, rng.uniform(-0.25, 0.25, size=len(xvals)), color=C_BLUE, alpha=0.65, s=12, edgecolors="none")
        ax_s.set_yticks([])
        ax_s.set_ylim(-0.55, 0.55)
        style_ax(ax_s, xlabel="Tọa độ x_pos cuối lượt", fmt_x=False)

    save_pdf(fig, "fig7_failures.pdf")


def main():
    make_fig1_teaser()
    make_fig2_architecture()
    make_learning_curves(read_progress(RUNS / "1-1_seed1/progress.csv"), "fig3_curves_1-1.pdf")
    make_learning_curves(read_progress(RUNS / "4-1_seed1/progress.csv"), "fig4_curves_4-1.pdf")
    d71, off = build_71_combined()
    make_learning_curves(d71, "fig5_curves_7-1.pdf", vline_step=off, vline_label="Pha 2 (8M)\nlr=6e-5, ent=0.002")
    make_fig6_ablation()
    make_fig7_failures()


if __name__ == "__main__":
    main()
