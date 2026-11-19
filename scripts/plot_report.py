"""Vẽ biểu đồ chuẩn báo cáo (300 DPI) từ progress.csv, evals.csv và ket-qua-*.json.

Ví dụ chạy từ dòng lệnh:
    python -m scripts.plot_report curves --csv runs/1-1/run1/progress.csv --out reports/curves_1-1.png
    python -m scripts.plot_report ablation --a r1/progress.csv r2/progress.csv --b b1/progress.csv b2/progress.csv --label-a "gamma=0.90" --label-b "gamma=0.99" --out reports/ablation_gamma.png
    python -m scripts.plot_report failures --json ket-qua-1-1.json --level 1-1 --out reports/failure_x_1-1.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

LEARNING_METRICS = [
    ("ep_return", "Episode Return"),
    ("x_pos", "Mean x_pos"),
    ("flag_rate", "Flag Rate (100-ep)"),
    ("entropy", "Policy Entropy"),
    ("approx_kl", "Approx KL"),
    ("clipfrac", "Clip Fraction"),
]

STAGE_LANDMARKS = {
    "1-1": [(950, "Ống cống cao"), (1400, "Hố nhảy rộng"), (3161, "Cờ đích (~3161)")],
    "1-4": [(1400, "Hồ dung nham"), (2050, "Bowser & rìu"), (2240, "Rìu đích")],
    "4-1": [(1200, "Lakitu / Spinies"), (2400, "Đoạn ống cuối"), (3161, "Cờ đích")],
    "7-1": [(1300, "Bệ pháo Bullet Bill"), (2200, "Hammer Bros"), (3161, "Cờ đích")],
}


def _to_float(raw):
    if raw is None or raw == "":
        return float("nan")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _read_progress_csv(csv_path):
    """Đọc progress.csv thành dict các mảng numpy float64 đã sắp xếp theo global_step."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Progress CSV not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows or "global_step" not in fieldnames:
        raise ValueError(f"Progress CSV is empty or missing 'global_step': {path}")

    data = {col: np.array([_to_float(r.get(col)) for r in rows], dtype=np.float64) for col in fieldnames}
    order = np.argsort(data["global_step"], kind="mergesort")
    return {k: v[order] for k, v in data.items()}


def _ema(values, alpha=0.15):
    """Tính trung bình trượt hàm mũ (EMA), bỏ qua các giá trị NaN."""
    out = np.full_like(values, np.nan, dtype=np.float64)
    running = None
    for i, val in enumerate(values):
        if np.isfinite(val):
            running = float(val) if running is None else alpha * float(val) + (1.0 - alpha) * running
        if running is not None:
            out[i] = running
    return out


def _format_steps(x, _pos):
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{int(x)}"


def _style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Global Step", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.xaxis.set_major_formatter(FuncFormatter(_format_steps))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, out_path, dpi=300):
    target = Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return target


def plot_learning_curves(progress_csv, out_path, ema_alpha=0.15, title=None):
    """Vẽ 6 đường cong học (ep_return, x_pos, flag_rate, entropy, approx_kl, clipfrac) theo global_step."""
    data = _read_progress_csv(progress_csv)
    steps = data["global_step"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e", "#17becf"]

    for ax, (col, label), color in zip(axes.flat, LEARNING_METRICS, colors):
        _style_axis(ax, label, label)
        if col not in data:
            ax.text(0.5, 0.5, f"Missing column: {col}", ha="center", va="center", transform=ax.transAxes)
            continue

        vals = data[col]
        mask = np.isfinite(steps) & np.isfinite(vals)
        if mask.any():
            ax.plot(steps[mask], vals[mask], color=color, alpha=0.25, linewidth=1.0, label="Raw")
            smooth = _ema(vals[mask], alpha=ema_alpha)
            ax.plot(steps[mask], smooth, color=color, linewidth=2.0, label=f"EMA ({ema_alpha:.2f})")
            ax.legend(loc="best", fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "No finite values", ha="center", va="center", transform=ax.transAxes)

        if col == "flag_rate":
            ax.axhline(0.8, color="black", linestyle=":", linewidth=1.2, label="Mốc 0.8")
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(title or f"Đường cong học PPO — {Path(progress_csv).parent.name}", fontsize=13, fontweight="bold")
    return _save_fig(fig, out_path)


def _aggregate_seed_curves(datasets, metric, num_points=200):
    """Nội suy nhiều seed về cùng lưới global_step và trả về (grid, mean, std)."""
    series = []
    for d in datasets:
        if metric not in d:
            continue
        mask = np.isfinite(d["global_step"]) & np.isfinite(d[metric])
        if mask.any():
            series.append((d["global_step"][mask], d[metric][mask]))

    if not series:
        return None, None, None

    lo = max(float(s[0]) for s, _ in series)
    hi = min(float(s[-1]) for s, _ in series)
    if lo >= hi:
        lo = min(float(s[0]) for s, _ in series)
        hi = max(float(s[-1]) for s, _ in series)

    grid = np.array([lo], dtype=np.float64) if lo == hi else np.linspace(lo, hi, num=num_points)
    interp = np.stack([np.interp(grid, s, v) for s, v in series], axis=0)
    return grid, np.mean(interp, axis=0), np.std(interp, axis=0)


def plot_ablation_comparison(
    csvs_a,
    csvs_b,
    label_a,
    label_b,
    out_path,
    metrics=("ep_return", "x_pos", "flag_rate"),
    title=None,
):
    """Vẽ biểu đồ so sánh 2 cấu hình (trung bình ± độ lệch chuẩn qua các seed)."""
    if not csvs_a or not csvs_b:
        raise ValueError("Both csvs_a and csvs_b must contain at least one progress.csv path.")

    data_a = [_read_progress_csv(p) for p in csvs_a]
    data_b = [_read_progress_csv(p) for p in csvs_b]
    metric_labels = dict(LEARNING_METRICS)

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.6), constrained_layout=True)
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pretty = metric_labels.get(metric, metric)
        _style_axis(ax, pretty, pretty)
        for datasets, label, color in ((data_a, label_a, "#1f77b4"), (data_b, label_b, "#d62728")):
            grid, mean, std = _aggregate_seed_curves(datasets, metric)
            if grid is None:
                continue
            ax.plot(grid, mean, color=color, linewidth=2.0, label=f"{label} (n={len(datasets)})")
            if len(datasets) > 1:
                ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.2)

        if metric == "flag_rate":
            ax.axhline(0.8, color="black", linestyle=":", linewidth=1.2)
            ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8.5, frameon=False)

    fig.suptitle(title or f"Thí nghiệm so sánh: {label_a} vs {label_b}", fontsize=13, fontweight="bold")
    return _save_fig(fig, out_path)


def _collect_eval_x_positions(eval_jsons, eval_csvs=None):
    x_values = []
    flag_rates = []

    for json_path in eval_jsons or []:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        xs = [float(x) for x in payload.get("x_pos", [])]
        x_values.extend(xs)
        if "flag_rate" in payload and xs:
            flag_rates.append((float(payload["flag_rate"]), len(xs)))

    for csv_path in eval_csvs or []:
        with Path(csv_path).open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                tokens = (row.get("x_pos") or "").split()
                xs = [float(tok) for tok in tokens]
                x_values.extend(xs)
                if row.get("flag_rate") and xs:
                    flag_rates.append((float(row["flag_rate"]), len(xs)))

    weighted_flag = None
    if flag_rates:
        total_w = sum(w for _, w in flag_rates)
        weighted_flag = sum(fr * w for fr, w in flag_rates) / total_w
    return np.asarray(x_values, dtype=np.float64), weighted_flag


def plot_failure_x_distribution(eval_jsons, level, out_path, eval_csvs=None, title=None):
    """Vẽ histogram + strip-plot phân bố tọa độ x_pos cuối lượt để phân tích điểm chết."""
    x_vals, weighted_flag = _collect_eval_x_positions(eval_jsons, eval_csvs=eval_csvs)
    if x_vals.size == 0:
        raise ValueError("No x_pos values found in provided eval_jsons / eval_csvs.")

    fig, (ax_hist, ax_strip) = plt.subplots(
        2,
        1,
        figsize=(10, 5.8),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.0]},
        constrained_layout=True,
    )

    bins = min(24, max(8, int(np.sqrt(x_vals.size) * 2)))
    ax_hist.hist(x_vals, bins=bins, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax_hist.set_ylabel("Số lượt (Episodes)", fontsize=10)
    ax_hist.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax_hist.spines["top"].set_visible(False)
    ax_hist.spines["right"].set_visible(False)

    mean_x = float(np.mean(x_vals))
    med_x = float(np.median(x_vals))
    ax_hist.axvline(mean_x, color="#d62728", linestyle="-", linewidth=1.8, label=f"Mean x = {mean_x:.0f}")
    ax_hist.axvline(med_x, color="#ff7f0e", linestyle="--", linewidth=1.5, label=f"Median x = {med_x:.0f}")

    ymax = ax_hist.get_ylim()[1]
    for idx, (lx, note) in enumerate(STAGE_LANDMARKS.get(str(level), [])):
        ax_hist.axvline(lx, color="#555555", linestyle=":", linewidth=1.2)
        ax_hist.text(
            lx,
            ymax * (0.94 - 0.09 * (idx % 2)),
            f" {note} ({lx})",
            fontsize=8,
            color="#333333",
            va="top",
        )

    summary = f"N = {x_vals.size} lượt"
    if weighted_flag is not None:
        summary += f"\nflag_rate = {weighted_flag:.2%}"
    ax_hist.text(
        0.02,
        0.95,
        summary,
        transform=ax_hist.transAxes,
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f8f9fa", "edgecolor": "#cccccc"},
    )
    ax_hist.legend(loc="upper right", fontsize=9, frameon=False)

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.25, 0.25, size=x_vals.size)
    ax_strip.scatter(x_vals, jitter, color="#4c72b0", alpha=0.75, s=28)
    ax_strip.set_yticks([])
    ax_strip.set_ylim(-0.6, 0.6)
    ax_strip.set_xlabel("Tọa độ x_pos cuối lượt", fontsize=10)
    ax_strip.grid(True, axis="x", linestyle="--", alpha=0.35)
    ax_strip.spines["top"].set_visible(False)
    ax_strip.spines["right"].set_visible(False)

    fig.suptitle(title or f"Phân bố vị trí kết thúc lượt (x_pos) — Màn {level}", fontsize=13, fontweight="bold")
    return _save_fig(fig, out_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_curves = sub.add_parser("curves", help="Vẽ 6 đường cong học từ progress.csv")
    p_curves.add_argument("--csv", required=True, help="Đường dẫn tệp progress.csv")
    p_curves.add_argument("--out", required=True, help="Đường dẫn ảnh xuất (.png/.pdf)")
    p_curves.add_argument("--ema", type=float, default=0.15, help="Hệ số làm mượt EMA")
    p_curves.add_argument("--title", default=None)

    p_ablation = sub.add_parser("ablation", help="So sánh 2 cấu hình qua nhiều seed")
    p_ablation.add_argument("--a", nargs="+", required=True, help="Các tệp progress.csv nhóm A")
    p_ablation.add_argument("--b", nargs="+", required=True, help="Các tệp progress.csv nhóm B")
    p_ablation.add_argument("--label-a", required=True)
    p_ablation.add_argument("--label-b", required=True)
    p_ablation.add_argument("--out", required=True)
    p_ablation.add_argument("--title", default=None)

    p_fail = sub.add_parser("failures", help="Vẽ phân bố x_pos cuối lượt")
    p_fail.add_argument("--json", nargs="*", default=[], help="Các tệp ket-qua-*.json")
    p_fail.add_argument("--eval-csv", nargs="*", default=[], help="Các tệp evals.csv (tùy chọn)")
    p_fail.add_argument("--level", required=True, help="Tên màn, ví dụ 1-1, 1-4, 7-1")
    p_fail.add_argument("--out", required=True)
    p_fail.add_argument("--title", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "curves":
        out = plot_learning_curves(args.csv, args.out, ema_alpha=args.ema, title=args.title)
    elif args.cmd == "ablation":
        out = plot_ablation_comparison(args.a, args.b, args.label_a, args.label_b, args.out, title=args.title)
    else:
        out = plot_failure_x_distribution(args.json, args.level, args.out, eval_csvs=args.eval_csv, title=args.title)
    print(f"Saved report figure to {out}")


if __name__ == "__main__":
    main()
