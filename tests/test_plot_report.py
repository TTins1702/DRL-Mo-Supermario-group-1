"""Kiểm tra script vẽ biểu đồ báo cáo trong scripts/plot_report.py."""
import json
from pathlib import Path

import pytest

from scripts.plot_report import (
    main as plot_report_main,
    plot_ablation_comparison,
    plot_failure_x_distribution,
    plot_learning_curves,
)


def _write_dummy_progress(path: Path, offset: float = 0.0, steps=(4096, 8192, 12288)):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "global_step,sps,lr,policy_loss,value_loss,entropy,approx_kl,clipfrac,"
        "minibatch_updates,episodes,ep_return,ep_length,x_pos,flag_rate"
    ]
    for i, step in enumerate(steps):
        ep_ret = "nan" if i == 0 and offset > 0 else f"{15.0 + i * 15.0 + offset:.2f}"
        x_val = 850.0 + i * 1100.0 + offset * 10.0
        flag = min(1.0, 0.4 * i)
        lines.append(
            f"{step},300.0,{0.00025 - i * 5e-5:.6f},{-0.01 - 0.005 * i:.4f},"
            f"{1.2 - 0.2 * i:.3f},{1.8 - 0.15 * i:.3f},{0.005 + 0.002 * i:.4f},"
            f"{0.05 + 0.02 * i:.3f},32,{10 * (i + 1)},{ep_ret},{120.0 + i * 40:.1f},"
            f"{x_val:.1f},{flag:.2f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_plot_functions_create_image_files(tmp_path):
    p1 = tmp_path / "r1" / "progress.csv"
    p2 = tmp_path / "r2" / "progress.csv"
    _write_dummy_progress(p1, steps=(4096, 8192))
    _write_dummy_progress(p2, steps=(4096, 8192))

    out_curves = tmp_path / "curves.png"
    plot_learning_curves(p1, out_curves)
    assert out_curves.exists() and out_curves.stat().st_size > 0

    out_ablation = tmp_path / "ablation.png"
    plot_ablation_comparison([p1], [p2], "gamma=0.90", "gamma=0.99", out_ablation)
    assert out_ablation.exists() and out_ablation.stat().st_size > 0

    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps({"level": "1-1", "flag_rate": 0.85, "x_pos": [1200, 3161, 3161, 1800]}),
        encoding="utf-8",
    )
    out_fail = tmp_path / "failure_x.png"
    plot_failure_x_distribution([eval_json], "1-1", out_fail)
    assert out_fail.exists() and out_fail.stat().st_size > 0


def test_plot_ablation_comparison_multi_seed_and_empty_error(tmp_path):
    a1 = tmp_path / "a1" / "progress.csv"
    a2 = tmp_path / "a2" / "progress.csv"
    b1 = tmp_path / "b1" / "progress.csv"
    b2 = tmp_path / "b2" / "progress.csv"
    _write_dummy_progress(a1, offset=0.0, steps=(4096, 8192, 12288))
    _write_dummy_progress(a2, offset=3.0, steps=(4096, 12288))
    _write_dummy_progress(b1, offset=-2.0, steps=(4096, 8192, 12288))
    _write_dummy_progress(b2, offset=1.0, steps=(4096, 8192, 12288))

    out = tmp_path / "nested" / "ablation_2seed.png"
    ret = plot_ablation_comparison([a1, a2], [b1, b2], "gamma=0.90", "gamma=0.99", out)
    assert ret == out and out.exists() and out.stat().st_size > 0

    with pytest.raises(ValueError):
        plot_ablation_comparison([], [b1], "gamma=0.90", "gamma=0.99", tmp_path / "bad.png")


def test_plot_failure_x_distribution_with_evals_csv_and_empty_error(tmp_path):
    eval_json = tmp_path / "eval_1_4.json"
    eval_json.write_text(
        json.dumps({"level": "1-4", "flag_rate": 0.5, "x_pos": [950, 2240]}),
        encoding="utf-8",
    )
    evals_csv = tmp_path / "evals.csv"
    evals_csv.write_text(
        "global_step,flag_rate,x_pos\n"
        "500000,0.4,1200 1800 2240 950 1500\n",
        encoding="utf-8",
    )
    out = tmp_path / "fail_1_4.png"
    plot_failure_x_distribution([eval_json], "1-4", out, eval_csvs=[evals_csv])
    assert out.exists() and out.stat().st_size > 0

    empty_json = tmp_path / "empty.json"
    empty_json.write_text(json.dumps({"level": "1-1", "flag_rate": 0.0, "x_pos": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        plot_failure_x_distribution([empty_json], "1-1", tmp_path / "err1.png")
    with pytest.raises(ValueError):
        plot_failure_x_distribution([], "1-1", tmp_path / "err2.png")


def test_plot_report_cli_subcommands(tmp_path):
    p1 = tmp_path / "s1" / "progress.csv"
    p2 = tmp_path / "s2" / "progress.csv"
    _write_dummy_progress(p1, offset=0.0)
    _write_dummy_progress(p2, offset=2.0)

    out_c = tmp_path / "cli_curves.png"
    plot_report_main(["curves", "--csv", str(p1), "--out", str(out_c)])
    assert out_c.exists() and out_c.stat().st_size > 0

    out_a = tmp_path / "cli_ablation.png"
    plot_report_main([
        "ablation",
        "--a", str(p1),
        "--b", str(p2),
        "--label-a", "A",
        "--label-b", "B",
        "--out", str(out_a),
    ])
    assert out_a.exists() and out_a.stat().st_size > 0

    eval_json = tmp_path / "cli_eval.json"
    eval_json.write_text(
        json.dumps({"level": "7-1", "flag_rate": 0.75, "x_pos": [1400, 2900, 3100, 1100]}),
        encoding="utf-8",
    )
    out_f = tmp_path / "cli_fail.png"
    plot_report_main(["failures", "--json", str(eval_json), "--level", "7-1", "--out", str(out_f)])
    assert out_f.exists() and out_f.stat().st_size > 0
