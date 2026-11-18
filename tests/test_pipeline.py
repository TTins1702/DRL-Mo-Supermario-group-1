"""Kiểm tra đầu-cuối trên trình giả lập thật: make_env, một lượt huấn luyện tí hon, script đánh giá.

Chạy chậm hơn các bài kiểm tra khác (vài chục giây); cần các phần tự cài đặt đã xong.
"""
import json

import numpy as np
import torch
from omegaconf import OmegaConf

from mario_ppo.env import make_env, parse_level


def test_parse_level():
    assert parse_level("1-1") == (1, 1)
    assert parse_level("8-4") == (8, 4)


def test_make_env_observation_and_reward():
    env = make_env("1-1", seed=0)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
        assert env.action_space.n == 7
        obs, reward, terminated, truncated, info = env.step(1)  # right
        assert isinstance(float(reward), float)
        assert "x_pos" in info and "flag_get" in info
    finally:
        env.close()


def test_eval_env_has_raw_reward_and_no_noops():
    env = make_env("1-1", seed=0, eval_mode=True, noop_max=30)
    try:
        env.reset(seed=0)
        _, r, *_ = env.step(1)
        assert not hasattr(env, "return_rms")  # không chuẩn hóa phần thưởng khi đánh giá
    finally:
        env.close()


def _tiny_cfg(tmp_path):
    cfg = OmegaConf.load("configs/default.yaml")
    over = OmegaConf.from_dotlist([
        f"run_dir={tmp_path}/run", "algo.num_envs=2", "algo.num_steps=16", "algo.num_minibatches=2",
        "algo.update_epochs=1", "algo.total_steps=64", "checkpoint_every=32", "eval_episodes=1",
        "eval_max_steps=20", "video=false",
    ])
    return OmegaConf.merge(cfg, over)


def test_train_smoke_writes_progress_and_checkpoints(tmp_path):
    from mario_ppo.train import train
    run = train(_tiny_cfg(tmp_path))
    assert (run / "progress.csv").exists()
    ckpts = sorted((run / "checkpoints").glob("*.pt"))
    assert ckpts, "no checkpoint written"
    ck = torch.load(ckpts[-1], weights_only=False)
    assert {"model", "config", "global_step"} <= set(ck)
    assert ck["global_step"] >= 64


def test_evaluate_checkpoint_reports_flag_rate(tmp_path):
    from mario_ppo.evaluate import evaluate_checkpoint
    from mario_ppo.train import train
    run = train(_tiny_cfg(tmp_path))
    ckpt = sorted((run / "checkpoints").glob("*.pt"))[-1]
    out = tmp_path / "eval.json"
    res = evaluate_checkpoint(ckpt, level="1-1", episodes=2, seed=2026, max_steps=30, out=out)
    assert set(res) >= {"level", "episodes", "flag_rate", "x_pos", "noop_steps", "solved", "checkpoint"}
    assert res["episodes"] == 2 and len(res["x_pos"]) == 2
    assert 0.0 <= res["flag_rate"] <= 1.0
    assert json.loads(out.read_text())["flag_rate"] == res["flag_rate"]
    # cùng seed -> cùng các điểm xuất phát
    res2 = evaluate_checkpoint(ckpt, level="1-1", episodes=2, seed=2026, max_steps=30)
    assert res2["noop_steps"] == res["noop_steps"]
