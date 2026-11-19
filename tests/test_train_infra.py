"""Kiểm tra hạ tầng Phân hệ 4: đa nền tảng vec env, init_ckpt/resume, configs/*.yaml, notebook Colab/Kaggle."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

import mario_ppo.model as model_mod
import mario_ppo.ppo as ppo_mod
import mario_ppo.train as train_mod


def test_make_vec_env_selects_spawn_on_win32_and_forkserver_on_linux(monkeypatch):
    import mario_ppo.env  # đảm bảo mo_gymnasium nạp xong trước khi monkeypatch gym.vector

    calls = []

    def fake_sync(fns, autoreset_mode=None):
        calls.append(("sync", len(fns), None, autoreset_mode))
        return "sync_env"

    def fake_async(fns, context=None, autoreset_mode=None):
        calls.append(("async", len(fns), context, autoreset_mode))
        return "async_env"

    monkeypatch.setattr(gym.vector, "SyncVectorEnv", fake_sync)
    monkeypatch.setattr(gym.vector, "AsyncVectorEnv", fake_async)

    assert train_mod._make_vec_env({"actions": "simple"}, "1-1", 1, seed=1) == "sync_env"

    monkeypatch.setattr(sys, "platform", "win32")
    assert train_mod._make_vec_env({"actions": "simple"}, "1-1", 2, seed=1) == "async_env"

    monkeypatch.setattr(sys, "platform", "linux")
    assert train_mod._make_vec_env({"actions": "simple"}, "1-1", 4, seed=1) == "async_env"

    mode = gym.vector.AutoresetMode.SAME_STEP
    assert calls == [
        ("sync", 1, None, mode),
        ("async", 2, "spawn", mode),
        ("async", 4, "forkserver", mode),
    ]


class _StubActorCritic(nn.Module):
    def __init__(self, n_actions, in_channels=4, **kwargs):
        super().__init__()
        self.net = nn.Linear(in_channels, n_actions)
        self.val = nn.Linear(in_channels, 1)

    def forward(self, obs):
        feat = obs.float().mean(dim=(-2, -1))
        return self.net(feat), self.val(feat).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        logits, value = self(obs)
        action = torch.zeros(obs.shape[0], dtype=torch.long)
        logp = torch.log_softmax(logits, dim=-1)[:, 0]
        return action, logp.cpu(), value.cpu()


class _StubVecEnv:
    def __init__(self, num_envs, obs_shape=(4, 84, 84)):
        self.num_envs = num_envs
        self.single_observation_space = gym.spaces.Box(0, 255, obs_shape, dtype=np.uint8)

    def reset(self, seed=None):
        return np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.uint8), {}

    def step(self, actions):
        obs = np.zeros((self.num_envs, *self.single_observation_space.shape), dtype=np.uint8)
        rew = np.ones(self.num_envs, dtype=np.float32)
        term = np.zeros(self.num_envs, dtype=bool)
        trunc = np.zeros(self.num_envs, dtype=bool)
        return obs, rew, term, trunc, {}

    def close(self):
        pass


def _patch_train_stubs(monkeypatch):
    monkeypatch.setattr(model_mod, "ActorCritic", _StubActorCritic)
    monkeypatch.setattr(
        ppo_mod,
        "compute_gae",
        lambda rewards, values, term, trunc, fv, last_v, gamma, lam: (
            torch.ones_like(rewards),
            torch.ones_like(rewards),
        ),
    )

    def fake_ppo_update(model, opt, batch, cfg):
        if cfg.update_epochs > 0:
            logits, val = model(batch["obs"])
            loss = logits.pow(2).mean() + val.pow(2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 1.0,
            "approx_kl": 0.0,
            "clipfrac": 0.0,
            "minibatch_updates": int(cfg.update_epochs),
        }

    monkeypatch.setattr(ppo_mod, "ppo_update", fake_ppo_update)
    monkeypatch.setattr(
        train_mod,
        "_make_vec_env",
        lambda env_cfg, level, num_envs, seed: _StubVecEnv(num_envs),
    )


def _infra_cfg(run_dir: Path, **overrides):
    cfg = OmegaConf.load("configs/default.yaml")
    dot = [
        f"run_dir={run_dir}",
        "device=cpu",
        "algo.num_envs=2",
        "algo.num_steps=4",
        "algo.num_minibatches=2",
        "algo.update_epochs=1",
        "algo.total_steps=8",
        "checkpoint_every=8",
        "eval_episodes=0",
        "video=false",
    ]
    for k, v in overrides.items():
        dot.append(f"{k}={v}")
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(dot))


def test_train_init_ckpt_transfers_weights_with_fresh_optimizer_at_step_zero(tmp_path, monkeypatch):
    _patch_train_stubs(monkeypatch)

    # Tạo checkpoint nguồn có trọng số đặc trưng, optimizer state đã có dữ liệu và global_step=100
    src_run = train_mod.train(_infra_cfg(tmp_path / "src", seed=1))
    src_ckpt = sorted((src_run / "checkpoints").glob("*.pt"))[-1]
    ck_src = torch.load(src_ckpt, weights_only=False)
    assert len(ck_src["optimizer"]["state"]) > 0
    ck_src["model"]["net.weight"].fill_(7.5)
    ck_src["global_step"] = 100
    torch.save(ck_src, src_ckpt)

    # Dùng update_epochs=0 để kiểm tra trạng thái mô hình/optimizer ngay sau lúc nạp init_ckpt
    cfg2 = _infra_cfg(tmp_path / "transfer", seed=99, init_ckpt=str(src_ckpt), **{"algo.update_epochs": 0})
    run2 = train_mod.train(cfg2)
    ck2 = torch.load(sorted((run2 / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    assert ck2["global_step"] == 8
    assert torch.allclose(ck2["model"]["net.weight"], torch.full_like(ck2["model"]["net.weight"], 7.5))
    assert ck2["optimizer"]["state"] == {}


def test_train_resume_takes_precedence_over_init_ckpt(tmp_path, monkeypatch):
    _patch_train_stubs(monkeypatch)

    src_run = train_mod.train(_infra_cfg(tmp_path / "src", seed=1))
    resume_ckpt = sorted((src_run / "checkpoints").glob("*.pt"))[-1]
    ck_res = torch.load(resume_ckpt, weights_only=False)
    ck_res["model"]["net.weight"].fill_(3.25)
    ck_res["global_step"] = 40
    torch.save(ck_res, resume_ckpt)

    init_ckpt = tmp_path / "other_init.pt"
    ck_init = torch.load(resume_ckpt, weights_only=False)
    ck_init["model"]["net.weight"].fill_(9.0)
    ck_init["global_step"] = 200
    torch.save(ck_init, init_ckpt)

    cfg = _infra_cfg(
        tmp_path / "resumed",
        resume=str(resume_ckpt),
        init_ckpt=str(init_ckpt),
        **{"algo.total_steps": 48, "algo.update_epochs": 0},
    )
    run = train_mod.train(cfg)
    ck_out = torch.load(sorted((run / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    assert ck_out["global_step"] == 48
    assert torch.allclose(ck_out["model"]["net.weight"], torch.full_like(ck_out["model"]["net.weight"], 3.25))
    assert len(ck_out["optimizer"]["state"]) > 0


def test_default_yaml_preserves_defaults_and_has_init_ckpt():
    cfg = OmegaConf.load("configs/default.yaml")
    assert "init_ckpt" in cfg and cfg.init_ckpt is None
    assert cfg.resume is None
    assert cfg.level == "1-1"
    assert cfg.algo.num_envs == 16
    assert cfg.algo.num_steps == 256
    assert cfg.algo.total_steps == 5_000_000
    assert cfg.algo.gamma == pytest.approx(0.9)
    assert cfg.env.actions == "simple"
    for trick in ("max_no_progress_steps", "flag_bonus", "stuck_penalty"):
        assert trick not in cfg.env


def test_stage_and_ablation_configs_complete_and_match_spec():
    default_cfg = OmegaConf.load("configs/default.yaml")
    expected = {
        "configs/1-1.yaml": dict(level="1-1", total_steps=4_000_000, gamma=0.9, max_no_progress=180, flag_bonus=5.0),
        "configs/1-4.yaml": dict(level="1-4", total_steps=6_000_000, gamma=0.95, max_no_progress=220, flag_bonus=8.0),
        "configs/4-1.yaml": dict(level="4-1", total_steps=6_000_000, gamma=0.95, max_no_progress=180, flag_bonus=6.0),
        "configs/7-1.yaml": dict(level="7-1", total_steps=8_000_000, gamma=0.95, max_no_progress=200, flag_bonus=8.0),
        "configs/ablation_gamma099.yaml": dict(
            level="1-1", total_steps=3_000_000, gamma=0.99, max_no_progress=180, flag_bonus=5.0
        ),
        "configs/ablation_notricks.yaml": dict(
            level="1-1", total_steps=3_000_000, gamma=0.9, max_no_progress=0, flag_bonus=0.0
        ),
    }

    for path_str, spec in expected.items():
        path = Path(path_str)
        assert path.exists(), f"Missing config {path_str}"
        cfg = OmegaConf.load(path)
        assert set(default_cfg.keys()) <= set(cfg.keys())
        assert set(default_cfg.env.keys()) <= set(cfg.env.keys())
        assert set(default_cfg.algo.keys()) <= set(cfg.algo.keys())

        assert cfg.level == spec["level"]
        assert cfg.init_ckpt is None and cfg.resume is None
        assert cfg.algo.num_envs == 8 and cfg.algo.num_steps == 512
        assert cfg.algo.num_envs * cfg.algo.num_steps == 4096
        assert cfg.algo.total_steps == spec["total_steps"]
        assert cfg.algo.gamma == pytest.approx(spec["gamma"])
        assert cfg.env.actions == "simple"
        assert cfg.env.max_no_progress_steps == spec["max_no_progress"]
        assert cfg.env.flag_bonus == pytest.approx(spec["flag_bonus"])


def test_colab_kaggle_notebook_valid_json_and_python_cells():
    nb_path = Path("notebooks/train_colab_kaggle.ipynb")
    assert nb_path.exists()
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    cells = nb["cells"]
    assert any(c["cell_type"] == "markdown" for c in cells)
    code_cells = [c for c in cells if c["cell_type"] == "code"]
    assert len(code_cells) >= 5

    joined = []
    for i, cell in enumerate(code_cells):
        src = "".join(cell["source"])
        ast.parse(src, filename=f"cell_{i}")
        joined.append(src)
    full_src = "\n".join(joined)
    for token in (
        "google.colab",
        "/kaggle/working",
        "step_*.pt",
        "mario_ppo.train",
        "mario_ppo.evaluate",
        "2026",
        "scripts.plot_report",
    ):
        assert token in full_src, f"Missing {token!r} in notebook"
