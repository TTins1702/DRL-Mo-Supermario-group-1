"""Vòng huấn luyện PPO (CÓ SẴN). Gọi các phần nhóm tự cài: make_env/wrappers, ActorCritic, compute_gae, ppo_update.

    python -m mario_ppo.train                                  # cấu hình mặc định configs/default.yaml
    python -m mario_ppo.train level=1-4 algo.total_steps=8000000 # ghi đè bất kỳ khóa nào
    python -m mario_ppo.train --config configs/my.yaml seed=2
    python -m mario_ppo.train resume=runs/1-1/<run>/checkpoints/step_002000000.pt

Kết quả trong run_dir (mặc định runs/<level>/<thời điểm>):
  config.yaml, progress.csv (một dòng mỗi lần cập nhật), evals.csv, checkpoints/step_*.pt, videos/step_*.mp4
"""
import argparse
import csv
import functools
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

# Không import torch ở mức module: các tiến trình môi trường (forkserver) nạp lại module chính,
# import torch ở đó tốn ~300 MB RAM cho mỗi trình giả lập.


def _make_vec_env(env_cfg, level, num_envs, seed):
    from mario_ppo.env import make_env
    fns = [functools.partial(make_env, level, seed=seed + i, **env_cfg) for i in range(num_envs)]
    mode = gym.vector.AutoresetMode.SAME_STEP  # lượt mới bắt đầu ngay trong bước kết thúc; quan sát cuối ở info
    if num_envs == 1:
        return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
    return gym.vector.AsyncVectorEnv(fns, context="forkserver", autoreset_mode=mode)


class _CSV:
    def __init__(self, path):
        self.path, self.f, self.w = Path(path), None, None

    def write(self, row):
        if self.w is None:
            new = not self.path.exists()
            self.f = open(self.path, "a", newline="")
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            if new:
                self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self):
        if self.f:
            self.f.close()


def _device(name):
    import torch
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(cfg):
    """Huấn luyện theo cấu hình OmegaConf `cfg`; trả đường dẫn run_dir."""
    import torch

    from mario_ppo.evaluate import run_episodes
    from mario_ppo.model import ActorCritic
    from mario_ppo.ppo import compute_gae, linear_schedule, ppo_update
    from mario_ppo.wrappers import ACTION_SETS

    a = cfg.algo
    run = Path(cfg.run_dir or f"runs/{cfg.level}/{datetime.now():%Y-%m-%d_%H-%M-%S}")
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "videos").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run / "config.yaml", resolve=True)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    env_cfg = dict(cfg_dict["env"], gamma=a.gamma)

    if cfg.torch_threads:
        torch.set_num_threads(cfg.torch_threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = _device(cfg.device)

    n_actions = len(ACTION_SETS[cfg.env.actions])
    model = ActorCritic(n_actions, in_channels=cfg.env.frame_stack, **cfg_dict.get("model", {})).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, eps=a.adam_eps)
    global_step = 0
    if cfg.resume:
        ck = torch.load(cfg.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        global_step = ck["global_step"]
        print(f"resumed from {cfg.resume} at step {global_step}")

    progress, evals = _CSV(run / "progress.csv"), _CSV(run / "evals.csv")
    envs = _make_vec_env(env_cfg, cfg.level, a.num_envs, cfg.seed)
    T, N = a.num_steps, a.num_envs
    obs_shape = envs.single_observation_space.shape
    b_obs = torch.zeros((T, N, *obs_shape), dtype=torch.uint8)
    b_act = torch.zeros((T, N), dtype=torch.long)
    b_logp, b_val, b_rew, b_term, b_trunc, b_final = (torch.zeros((T, N)) for _ in range(6))
    recent = deque(maxlen=100)  # (lợi tức thô, độ dài, x_pos, chạm cờ) của 100 lượt gần nhất
    next_ckpt = (global_step // cfg.checkpoint_every + 1) * cfg.checkpoint_every

    def checkpoint():
        tag = f"step_{global_step:09d}"
        path = run / "checkpoints" / f"{tag}.pt"
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "global_step": global_step, "config": cfg_dict}, path)
        if cfg.eval_episodes:
            res = run_episodes(model, cfg.level, cfg_dict["env"], episodes=cfg.eval_episodes,
                               seed=cfg.seed * 1000 + global_step // cfg.checkpoint_every,
                               max_steps=cfg.eval_max_steps,
                               video=(run / "videos" / f"{tag}.mp4") if cfg.video else None)
            evals.write({"global_step": global_step, "flag_rate": res["flag_rate"],
                         "x_pos": " ".join(map(str, res["x_pos"]))})
            print(f"[eval] step {global_step}: flag_rate {res['flag_rate']:.2f} x_pos {res['x_pos']}", flush=True)
        return path

    try:
        obs, _ = envs.reset(seed=cfg.seed)
        obs = torch.as_tensor(obs)
        t_start, step_start = time.perf_counter(), global_step
        while global_step < a.total_steps:
            lr = linear_schedule(a.lr, global_step, a.total_steps) if a.anneal_lr else a.lr
            for g in opt.param_groups:
                g["lr"] = lr

            # ---- thu một đoạn T bước trên N môi trường ----
            model.eval()
            b_final.zero_()
            for t in range(T):
                action, logp, value = model.act(obs)
                next_obs, reward, term, trunc, info = envs.step(action.numpy())
                b_obs[t], b_act[t], b_logp[t], b_val[t] = obs, action, logp, value
                b_rew[t] = torch.as_tensor(reward, dtype=torch.float32)
                b_term[t] = torch.as_tensor(term, dtype=torch.float32)
                b_trunc[t] = torch.as_tensor(np.logical_and(trunc, ~term), dtype=torch.float32)
                done = np.logical_or(term, trunc)
                if done.any():
                    fin = info["final_info"]
                    for i in np.flatnonzero(done):
                        recent.append((float(fin["episode"]["r"][i]), int(fin["episode"]["l"][i]),
                                       int(fin["x_pos"][i]), bool(fin["flag_get"][i])))
                    cut = np.flatnonzero(np.logical_and(trunc, ~term))
                    if len(cut):  # giá trị của quan sát cuối của lượt bị cắt (không phải quan sát đầu lượt mới)
                        final_obs = torch.as_tensor(np.stack([info["final_obs"][i] for i in cut]))
                        _, _, v_fin = model.act(final_obs)
                        b_final[t, cut] = v_fin
                obs = torch.as_tensor(next_obs)
            global_step += T * N
            _, _, last_value = model.act(obs)

            # ---- lợi thế + cập nhật ----
            adv, ret = compute_gae(b_rew, b_val, b_term, b_trunc, b_final, last_value, a.gamma, a.gae_lambda)
            model.train()
            batch = {"obs": b_obs.reshape(T * N, *obs_shape), "actions": b_act.reshape(-1),
                     "old_logp": b_logp.reshape(-1), "old_values": b_val.reshape(-1),
                     "advantages": adv.reshape(-1), "returns": ret.reshape(-1)}
            batch = {k: v.to(device) for k, v in batch.items()}
            stats = ppo_update(model, opt, batch, a)

            # ---- ghi nhật ký ----
            sps = (global_step - step_start) / (time.perf_counter() - t_start)
            ep = np.array(recent, dtype=float) if recent else np.full((1, 4), np.nan)
            row = {"global_step": global_step, "sps": round(sps, 1), "lr": lr,
                   **{k: round(float(v), 5) for k, v in stats.items()},
                   "episodes": len(recent), "ep_return": round(float(ep[:, 0].mean()), 3),
                   "ep_length": round(float(ep[:, 1].mean()), 1), "x_pos": round(float(ep[:, 2].mean()), 1),
                   "flag_rate": round(float(ep[:, 3].mean()), 3)}
            progress.write(row)
            print(f"step {global_step:>9d}  sps {sps:6.1f}  return {row['ep_return']:8.2f}  "
                  f"x {row['x_pos']:7.1f}  flag {row['flag_rate']:.2f}  kl {row.get('approx_kl', 0):.4f}", flush=True)
            if global_step >= next_ckpt:
                checkpoint()
                next_ckpt += cfg.checkpoint_every
        if not list((run / "checkpoints").glob(f"step_{global_step:09d}.pt")):
            checkpoint()
    finally:
        envs.close()
        progress.close()
        evals.close()
    return run


def load_config(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("overrides", nargs="*", help="khóa=giá trị, ví dụ algo.lr=1e-4 env.noop_max=0")
    args = p.parse_args(argv)
    return OmegaConf.merge(OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides))


if __name__ == "__main__":
    print(f"run dir: {train(load_config())}")
