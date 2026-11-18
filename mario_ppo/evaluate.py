"""Đánh giá một checkpoint trên một màn đã biết (CÓ SẴN; dùng cho chấm điểm, không sửa).

    python -m mario_ppo.evaluate --ckpt runs/1-1/<run>/checkpoints/step_005000000.pt
    python -m mario_ppo.evaluate --ckpt <ckpt> --level 1-4 --episodes 20 --out ket-qua-1-4.json --video 1-4.mp4

Giao thức: `episodes` lượt, lượt i bắt đầu sau n_i bước NOOP (n_i ngẫu nhiên 0..30 sinh từ --seed, giống nhau
cho mọi nhóm), chính sách lấy mẫu ngẫu nhiên như khi huấn luyện, tối đa --max-steps bước tác tử mỗi lượt.
Một lượt thành công khi Mario chạm cờ (info["flag_get"]). Màn được coi là VƯỢT khi flag_rate >= --threshold.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mario_ppo.env import make_env
from mario_ppo.model import ActorCritic
from mario_ppo.wrappers import ACTION_SETS

NOOP = 0


def run_episodes(model, level, env_cfg, episodes=20, seed=2026, max_steps=3000, max_noop=30, video=None):
    """Chạy `episodes` lượt của `model` trên `level`; nếu có `video`, ghi lượt chạm cờ đầu tiên
    (hoặc lượt đi xa nhất) ra tệp mp4."""
    rng = np.random.default_rng(seed)
    noops = [int(n) for n in rng.integers(0, max_noop + 1, size=episodes)]
    env = make_env(level, **dict(env_cfg, eval_mode=True, record_frames=video is not None))
    was_training = model.training
    model.eval()
    torch.manual_seed(seed)
    results, best_frames, best_key = [], None, None
    try:
        for i, n in enumerate(noops):
            obs, info = env.reset(seed=seed + i)
            done, steps = False, 0
            for _ in range(n):
                obs, _, term, trunc, info = env.step(NOOP)
                if term or trunc:
                    done = True
                    break
            while not done and steps < max_steps:
                action, _, _ = model.act(torch.as_tensor(np.asarray(obs)[None]))
                obs, _, term, trunc, info = env.step(int(action))
                done, steps = term or trunc, steps + 1
            r = {"x_pos": int(info.get("x_pos", 0)), "flag_get": bool(info.get("flag_get", False)), "steps": steps}
            results.append(r)
            if video is not None:
                key = (r["flag_get"], r["x_pos"])
                if best_key is None or (key > best_key and not best_key[0]):
                    best_key, best_frames = key, list(env.get_wrapper_attr("frames"))
    finally:
        env.close()
        model.train(was_training)
    if video is not None and best_frames:
        import imageio.v3 as iio
        iio.imwrite(video, np.stack(best_frames), fps=60, codec="libx264")
    return {"episodes": episodes, "flag_rate": float(np.mean([r["flag_get"] for r in results])),
            "x_pos": [r["x_pos"] for r in results], "steps": [r["steps"] for r in results], "noop_steps": noops}


def load_model(ckpt, device="cpu"):
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model = ActorCritic(len(ACTION_SETS[cfg["env"]["actions"]]), in_channels=cfg["env"]["frame_stack"],
                        **cfg.get("model", {}))
    model.load_state_dict(ck["model"])
    return model.to(device), ck


def evaluate_checkpoint(ckpt, level=None, episodes=20, seed=2026, max_steps=3000, out=None, video=None,
                        threshold=0.8, device="cpu"):
    model, ck = load_model(ckpt, device)
    cfg = ck["config"]
    level = level or cfg["level"]
    res = run_episodes(model, level, cfg["env"], episodes=episodes, seed=seed, max_steps=max_steps, video=video)
    res = {"level": level, "checkpoint": str(ckpt), "global_step": int(ck.get("global_step", 0)), "seed": seed,
           **res, "mean_x_pos": float(np.mean(res["x_pos"])), "threshold": threshold,
           "solved": res["flag_rate"] >= threshold}
    if out is not None:
        Path(out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--level", default=None, help="mặc định: màn trong cấu hình của checkpoint")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--max-steps", type=int, default=3000)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--out", default=None, help="tệp JSON kết quả")
    p.add_argument("--video", default=None, help="tệp mp4 của lượt tốt nhất")
    args = p.parse_args(argv)
    res = evaluate_checkpoint(args.ckpt, args.level, args.episodes, args.seed, args.max_steps, args.out,
                              args.video, args.threshold)
    print(f"{res['level']}: flag_rate {res['flag_rate']:.2f} ({sum(1 for _ in res['x_pos'])} lượt), "
          f"x_pos trung bình {res['mean_x_pos']:.0f} -> {'VƯỢT' if res['solved'] else 'chưa vượt'}")


if __name__ == "__main__":
    main()
