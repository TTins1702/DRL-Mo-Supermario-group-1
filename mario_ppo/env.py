"""Tạo môi trường mo-supermario-v0 đã tiền xử lý (CÓ SẴN; nhóm được thêm mẹo huấn luyện ở chỗ đánh dấu).

Chuỗi wrapper (từ trong ra ngoài):
  môi trường gốc -> DeathPenaltyFix -> XPosWrapFix -> [RecordFrames] -> ButtonActions -> FrameSkip
  -> [mẹo huấn luyện của nhóm] -> [RandomNoopStart] -> ScalarizeReward -> GrayResize -> FrameStack
  -> RecordEpisodeStatistics -> [NormalizeReward]

eval_mode=True (script đánh giá dùng): bỏ mẹo huấn luyện, bỏ NOOP ngẫu nhiên, bỏ chuẩn hóa phần thưởng.
Phần quan sát (FrameSkip, GrayResize, FrameStack) phải GIỐNG HỆT khi huấn luyện, nếu không chính sách
sẽ nhận đầu vào khác với lúc học.
"""
import gymnasium as gym
import mo_gymnasium as mo

from mario_ppo.wrappers import (ACTION_SETS, ButtonActions, DeathPenaltyFix, FrameSkip, GrayResize,
                                RandomNoopStart, RecordFrames, ScalarizeReward, XPosWrapFix)

# [x_pos, time, death, coin, enemy]; môi trường đã co giãn mỗi thành phần theo hệ số 5/150.
DEFAULT_WEIGHTS = (1.0, 1.0, 2.0, 0.05, 0.05)


def parse_level(level):
    """'1-1' -> (1, 1)."""
    world, stage = (int(v) for v in str(level).split("-"))
    if not (1 <= world <= 8 and 1 <= stage <= 4):
        raise ValueError(f"invalid level {level!r}")
    return world, stage


def make_env(level="1-1", seed=None, actions="simple", frame_skip=4, frame_stack=4, size=84,
             weights=DEFAULT_WEIGHTS, noop_max=0, normalize_reward=True, gamma=0.9, eval_mode=False,
             record_frames=False, **tricks):
    """`tricks`: các khóa bổ sung trong mục env của cấu hình, dành cho mẹo của nhóm (mặc định bỏ qua)."""
    env = mo.make("mo-supermario-v0", target=parse_level(level))
    env = DeathPenaltyFix(env)
    env = XPosWrapFix(env)
    if record_frames:
        env = RecordFrames(env)
    env = ButtonActions(env, ACTION_SETS[actions])
    env = FrameSkip(env, skip=frame_skip)
    if not eval_mode:
        # ---- mẹo huấn luyện của nhóm (tùy chọn), ví dụ: ----
        # if tricks.get("max_no_progress_steps"):
        #     env = NoProgressTruncation(env, tricks["max_no_progress_steps"])
        if noop_max:
            env = RandomNoopStart(env, noop_max)
    env = ScalarizeReward(env, weights)
    env = GrayResize(env, size=size)
    env = gym.wrappers.FrameStackObservation(env, frame_stack)
    env = gym.wrappers.RecordEpisodeStatistics(env)  # lợi tức thô (chưa chuẩn hóa) trong info["episode"]
    if normalize_reward and not eval_mode:
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return env
