"""Kiểm tra các wrapper tiền xử lý: FrameSkip, GrayResize, ScalarizeReward."""
import gymnasium as gym
import numpy as np
import pytest

from mario_ppo.wrappers import FrameSkip, GrayResize, ScalarizeReward


class _FakeMarioEnv(gym.Env):
    """Môi trường giả: khung hình thứ k có mọi điểm ảnh bằng k, phần thưởng véc-tơ 5 chiều bằng k * [1..5]."""

    observation_space = gym.spaces.Box(0, 255, (240, 256, 3), np.uint8)
    action_space = gym.spaces.Discrete(7)

    def __init__(self, end_at=None, pattern=None):
        self.end_at, self.pattern, self.k = end_at, pattern, 0
        self.actions = []

    def _frame(self):
        if self.pattern is not None:
            return self.pattern[self.k % len(self.pattern)].copy()
        return np.full((240, 256, 3), self.k, np.uint8)

    def reset(self, seed=None, options=None):
        self.k, self.actions = 0, []
        return self._frame(), {}

    def step(self, action):
        self.k += 1
        self.actions.append(int(action))
        reward = self.k * np.arange(1, 6, dtype=np.float32)
        terminated = self.end_at is not None and self.k >= self.end_at
        return self._frame(), reward, terminated, False, {"x_pos": self.k}


def test_frameskip_repeats_action_and_sums_vector_reward():
    env = FrameSkip(_FakeMarioEnv(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(3)
    assert env.unwrapped.actions == [3, 3, 3, 3]
    assert np.allclose(reward, (1 + 2 + 3 + 4) * np.arange(1, 6))
    assert not terminated and not truncated
    assert info["x_pos"] == 4


def test_frameskip_max_pools_last_two_frames():
    a = np.zeros((240, 256, 3), np.uint8)
    b = np.zeros((240, 256, 3), np.uint8)
    a[0, 0] = 200  # chỉ có ở khung chẵn
    b[1, 1] = 100  # chỉ có ở khung lẻ
    env = FrameSkip(_FakeMarioEnv(pattern=[a, b]), skip=4)
    env.reset()
    obs, *_ = env.step(0)
    assert obs.shape == (240, 256, 3) and obs.dtype == np.uint8
    assert obs[0, 0, 0] == 200 and obs[1, 1, 0] == 100


def test_frameskip_stops_at_episode_end():
    env = FrameSkip(_FakeMarioEnv(end_at=2), skip=4)
    env.reset()
    _, reward, terminated, _, _ = env.step(1)
    assert terminated
    assert env.unwrapped.actions == [1, 1]
    assert np.allclose(reward, 3 * np.arange(1, 6))


def test_grayresize_shape_dtype_and_brightness():
    env = GrayResize(FrameSkip(_FakeMarioEnv(), skip=1), size=84)
    assert env.observation_space.shape == (84, 84)
    obs, _ = env.reset()
    assert obs.shape == (84, 84) and obs.dtype == np.uint8
    for _ in range(50):
        obs, *_ = env.step(0)
    # ảnh xám đều 50 -> độ sáng xấp xỉ 50
    assert abs(float(obs.mean()) - 50.0) <= 1.0


def test_grayresize_uses_luminance_not_channel_mean():
    class _Red(_FakeMarioEnv):
        def _frame(self):
            f = np.zeros((240, 256, 3), np.uint8)
            f[..., 0] = 255  # đỏ thuần
            return f
    obs, _ = GrayResize(_Red(), size=84).reset()
    # Y = 0,299 R + 0,587 G + 0,114 B -> khoảng 76
    assert 70 <= float(obs.mean()) <= 82


def test_scalarize_reward_dot_product_and_info():
    w = [1.0, 0.5, 2.0, 0.0, -1.0]
    env = ScalarizeReward(FrameSkip(_FakeMarioEnv(), skip=1), w)
    env.reset()
    _, reward, _, _, info = env.step(0)
    vec = np.arange(1, 6, dtype=np.float32)
    assert isinstance(reward, float)
    assert reward == pytest.approx(float(np.dot(w, vec)))
    assert np.allclose(info["vector_reward"], vec)


def test_scalarize_reward_rejects_wrong_weight_count():
    with pytest.raises(ValueError):
        ScalarizeReward(_FakeMarioEnv(), [1.0, 2.0])


from mario_ppo.wrappers import MarioRewardShaping, NoProgressTruncation


class _StuckMarioEnv(_FakeMarioEnv):
    def __init__(self, x_seq, end_at=None, flag_at=None):
        super().__init__(end_at=end_at)
        self.x_seq = x_seq
        self.flag_at = flag_at

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info["x_pos"] = self.x_seq[min(self.k - 1, len(self.x_seq) - 1)]
        info["flag_get"] = self.flag_at is not None and self.k == self.flag_at
        if info["x_pos"] == 0:
            reward[0] = 0.0
        return obs, reward, terminated, truncated, info


def test_no_progress_truncation_cuts_stalled_episode_but_not_terminal():
    env = NoProgressTruncation(_StuckMarioEnv([10, 10, 10, 10]), max_steps=2)
    env.reset()
    _, _, term1, trunc1, _ = env.step(0)  # x=10 (max_x mới) -> stalled=0
    _, _, term2, trunc2, _ = env.step(0)  # x=10 -> stalled=1
    _, _, term3, trunc3, _ = env.step(0)  # x=10 -> stalled=2 -> bị cắt (truncated)
    assert not trunc1 and not trunc2 and trunc3 and not term3

    # Nếu lượt kết thúc thật đúng lúc stalled chạm ngưỡng max_steps, truncated phải giữ là False
    env_term = NoProgressTruncation(_StuckMarioEnv([10, 10, 10], end_at=3), max_steps=2)
    env_term.reset()
    env_term.step(0)
    env_term.step(0)
    _, _, term, trunc, _ = env_term.step(0)
    assert term and not trunc


def test_mario_reward_shaping_modifies_x_component_before_scalarize():
    env = MarioRewardShaping(_StuckMarioEnv([0, 20], flag_at=2), flag_bonus=5.0, stuck_penalty=-0.25)
    env.reset()
    _, r1, _, _, _ = env.step(0)  # x=0 -> reward[0] ban đầu 0.0 -> -0.25
    _, r2, _, _, _ = env.step(0)  # flag_get=True -> reward[0] ban đầu 2.0 -> 2.0 + 5.0 = 7.0
    assert r1[0] == pytest.approx(-0.25)
    assert r2[0] == pytest.approx(7.0)


@pytest.mark.parametrize("end_at", [1, 2, 3, 4])
@pytest.mark.parametrize("end_kind", ["terminated", "truncated"])
def test_frameskip_stops_on_either_end_and_keeps_last_info(end_at, end_kind):
    class _EndingEnv(_FakeMarioEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            ended = self.k == end_at
            return obs, reward, ended and end_kind == "terminated", ended and end_kind == "truncated", info

    env = FrameSkip(_EndingEnv(), skip=4)
    env.reset()
    obs, reward, terminated, truncated, info = env.step(2)
    assert env.unwrapped.actions == [2] * end_at
    assert reward.dtype == np.float32 and reward.shape == (5,)
    assert np.array_equal(reward, sum(range(1, end_at + 1)) * np.arange(1, 6))
    assert terminated == (end_kind == "terminated")
    assert truncated == (end_kind == "truncated")
    assert info["x_pos"] == end_at
    assert np.all(obs == end_at)


def test_frameskip_uses_only_last_two_frames_and_no_reset_frame():
    class _SequenceEnv(_FakeMarioEnv):
        def _frame(self):
            return np.full((240, 256, 3), [200, 250, 30, 40, 5][self.k], np.uint8)

    env = FrameSkip(_SequenceEnv(), skip=4)
    obs, _ = env.reset()
    assert np.all(obs == 200)
    obs, reward, terminated, truncated, info = env.step(0)
    assert np.all(obs == 40)  # max(frame 3=30, frame 4=40), không lấy reset/khung đầu
    assert reward.dtype == np.float32 and info["x_pos"] == 4
    assert not terminated and not truncated


@pytest.mark.parametrize("rgb,expected", [
    ((0, 0, 0), 0), ((255, 255, 255), 255), ((255, 0, 0), 76),
    ((0, 255, 0), 150), ((0, 0, 255), 29), ((50, 50, 50), 50),
])
@pytest.mark.parametrize("size", [42, 84])
def test_grayresize_rgb_luminance_reset_and_step_at_configured_size(rgb, expected, size):
    class _SolidRGBEnv(_FakeMarioEnv):
        def _frame(self):
            return np.broadcast_to(np.asarray(rgb, dtype=np.uint8), (240, 256, 3)).copy()

    env = GrayResize(_SolidRGBEnv(), size=size)
    assert env.observation_space.shape == (size, size)
    assert env.observation_space.dtype == np.uint8
    assert np.all(env.observation_space.low == 0)
    assert np.all(env.observation_space.high == 255)
    reset_obs, _ = env.reset()
    step_obs, *_ = env.step(0)
    for obs in (reset_obs, step_obs):
        assert obs.shape == (size, size) and obs.dtype == np.uint8
        assert np.all(np.abs(obs.astype(np.int16) - expected) <= 1)


def test_scalarize_reward_uses_declared_objective_count_and_preserves_info():
    class _ThreeObjectiveEnv(_FakeMarioEnv):
        reward_space = gym.spaces.Box(-np.inf, np.inf, (3,), np.float32)

        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            return obs, np.array([2.0, -4.0, 8.0], dtype=np.float64), terminated, truncated, dict(info, marker="kept")

    with pytest.raises(ValueError):
        ScalarizeReward(_ThreeObjectiveEnv(), [1.0] * 5)
    env = ScalarizeReward(_ThreeObjectiveEnv(), [0.5, 0.25, -1.0])
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(reward, float) and reward == pytest.approx(-8.0)
    assert info["marker"] == "kept" and info["x_pos"] == 1
    assert info["vector_reward"].shape == (3,) and info["vector_reward"].dtype == np.float32
    assert np.array_equal(info["vector_reward"], [2.0, -4.0, 8.0])
    assert obs.shape == (240, 256, 3) and not terminated and not truncated


def test_scalarize_reward_defaults_to_five_objectives_when_space_is_none():
    class _NoRewardSpace(_FakeMarioEnv):
        reward_space = None

    with pytest.raises(ValueError):
        ScalarizeReward(_NoRewardSpace(), [1.0] * 3)
    env = ScalarizeReward(_NoRewardSpace(), [0.0, 0.5, 0.0, 0.0, -1.0])
    env.reset()
    _, reward, _, _, info = env.step(0)
    assert reward == pytest.approx(-4.0)
    assert info["vector_reward"].dtype == np.float32


def test_scalarize_reward_rejects_non_vector_weights():
    with pytest.raises(ValueError):
        ScalarizeReward(_FakeMarioEnv(), [[1.0, 1.0, 1.0, 1.0, 1.0]])


def test_no_progress_truncation_uses_highest_x_and_resets_each_episode():
    env = NoProgressTruncation(_StuckMarioEnv([10, 10, 11, 10, 11, 11]), max_steps=2)
    env.reset()
    flags = [env.step(0)[3] for _ in range(5)]
    assert flags == [False, False, False, False, True]
    env.reset()
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and not truncated and info["x_pos"] == 10


def test_no_progress_truncation_preserves_environment_truncation():
    class _TimeLimitEnv(_StuckMarioEnv):
        def step(self, action):
            obs, reward, terminated, _, info = super().step(action)
            return obs, reward, terminated, self.k == 2, info

    env = NoProgressTruncation(_TimeLimitEnv([10, 10]), max_steps=10)
    env.reset()
    env.step(0)
    obs, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and info["x_pos"] == 10
    assert obs.shape == (240, 256, 3) and reward.shape == (5,)


def test_mario_reward_shaping_copies_source_vector_and_keeps_other_objectives():
    class _ReusedRewardEnv(_FakeMarioEnv):
        def __init__(self):
            super().__init__()
            self.shared_reward = np.array([0.0, -2.0, -3.0, 4.0, 5.0], dtype=np.float32)

        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            return obs, self.shared_reward, terminated, truncated, info

    base = _ReusedRewardEnv()
    env = MarioRewardShaping(base, stuck_penalty=-0.25)
    env.reset()
    _, shaped, terminated, truncated, info = env.step(0)
    assert shaped.dtype == np.float32
    assert np.array_equal(shaped, [-0.25, -2.0, -3.0, 4.0, 5.0])
    assert np.array_equal(base.shared_reward, [0.0, -2.0, -3.0, 4.0, 5.0])
    assert not np.shares_memory(shaped, base.shared_reward)
    assert info["x_pos"] == 1 and not terminated and not truncated

    base.shared_reward[0] = 3.0
    assert shaped[0] == pytest.approx(-0.25)


@pytest.mark.parametrize("terminated,truncated,expected", [
    (False, False, -0.25), (False, True, -0.25), (True, False, 0.0),
])
def test_mario_reward_shaping_penalty_depends_on_termination_not_truncation(terminated, truncated, expected):
    class _EndingEnv(_FakeMarioEnv):
        def step(self, action):
            obs, reward, _, _, info = super().step(action)
            reward[0] = 0.0
            return obs, reward, terminated, truncated, info

    env = MarioRewardShaping(_EndingEnv(), stuck_penalty=-0.25)
    env.reset()
    _, reward, got_terminated, got_truncated, _ = env.step(0)
    assert reward[0] == pytest.approx(expected)
    assert (got_terminated, got_truncated) == (terminated, truncated)


def test_mario_reward_shaping_applies_flag_bonus_before_stuck_penalty():
    class _FlagEnv(_FakeMarioEnv):
        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            reward[0] = -6.0
            return obs, reward, terminated, truncated, dict(info, flag_get=True)

    env = MarioRewardShaping(_FlagEnv(), flag_bonus=5.0, stuck_penalty=-0.25)
    env.reset()
    _, reward, _, _, _ = env.step(0)
    assert reward[0] == pytest.approx(-1.25)

    neutral = MarioRewardShaping(_FlagEnv())
    neutral.reset()
    _, unchanged, _, _, _ = neutral.step(0)
    assert unchanged[0] == pytest.approx(-6.0)


def test_make_env_uses_tricks_only_for_training_and_keeps_shared_preprocessing(monkeypatch):
    from mario_ppo import env as mario_env

    class _BaseEnv(_FakeMarioEnv):
        reward_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)

        def step(self, action):
            obs, reward, terminated, truncated, info = super().step(action)
            reward[0] = 0.1  # môi trường thật đã co giãn x_pos trước XPosWrapFix
            return obs, reward, terminated, truncated, dict(info, flag_get=True)

    def _make_base(env_id, target):
        assert env_id == "mo-supermario-v0" and target == (1, 1)
        return _BaseEnv()

    monkeypatch.setattr(mario_env.mo, "make", _make_base)
    options = dict(actions="simple", frame_skip=1, frame_stack=2, size=42,
                   weights=(1.0, 0.0, 0.0, 0.0, 0.0), normalize_reward=False,
                   max_no_progress_steps=2, flag_bonus=5.0, stuck_penalty=-0.25)
    train_env = mario_env.make_env("1-1", **options)
    eval_env = mario_env.make_env("1-1", eval_mode=True, **options)

    def _chain(env):
        names = []
        while isinstance(env, gym.Wrapper):
            names.append(type(env).__name__)
            env = env.env
        return names

    try:
        train_chain = _chain(train_env)
        eval_chain = _chain(eval_env)
        assert train_chain.index("MarioRewardShaping") < train_chain.index("NoProgressTruncation")
        assert train_chain.index("NoProgressTruncation") < train_chain.index("FrameSkip")
        assert train_chain.index("ScalarizeReward") < train_chain.index("MarioRewardShaping")
        assert "MarioRewardShaping" not in eval_chain and "NoProgressTruncation" not in eval_chain
        assert [name for name in train_chain if name in ("FrameSkip", "GrayResize", "FrameStackObservation", "ButtonActions")] == [
            name for name in eval_chain if name in ("FrameSkip", "GrayResize", "FrameStackObservation", "ButtonActions")
        ]
        train_obs, _ = train_env.reset()
        eval_obs, _ = eval_env.reset()
        assert train_obs.shape == eval_obs.shape == (2, 42, 42)
        assert train_env.action_space.n == eval_env.action_space.n == 7
        _, train_reward, _, _, train_info = train_env.step(0)
        _, eval_reward, _, _, eval_info = eval_env.step(0)
        assert train_reward == pytest.approx(5.1)
        assert eval_reward == pytest.approx(0.1)
        assert train_info["vector_reward"][0] == pytest.approx(5.1)
        assert eval_info["vector_reward"][0] == pytest.approx(0.1)
    finally:
        train_env.close()
        eval_env.close()


def test_make_env_default_tricks_absent_and_eval_ignores_noops_normalization(monkeypatch):
    from mario_ppo import env as mario_env

    class _BaseEnv(_FakeMarioEnv):
        reward_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)

    monkeypatch.setattr(mario_env.mo, "make", lambda *args, **kwargs: _BaseEnv())
    default_train = mario_env.make_env("1-1", normalize_reward=False, noop_max=0)
    eval_env = mario_env.make_env("1-1", eval_mode=True, noop_max=30, normalize_reward=True,
                                  max_no_progress_steps=2, flag_bonus=5.0, stuck_penalty=-0.25)
    try:
        for env in (default_train, eval_env):
            chain = []
            while isinstance(env, gym.Wrapper):
                chain.append(type(env).__name__)
                env = env.env
            assert "NoProgressTruncation" not in chain
            assert "MarioRewardShaping" not in chain
            assert "RandomNoopStart" not in chain
            assert "NormalizeReward" not in chain
    finally:
        default_train.close()
        eval_env.close()


def test_make_env_full_wrapper_pipeline_preserves_vector_and_eval_reward(monkeypatch):
    from mario_ppo import env as mario_env

    class _TwoFrameFlagEnv(_FakeMarioEnv):
        reward_space = gym.spaces.Box(-np.inf, np.inf, (5,), np.float32)

        def __init__(self):
            super().__init__(end_at=2)

        def _frame(self):
            return np.full((240, 256, 3), 10 * self.k, np.uint8)

        def step(self, action):
            obs, _, terminated, truncated, info = super().step(action)
            reward = np.array([0.1, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return obs, reward, terminated, truncated, dict(info, flag_get=terminated)

    created = []

    def _make_base(*args, **kwargs):
        base = _TwoFrameFlagEnv()
        created.append(base)
        return base

    monkeypatch.setattr(mario_env.mo, "make", _make_base)
    options = dict(frame_skip=4, frame_stack=2, size=42,
                   weights=(1.0, 0.5, 0.0, 0.0, 0.0), normalize_reward=False,
                   max_no_progress_steps=2, flag_bonus=5.0, stuck_penalty=-0.25)
    train_env = mario_env.make_env("1-1", **options)
    eval_env = mario_env.make_env("1-1", eval_mode=True, **options)
    try:
        for env, expected_vec, expected_scalar in (
            (train_env, [5.2, 2.0, 0.0, 0.0, 0.0], 6.2),
            (eval_env, [0.2, 2.0, 0.0, 0.0, 0.0], 1.2),
        ):
            initial, _ = env.reset()
            assert initial.shape == (2, 42, 42) and initial.dtype == np.uint8
            assert np.all(initial == 0)
            obs, scalar, terminated, truncated, info = env.step(1)
            assert terminated and not truncated and info["flag_get"] and info["x_pos"] == 2
            assert created[0 if env is train_env else 1].actions == [128, 128]
            assert obs.shape == (2, 42, 42) and obs.dtype == np.uint8
            assert np.all(obs[0] == 0) and np.all(obs[1] == 20)
            assert info["vector_reward"].dtype == np.float32
            assert np.allclose(info["vector_reward"], expected_vec)
            assert scalar == pytest.approx(expected_scalar)
    finally:
        train_env.close()
        eval_env.close()
