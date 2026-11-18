"""Wrapper Gymnasium cho mo-supermario-v0.

Phần CÓ SẴN (không cần sửa): chuyển hành động, sửa lỗi của môi trường gốc, xuất phát ngẫu nhiên, ghi khung hình.
Phần TODO: FrameSkip, GrayResize, ScalarizeReward. Có thể thêm wrapper riêng của nhóm ở cuối tệp
(ví dụ cắt lượt khi Mario đứng yên, thưởng khi chạm cờ, ...) và bật chúng trong env.py.
"""
import cv2
import gymnasium as gym
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT, RIGHT_ONLY, SIMPLE_MOVEMENT

# Các tập hành động có thể chọn qua cấu hình env.actions. Hành động 0 luôn là NOOP.
ACTION_SETS = {
    "right_only": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "simple_down": SIMPLE_MOVEMENT + [["down"]],  # thêm DOWN để chui ống
    "complex": COMPLEX_MOVEMENT,
}

_BUTTON_BITS = {"right": 0x80, "left": 0x40, "down": 0x20, "up": 0x10,
                "start": 0x08, "select": 0x04, "B": 0x02, "A": 0x01, "NOOP": 0x00}


# ---------------------------------------------------------------------------------------------------------
# CÓ SẴN
# ---------------------------------------------------------------------------------------------------------

class ButtonActions(gym.ActionWrapper):
    """Discrete(256) (mặt nạ bit của tay cầm NES) -> Discrete(len(movements))."""

    def __init__(self, env, movements=SIMPLE_MOVEMENT):
        super().__init__(env)
        self._masks = [sum(_BUTTON_BITS[b] for b in combo) for combo in movements]
        self.action_space = gym.spaces.Discrete(len(self._masks))

    def action(self, action):
        return self._masks[int(action)]


class DeathPenaltyFix(gym.Wrapper):
    """Môi trường gốc kết thúc lượt ngay khi Mario bắt đầu chết, trước khi số mạng giảm, nên thành phần
    phần thưởng 'death' luôn bằng 0. Wrapper này trả lại phần phạt -25 (đã co giãn như các thành phần khác).
    Bọc trực tiếp môi trường gốc (trước FrameSkip)."""

    DEATH_INDEX = 2  # thứ tự mục tiêu: [x_pos, time, death, coin, enemy]

    def __init__(self, env):
        super().__init__(env)
        self.penalty = -25.0 * env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated and not info.get("flag_get", False):
            reward = np.array(reward, dtype=np.float32, copy=True)
            if reward[self.DEATH_INDEX] == 0.0:
                reward[self.DEATH_INDEX] = self.penalty
        return obs, reward, terminated, truncated, info


class XPosWrapFix(gym.Wrapper):
    """Tọa độ x 16 bit có thể tràn (lâu đài mê cung đưa Mario về bên trái x = 0), khiến môi trường trả một
    phần thưởng x khổng lồ trong một khung. Bỏ các bước nhảy x không thể có. Bọc môi trường gốc."""

    MAX_DX = 16

    def __init__(self, env):
        super().__init__(env)
        self.scale = env.unwrapped.reward_space.shape[0] / 150.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if abs(reward[0]) > self.MAX_DX * self.scale:
            reward = np.array(reward, dtype=np.float32, copy=True)
            reward[0] = 0.0
        x = info.get("x_pos")
        if x is not None and x >= 0x8000:
            info = dict(info, x_pos=int(x) - 0x10000)
        return obs, reward, terminated, truncated, info


class RandomNoopStart(gym.Wrapper):
    """Mỗi lượt bắt đầu sau một số ngẫu nhiên 0..max_noops bước NOOP, để chính sách không học thuộc một
    nhịp thời gian duy nhất. Script đánh giá dùng cùng ý tưởng với các số NOOP cố định theo seed."""

    NOOP = 0

    def __init__(self, env, max_noops):
        super().__init__(env)
        self.max_noops = max_noops

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(int(self.np_random.integers(0, self.max_noops + 1))):
            obs, _, terminated, truncated, info = self.env.step(self.NOOP)
            if terminated or truncated:
                return self.env.reset()
        return obs, info


class RecordFrames(gym.Wrapper):
    """Giữ mọi khung hình RGB của lượt hiện tại trong self.frames (để ghi video)."""

    def __init__(self, env):
        super().__init__(env)
        self.frames = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.frames = [obs.copy()]
        return obs, info

    def step(self, action):
        out = self.env.step(action)
        self.frames.append(out[0].copy())
        return out


# ---------------------------------------------------------------------------------------------------------
# TODO
# ---------------------------------------------------------------------------------------------------------

class FrameSkip(gym.Wrapper):
    """Lặp lại một hành động `skip` khung hình liên tiếp.

    Mục đích: mỗi quyết định của tác tử kéo dài nhiều khung, giảm số bước cần học và tăng tốc thu dữ liệu.

    step(action) phải:
      - gọi self.env.step(action) tối đa `skip` lần, dừng sớm nếu lượt kết thúc hoặc bị cắt;
      - trả tổng phần thưởng véc-tơ (np.float32, cùng shape với phần thưởng gốc);
      - trả quan sát là max theo từng điểm ảnh của HAI khung cuối (NES nhấp nháy sprite giữa các khung);
        nếu chỉ chạy được một khung thì trả khung đó;
      - trả terminated, truncated, info của khung cuối cùng.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        raise NotImplementedError("TODO: FrameSkip.step")


class GrayResize(gym.ObservationWrapper):
    """Ảnh RGB (H, W, 3) uint8 -> ảnh xám (size, size) uint8.

    Mục đích: giảm kích thước đầu vào của mạng (240x256x3 -> 84x84) mà vẫn giữ đủ thông tin để chơi.
    Ảnh xám phải theo độ sáng (luminance, như cv2.COLOR_RGB2GRAY), không phải trung bình ba kênh.
    Cập nhật self.observation_space cho đúng shape mới.
    """

    def __init__(self, env, size=84):
        super().__init__(env)
        self.size = size
        raise NotImplementedError("TODO: GrayResize.__init__ (đặt self.observation_space)")

    def observation(self, obs):
        raise NotImplementedError("TODO: GrayResize.observation")


class ScalarizeReward(gym.Wrapper):
    """Phần thưởng véc-tơ 5 mục tiêu [x_pos, time, death, coin, enemy] -> số thực w · r.

    Mục đích: PPO cần phần thưởng vô hướng; trọng số w quyết định tác tử ưu tiên điều gì.
    - __init__: lưu trọng số; ném ValueError nếu số trọng số khác số mục tiêu
      (env.unwrapped.reward_space.shape[0], hoặc 5 nếu môi trường không có reward_space).
    - step: trả phần thưởng kiểu float của Python; giữ véc-tơ gốc trong info["vector_reward"].
    """

    def __init__(self, env, weights):
        super().__init__(env)
        raise NotImplementedError("TODO: ScalarizeReward.__init__")

    def step(self, action):
        raise NotImplementedError("TODO: ScalarizeReward.step")


# ---------------------------------------------------------------------------------------------------------
# Wrapper bổ sung của nhóm (tùy chọn) - thêm ở đây
# ---------------------------------------------------------------------------------------------------------
