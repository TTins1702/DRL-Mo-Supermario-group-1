"""Mạng actor-critic cho ảnh Mario (TODO).

Giao diện mà train.py và evaluate.py dùng (giữ nguyên tên và chữ ký):
  ActorCritic(n_actions, in_channels=4)
  forward(obs)  : obs uint8 (B, C, 84, 84) -> (logits (B, n_actions) float32, value (B,) float32)
  act(obs)      : lấy mẫu hành động, trả (action long (B,), logp (B,), value (B,)) trên CPU, không gradient
Nhóm được tự chọn kiến trúc (số lớp, kênh, IMPALA-CNN, ...) miễn giữ giao diện trên.
"""
import numpy as np
import torch
from torch import nn


def _init(layer, gain=np.sqrt(2)):
    """Khởi tạo trực giao, bias bằng 0 (thường dùng cho PPO)."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.zeros_(layer.bias)
    return layer


class ActorCritic(nn.Module):
    """Thân CNN dùng chung, hai đầu ra: logits của chính sách và giá trị trạng thái.

    Mục đích: một mạng vừa là actor (chọn hành động) vừa là critic (ước lượng giá trị cho GAE).
    TODO:
      - __init__: thân CNN nhận (in_channels, 84, 84), đầu chính sách n_actions logits, đầu giá trị 1 số.
        Khởi tạo đầu chính sách với trọng số rất nhỏ (ví dụ gain 0,01) để chính sách ban đầu gần đều.
      - forward: chuẩn hóa điểm ảnh uint8 về [0, 1], chạy thân, trả (logits, value) với value shape (B,).
    """

    def __init__(self, n_actions, in_channels=4, hidden_dim=512):
        super().__init__()
        self.trunk = nn.Sequential(
            _init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            _init(nn.Linear(64 * 7 * 7, hidden_dim)),
            nn.ReLU(),
        )
        self.policy = _init(nn.Linear(hidden_dim, n_actions), gain=0.01)
        self.value = _init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, obs):
        h = self.trunk(obs.to(dtype=torch.float32) / 255.0)
        return self.policy(h), self.value(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        """CÓ SẴN. Lấy mẫu từ chính sách hiện tại; nhận obs ở thiết bị bất kỳ, trả tensor CPU."""
        device = next(self.parameters()).device
        logits, value = self(obs.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
