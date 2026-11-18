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

    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        raise NotImplementedError("TODO: ActorCritic.__init__ (thân CNN, self.policy, self.value)")

    def forward(self, obs):
        raise NotImplementedError("TODO: ActorCritic.forward")

    @torch.no_grad()
    def act(self, obs):
        """CÓ SẴN. Lấy mẫu từ chính sách hiện tại; nhận obs ở thiết bị bất kỳ, trả tensor CPU."""
        device = next(self.parameters()).device
        logits, value = self(obs.to(device))
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.cpu(), dist.log_prob(action).cpu(), value.cpu()
