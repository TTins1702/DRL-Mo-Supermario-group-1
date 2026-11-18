"""Kiểm tra mục tiêu cắt, mất mát tổng, approxKL / tỉ lệ cắt và vòng cập nhật PPO.

Mini-batch tính tay: pi_old = (2/3; 1/3), v_old = 2, hành động 1, 1, 2 (ở đây là 0, 0, 1),
A_raw = (1; 1; -2), V_targ = (3; 3; 0), p = pi_theta(1|s) = 0,75, v(s,w) = 2,2, eps 0,2, c1 1, c2 0,01.
"""
import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from mario_ppo.ppo import approx_kl_and_clipfrac, clipped_policy_loss, ppo_loss, ppo_update, value_loss


def _cfg(**kw):
    base = dict(clip_coef=0.2, vf_coef=1.0, ent_coef=0.01, norm_adv=False, clip_vloss=False,
                update_epochs=1, num_minibatches=1, max_grad_norm=0.5, target_kl=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _hand_batch():
    old_logp = torch.log(torch.tensor([2 / 3, 2 / 3, 1 / 3]))
    actions = torch.tensor([0, 0, 1])
    adv = torch.tensor([1.0, 1.0, -2.0])
    v_targ = torch.tensor([3.0, 3.0, 0.0])
    old_values = torch.full((3,), 2.0)
    theta = torch.tensor([math.log(3.0), 0.0], requires_grad=True)  # softmax -> (0.75; 0.25)
    w = torch.tensor(2.2, requires_grad=True)
    logits = theta.expand(3, 2)
    dist = torch.distributions.Categorical(logits=logits)
    new_logp, entropy = dist.log_prob(actions), dist.entropy()
    values = w.expand(3)
    batch = dict(old_logp=old_logp, advantages=adv, returns=v_targ, old_values=old_values)
    return theta, w, new_logp, entropy, values, batch


def test_clipped_policy_loss_hand_batch():
    ratio = torch.tensor([1.125, 1.125, 0.75])
    adv = torch.tensor([1.0, 1.0, -2.0])
    assert clipped_policy_loss(ratio, adv, 0.2).item() == pytest.approx(-0.21667, abs=1e-4)


def test_clipped_policy_loss_never_above_unclipped_objective():
    torch.manual_seed(0)
    ratio = torch.exp(0.5 * torch.randn(1000))
    adv = torch.randn(1000)
    # -L^CLIP >= -L^CPI  <=>  L^CLIP <= L^CPI
    assert clipped_policy_loss(ratio, adv, 0.2).item() >= -(ratio * adv).mean().item() - 1e-6


def test_value_loss_hand_batch():
    v = torch.full((3,), 2.2)
    assert value_loss(v, torch.tensor([3.0, 3.0, 0.0])).item() == pytest.approx(2.04, abs=1e-5)


def test_value_loss_clipped_variant():
    v = torch.tensor([3.0])
    old = torch.tensor([2.0])
    targ = torch.tensor([3.0])
    # v_clip = 2.2 -> max((0)^2, (0.8)^2) = 0.64
    assert value_loss(v, targ, old_values=old, clip=0.2).item() == pytest.approx(0.64, abs=1e-5)


def test_diagnostics_hand_batch():
    log_ratio = torch.log(torch.tensor([1.125, 1.125, 0.75]))
    kl, clipfrac = approx_kl_and_clipfrac(log_ratio, 0.2)
    assert float(kl) == pytest.approx(0.01737, abs=1e-4)
    assert float(clipfrac) == pytest.approx(1 / 3, abs=1e-6)


def test_total_loss_hand_batch():
    _, _, new_logp, entropy, values, batch = _hand_batch()
    loss, stats = ppo_loss(new_logp, entropy, values, batch, _cfg())
    assert loss.item() == pytest.approx(1.8177, abs=1e-3)
    assert stats["policy_loss"] == pytest.approx(-0.21667, abs=1e-4)
    assert stats["value_loss"] == pytest.approx(2.04, abs=1e-4)
    assert stats["entropy"] == pytest.approx(0.5623, abs=1e-4)
    assert stats["approx_kl"] == pytest.approx(0.01737, abs=1e-4)
    assert stats["clipfrac"] == pytest.approx(1 / 3, abs=1e-6)


def test_total_loss_gradients_hand_batch():
    theta, w, new_logp, entropy, values, batch = _hand_batch()
    loss, _ = ppo_loss(new_logp, entropy, values, batch, _cfg(ent_coef=0.0))
    loss.backward()
    # dL/dtheta = -p(1-p)(1; -1); mẫu thứ ba bị cắt nên không đóng góp
    assert theta.grad.tolist() == pytest.approx([-0.1875, 0.1875], abs=1e-5)
    # dL/dw = 2 * mean(v - V_targ) = 2 * (-0.8 - 0.8 + 2.2) / 3
    assert w.grad.item() == pytest.approx(0.4, abs=1e-5)


def test_total_loss_normalised_advantages():
    _, _, new_logp, entropy, values, batch = _hand_batch()
    loss, stats = ppo_loss(new_logp, entropy, values, batch, _cfg(norm_adv=True))
    # độ lệch chuẩn tổng thể sqrt(2): A = (0,7071; 0,7071; -1,4142)
    s = 1 / math.sqrt(2)
    l_clip = (1.125 * s + 1.125 * s + 0.8 * (-2 * s)) / 3
    assert stats["policy_loss"] == pytest.approx(-l_clip, abs=1e-4)
    assert stats["value_loss"] == pytest.approx(2.04, abs=1e-4)  # V_targ luôn dùng A_raw


class _TinyAC(nn.Module):
    """Actor-critic tí hon trên đặc trưng véc-tơ, cùng giao diện với mô hình của đồ án."""

    def __init__(self, obs_dim=4, n_actions=3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, 32), nn.Tanh())
        self.pi, self.v = nn.Linear(32, n_actions), nn.Linear(32, 1)

    def forward(self, x):
        h = self.body(x.float())
        return self.pi(h), self.v(h).squeeze(-1)


def _rollout(model, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    obs = torch.randn(n, 4, generator=g)
    with torch.no_grad():
        logits, values = model(obs)
        dist = torch.distributions.Categorical(logits=logits)
        actions = torch.multinomial(dist.probs, 1, generator=g)[:, 0]
        logp = dist.log_prob(actions)
    adv = torch.randn(n, generator=g)
    return dict(obs=obs, actions=actions, old_logp=logp, old_values=values, advantages=adv, returns=values + adv)


def test_ppo_update_improves_surrogate_and_reports_stats():
    torch.manual_seed(0)
    model = _TinyAC()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = _rollout(model)
    before = [p.detach().clone() for p in model.parameters()]
    stats = ppo_update(model, opt, batch, _cfg(update_epochs=4, num_minibatches=4, vf_coef=0.5))
    for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac"):
        assert k in stats and math.isfinite(stats[k])
    assert stats["minibatch_updates"] == 16
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    # chính sách mới tăng xác suất của hành động có lợi thế dương
    with torch.no_grad():
        logits, _ = model(batch["obs"])
        new_logp = torch.distributions.Categorical(logits=logits).log_prob(batch["actions"])
    gain = ((new_logp - batch["old_logp"]).exp() * batch["advantages"]).mean()
    assert gain.item() > batch["advantages"].mean().item()


def test_ppo_update_kl_early_stop():
    torch.manual_seed(0)
    model = _TinyAC()
    opt = torch.optim.Adam(model.parameters(), lr=0.5)
    batch = _rollout(model)
    stats = ppo_update(model, opt, batch, _cfg(update_epochs=10, num_minibatches=4, target_kl=1e-9))
    # approxKL vượt ngưỡng ngay sau epoch đầu -> dừng các epoch còn lại
    assert stats["minibatch_updates"] < 40
