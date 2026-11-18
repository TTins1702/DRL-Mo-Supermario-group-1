"""Kiểm tra mạng actor-critic: giao diện forward / act mà vòng huấn luyện và script đánh giá dùng."""
import torch

from mario_ppo.model import ActorCritic


def _obs(batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (batch, 4, 84, 84), dtype=torch.uint8, generator=g)


def test_forward_shapes():
    model = ActorCritic(n_actions=7)
    logits, value = model(_obs())
    assert logits.shape == (5, 7) and value.shape == (5,)
    assert logits.dtype == torch.float32 and value.dtype == torch.float32


def test_forward_accepts_other_action_counts_and_channels():
    model = ActorCritic(n_actions=12, in_channels=2)
    logits, value = model(_obs()[:, :2])
    assert logits.shape == (5, 12) and value.shape == (5,)


def test_act_samples_valid_actions_with_matching_logp_and_value():
    torch.manual_seed(0)
    model = ActorCritic(n_actions=7)
    obs = _obs(batch=16)
    action, logp, value = model.act(obs)
    assert action.shape == (16,) and action.dtype == torch.long
    assert int(action.min()) >= 0 and int(action.max()) < 7
    with torch.no_grad():
        logits, v = model(obs)
    expected = torch.log_softmax(logits, dim=-1).gather(1, action[:, None])[:, 0]
    assert torch.allclose(logp, expected, atol=1e-5)
    assert torch.allclose(value, v, atol=1e-5)
    assert not logp.requires_grad


def test_initial_policy_is_close_to_uniform():
    # Khởi tạo đầu chính sách nhỏ -> mọi hành động gần đều nhau, khám phá tốt ở đầu huấn luyện.
    model = ActorCritic(n_actions=7)
    logits, _ = model(_obs(batch=32))
    probs = torch.softmax(logits, dim=-1)
    assert float((probs - 1 / 7).abs().max()) < 0.05


def test_gradients_reach_every_parameter():
    model = ActorCritic(n_actions=7)
    logits, value = model(_obs())
    (logits.logsumexp(-1).mean() + value.pow(2).mean()).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
