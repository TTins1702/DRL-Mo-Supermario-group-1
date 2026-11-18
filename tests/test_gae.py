"""Kiểm tra compute_gae: lợi thế GAE và mục tiêu của critic trên một đoạn thu.

Quy ước mảng: chỉ số hàng là thời gian t = 0..h-1, chỉ số cột là môi trường n = 0..N-1.
"""
import pytest
import torch

from mario_ppo.ppo import compute_gae


def _two_step(case):
    # h = 2, N = 1, gamma 0.9, lambda 0.8, R_1 = 1, R_2 = 2, v(S_0) = 2, v(S_1) = 3
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[2.0], [3.0]])
    terminated = torch.zeros(2, 1)
    truncated = torch.zeros(2, 1)
    final_values = torch.zeros(2, 1)
    if case == "terminal":
        terminated[1, 0] = 1.0
    else:  # truncated at step 2, v(S_2^final) = 4
        truncated[1, 0] = 1.0
        final_values[1, 0] = 4.0
    last_value = torch.tensor([123.0])  # must be ignored: the episode ended at the boundary
    return rewards, values, terminated, truncated, final_values, last_value


def test_two_step_terminal():
    adv, v_targ = compute_gae(*_two_step("terminal"), gamma=0.9, lam=0.8)
    assert adv[:, 0].tolist() == pytest.approx([0.98, -1.0], abs=1e-5)
    assert v_targ[:, 0].tolist() == pytest.approx([2.98, 2.0], abs=1e-5)


def test_two_step_truncated_bootstraps_from_final_obs():
    adv, v_targ = compute_gae(*_two_step("truncated"), gamma=0.9, lam=0.8)
    assert adv[:, 0].tolist() == pytest.approx([3.572, 2.6], abs=1e-5)
    assert v_targ[0, 0].item() == pytest.approx(5.572, abs=1e-5)  # truncated lambda-return


def test_rollout_boundary_bootstraps_from_last_value():
    # Không kết thúc, không cắt: bước cuối của đoạn coi như cắt ngắn với V'_h = V_h.
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[2.0], [3.0]])
    zeros = torch.zeros(2, 1)
    adv, _ = compute_gae(rewards, values, zeros, zeros, zeros, torch.tensor([4.0]), gamma=0.9, lam=0.8)
    assert adv[:, 0].tolist() == pytest.approx([3.572, 2.6], abs=1e-5)


def test_truncation_mid_rollout_cuts_recursion():
    # Lượt bị cắt sau bước t = 0; bước t = 1 thuộc lượt mới, không được truyền ngược sang t = 0.
    rewards = torch.tensor([[1.0], [10.0]])
    values = torch.tensor([[0.0], [0.0]])
    terminated = torch.zeros(2, 1)
    truncated = torch.tensor([[1.0], [0.0]])
    final_values = torch.tensor([[5.0], [0.0]])
    adv, _ = compute_gae(rewards, values, terminated, truncated, final_values, torch.tensor([0.0]),
                         gamma=0.5, lam=1.0)
    assert adv[0, 0].item() == pytest.approx(1.0 + 0.5 * 5.0)
    assert adv[1, 0].item() == pytest.approx(10.0)


def test_terminal_mid_rollout_no_bootstrap():
    rewards = torch.tensor([[1.0], [10.0]])
    values = torch.tensor([[0.5], [7.0]])
    terminated = torch.tensor([[1.0], [0.0]])
    zeros = torch.zeros(2, 1)
    adv, _ = compute_gae(rewards, values, terminated, zeros, zeros, torch.tensor([0.0]), gamma=0.9, lam=0.95)
    assert adv[0, 0].item() == pytest.approx(1.0 - 0.5)


def test_lambda_zero_is_td_error():
    torch.manual_seed(0)
    T, N = 5, 3
    r, v = torch.randn(T, N), torch.randn(T, N)
    last = torch.randn(N)
    zeros = torch.zeros(T, N)
    adv, _ = compute_gae(r, v, zeros, zeros, zeros, last, gamma=0.9, lam=0.0)
    v_next = torch.cat([v[1:], last[None]], dim=0)
    assert torch.allclose(adv, r + 0.9 * v_next - v, atol=1e-6)


def test_lambda_one_full_episode_is_monte_carlo():
    # lambda = 1, cả lượt kết thúc trong đoạn: A_t = G_t - v(S_t).
    r = torch.tensor([[1.0], [2.0], [3.0]])
    v = torch.tensor([[0.3], [0.2], [0.1]])
    term = torch.tensor([[0.0], [0.0], [1.0]])
    zeros = torch.zeros(3, 1)
    adv, v_targ = compute_gae(r, v, term, zeros, zeros, torch.tensor([9.0]), gamma=0.5, lam=1.0)
    G = torch.tensor([1 + 0.5 * 2 + 0.25 * 3, 2 + 0.5 * 3, 3.0])
    assert torch.allclose(v_targ[:, 0], G, atol=1e-6)
    assert torch.allclose(adv[:, 0], G - v[:, 0], atol=1e-6)


def test_shapes_and_independent_envs():
    torch.manual_seed(1)
    T, N = 8, 4
    r, v = torch.randn(T, N), torch.randn(T, N)
    term = (torch.rand(T, N) < 0.2).float()
    trunc = ((torch.rand(T, N) < 0.2).float()) * (1 - term)
    fv, last = torch.randn(T, N), torch.randn(N)
    adv, v_targ = compute_gae(r, v, term, trunc, fv, last, gamma=0.9, lam=0.95)
    assert adv.shape == (T, N) and v_targ.shape == (T, N)
    assert torch.allclose(v_targ, adv + v)
    # mỗi cột chỉ phụ thuộc dữ liệu của môi trường đó
    for n in range(N):
        a1, _ = compute_gae(r[:, n:n + 1], v[:, n:n + 1], term[:, n:n + 1], trunc[:, n:n + 1],
                            fv[:, n:n + 1], last[n:n + 1], gamma=0.9, lam=0.95)
        assert torch.allclose(a1[:, 0], adv[:, n], atol=1e-6)
