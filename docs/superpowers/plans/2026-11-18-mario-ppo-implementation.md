# Mario PPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the complete `mario_ppo` package (wrappers, Nature-CNN actor-critic, GAE, PPO update, training tricks, Colab/Kaggle configs, transfer learning, and report plotting scripts) to pass 100% of `pytest` suites and achieve $\ge 0.8$ `flag_rate` on levels `1-1` (25 pts), `1-4`/`4-1` (15 pts), and bonus level `7-1` (+10 pts) for the 4-member team (Việt, Sơn, Quang, Tín).

**Architecture:** Build directly on the official `mario_ppo` starter repository. Implement `FrameSkip`, `GrayResize`, `ScalarizeReward`, plus training-only wrappers (`NoProgressTruncation`, `MarioRewardShaping`) wired inside `if not eval_mode:` in `mario_ppo/env.py`; a shared 3-conv Nature-CNN `ActorCritic` in `mario_ppo/model.py`; vectorized backward GAE and minibatch clipped PPO in `mario_ppo/ppo.py`; and cross-platform multiprocessing + `init_ckpt` transfer learning + Colab/Kaggle configs and report plotting tools.

**Tech Stack:** Python $\ge 3.10$, PyTorch, Gymnasium (`mo-gymnasium==1.3.2`, `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`), OpenCV (`opencv-python-headless`), NumPy $< 2$, OmegaConf $\ge 2.3$, Matplotlib, Pytest.

**Spec:** [2026-11-18-mario-ppo-official-design.md](../specs/2026-11-18-mario-ppo-official-design.md)

## Global Constraints

- Do NOT modify `mario_ppo/evaluate.py` or the `eval_mode=True` code path of `mario_ppo/env.py`.
- All custom training tricks (`NoProgressTruncation`, `MarioRewardShaping`) MUST only be applied inside `if not eval_mode:` in `mario_ppo/env.py` and MUST be no-ops unless explicitly enabled via `env.*` config keys so `configs/default.yaml` behavior stays unchanged.
- `ActorCritic` input is strictly preprocessed image tensors `(B, C, 84, 84)`; never read NES RAM, `x_pos`, or `info` as network inputs.
- Preserve exact function and class signatures expected by `train.py`, `evaluate.py`, and `tests/test_*.py`.
- Cross-platform support: `mario_ppo/train.py` must use `context="spawn"` on Windows (`sys.platform == "win32"`) and `"forkserver"` on Linux/Colab/Kaggle so `pytest` passes on both local Windows machines and cloud Linux runtimes.

## Review Focus

1. **`NoProgressTruncation` when episode terminates simultaneously:** When `terminated=True` on the exact step `stalled` reaches `max_steps`, `truncated` must remain `False` (`not terminated`) so `compute_gae` does not bootstrap a terminal death/flag state from `final_values`.
2. **`MarioRewardShaping` vector reward immutability:** Modifying `reward[0]` for `flag_bonus` or `stuck_penalty` must copy `reward` as `np.float32` before mutating so underlying environment arrays are not corrupted, and must occur before `ScalarizeReward`.
3. **`ActorCritic` custom `hidden_dim` via `cfg.model` kwargs:** `train.py` and `evaluate.py` instantiate `ActorCritic(n_actions, in_channels=..., **cfg.get("model", {}))`; `ActorCritic.__init__` must accept `hidden_dim=512` (and ignore/validate optional kwargs) while keeping initial policy probabilities within `0.05` of uniform `1 / n_actions`.
4. **`compute_gae` truncation at rollout boundary `t = T - 1`:** When `truncated[T - 1, n] == 1`, `compute_gae` must bootstrap from `final_values[T - 1, n]` (the final observation of the truncated episode), NOT `last_value[n]` (which belongs to the newly reset episode under `AutoresetMode.SAME_STEP`).
5. **Transfer learning `init_ckpt` vs `resume` in `train.py`:** When fine-tuning `4-1` or `7-1` from a `1-1` checkpoint via `init_ckpt`, `train.py` must load only `model.load_state_dict(ck["model"])` while keeping `global_step = 0` and a fresh Adam optimizer so the learning rate schedule starts at `algo.lr` instead of decaying from step 4,000,000.

---

## File Structure & Team Ownership Map

| Subsystem | Owner(s) | Files Modified / Created | Test File |
|---|---|---|---|
| **Phân hệ 1: Wrappers & Environment Tricks** | **Việt & Tín** | Modify: `mario_ppo/wrappers.py`, `mario_ppo/env.py` | `tests/test_wrappers.py` |
| **Phân hệ 2: Actor-Critic Network** | **Sơn & Tín** | Modify: `mario_ppo/model.py` | `tests/test_model.py` |
| **Phân hệ 3: GAE & PPO Core** | **Quang & Tín** | Modify: `mario_ppo/ppo.py` | `tests/test_gae.py`, `tests/test_ppo.py` |
| **Phân hệ 4: Infra, Configs, Colab/Kaggle & Plotting** | **Tín (làm riêng)** | Modify: `mario_ppo/train.py`, `pyproject.toml`<br>Create: `configs/1-1.yaml`, `configs/1-4.yaml`, `configs/4-1.yaml`, `configs/7-1.yaml`, `configs/ablation_gamma099.yaml`, `configs/ablation_notricks.yaml`, `scripts/plot_report.py`, `notebooks/train_colab_kaggle.ipynb` | `tests/test_pipeline.py`, `tests/test_plot_report.py` |

---

### Task 1: Environment Wrappers & Training Tricks (Phân hệ 1 — Việt & Tín)

**Files:**
- Modify: `mario_ppo/wrappers.py:122-179`
- Modify: `mario_ppo/env.py:15-16,41-47`
- Test: `tests/test_wrappers.py`

**Interfaces:**
- Consumes: `gym.Wrapper`, `gym.ObservationWrapper`, `cv2.cvtColor`, `cv2.resize`
- Produces:
  - `FrameSkip(env, skip=4)` -> `step(action) -> (obs: np.ndarray[uint8], total_reward: np.ndarray[float32], terminated: bool, truncated: bool, info: dict)`
  - `GrayResize(env, size=84)` -> `observation(obs) -> np.ndarray[(size, size), uint8]`
  - `ScalarizeReward(env, weights)` -> `step(action) -> (obs, reward: float, terminated, truncated, info)` with `info["vector_reward"]: np.ndarray[float32]`
  - `NoProgressTruncation(env, max_steps=180)` -> sets `truncated = True` when `x_pos` does not exceed `self._max_x` for `max_steps` consecutive steps and `not terminated`
  - `MarioRewardShaping(env, flag_bonus=0.0, stuck_penalty=0.0)` -> adds `flag_bonus` to `reward[0]` when `info.get("flag_get", False)` is `True`, and adds `stuck_penalty` to `reward[0]` when `reward[0] <= 0 and not terminated`

- [ ] **Step 1: Add Review Focus unit tests for `NoProgressTruncation` and `MarioRewardShaping` in `tests/test_wrappers.py`**

Append to `tests/test_wrappers.py`:
```python
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
        if self.x_seq[min(self.k - 1, len(self.x_seq) - 1)] == 0:
            reward[0] = 0.0
        return obs, reward, terminated, truncated, info


def test_no_progress_truncation_cuts_stalled_episode_but_not_terminal():
    env = NoProgressTruncation(_StuckMarioEnv([10, 10, 10, 10]), max_steps=2)
    env.reset()
    _, _, term1, trunc1, _ = env.step(0)  # x=10 (new max_x) -> stalled=0
    _, _, term2, trunc2, _ = env.step(0)  # x=10 -> stalled=1
    _, _, term3, trunc3, _ = env.step(0)  # x=10 -> stalled=2 -> truncated!
    assert not trunc1 and not trunc2 and trunc3 and not term3

    # If episode terminates on the exact step stalled hits max_steps, truncated must stay False
    env_term = NoProgressTruncation(_StuckMarioEnv([10, 10, 10], end_at=3), max_steps=2)
    env_term.reset()
    env_term.step(0)
    env_term.step(0)
    _, _, term, trunc, _ = env_term.step(0)
    assert term and not trunc


def test_mario_reward_shaping_modifies_x_component_before_scalarize():
    env = MarioRewardShaping(_StuckMarioEnv([0, 20], flag_at=2), flag_bonus=5.0, stuck_penalty=-0.25)
    env.reset()
    _, r1, _, _, _ = env.step(0)  # x=0 -> reward[0] was 0.0 -> -0.25
    _, r2, _, _, _ = env.step(0)  # flag_get=True -> reward[0] was 2.0 -> 2.0 + 5.0 = 7.0
    assert r1[0] == pytest.approx(-0.25)
    assert r2[0] == pytest.approx(7.0)
```

- [ ] **Step 2: Run `pytest tests/test_wrappers.py` to verify tests fail**

Run: `pytest -q tests/test_wrappers.py`
Expected: FAIL (`ImportError` or `NotImplementedError: TODO: FrameSkip.step`)

- [ ] **Step 3: Implement `FrameSkip`, `GrayResize`, `ScalarizeReward`, `NoProgressTruncation`, and `MarioRewardShaping` in `mario_ppo/wrappers.py` and wire into `mario_ppo/env.py`**

In `mario_ppo/wrappers.py`:
- `FrameSkip.step(self, action)`: loop up to `self.skip` times calling `self.env.step(action)`, accumulate `total_reward` as `np.float32`, keep the last two observations (`prev_obs`, `last_obs`), break on `terminated or truncated`, and return `np.maximum(prev_obs, last_obs)` if $\ge 2$ steps ran else `last_obs`.
- `GrayResize.__init__(self, env, size=84)`: set `self.observation_space = gym.spaces.Box(0, 255, (size, size), dtype=np.uint8)`.
- `GrayResize.observation(self, obs)`: convert via `cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)` and resize via `cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)`.
- `ScalarizeReward.__init__(self, env, weights)`: determine expected length `n_obj = env.unwrapped.reward_space.shape[0]` if `getattr(env.unwrapped, "reward_space", None) is not None` else `5`; raise `ValueError` if `len(weights) != n_obj`; store `self.weights = np.asarray(weights, dtype=np.float32)`.
- `ScalarizeReward.step(self, action)`: store `vec = np.asarray(reward, dtype=np.float32)` in `info["vector_reward"]` and return `float(np.dot(self.weights, vec))`.
- Implement `NoProgressTruncation(env, max_steps=180)` and `MarioRewardShaping(env, flag_bonus=0.0, stuck_penalty=0.0)` at the bottom of `mario_ppo/wrappers.py` and wire them inside `if not eval_mode:` in `mario_ppo/env.py` before `RandomNoopStart`.

- [ ] **Step 4: Run `pytest tests/test_wrappers.py` to verify all wrapper tests pass**

Run: `pytest -v tests/test_wrappers.py`
Expected: PASS (9/9 passed)

- [ ] **Step 5: Commit Task 1**

```bash
git add mario_ppo/wrappers.py mario_ppo/env.py tests/test_wrappers.py
git commit -m "feat(wrappers): implement FrameSkip, GrayResize, ScalarizeReward and training tricks"
```

---

### Task 2: Shared Nature-CNN `ActorCritic` Network (Phân hệ 2 — Sơn & Tín)

**Files:**
- Modify: `mario_ppo/model.py:21-37`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `_init(layer, gain=np.sqrt(2))` in `mario_ppo/model.py`
- Produces:
  - `ActorCritic(n_actions: int, in_channels: int = 4, hidden_dim: int = 512)`
  - `ActorCritic.forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` returning `(logits (B, n_actions) float32, value (B,) float32)`

- [ ] **Step 1: Add Review Focus test for custom `hidden_dim` and `float32` / `uint8` normalization in `tests/test_model.py`**

Append to `tests/test_model.py`:
```python
def test_custom_hidden_dim_and_uniform_init():
    model = ActorCritic(n_actions=7, in_channels=4, hidden_dim=256)
    logits, value = model(_obs(batch=8))
    assert logits.shape == (8, 7) and value.shape == (8,)
    probs = torch.softmax(logits, dim=-1)
    assert float((probs - 1 / 7).abs().max()) < 0.05
```

- [ ] **Step 2: Run `pytest tests/test_model.py` to verify tests fail**

Run: `pytest -q tests/test_model.py`
Expected: FAIL (`NotImplementedError: TODO: ActorCritic.__init__`)

- [ ] **Step 3: Implement `ActorCritic.__init__` and `ActorCritic.forward` in `mario_ppo/model.py`**

In `mario_ppo/model.py`:
- `self.trunk = nn.Sequential(_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(), _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(), _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), nn.Flatten(), _init(nn.Linear(64 * 7 * 7, hidden_dim)), nn.ReLU())`
- `self.policy = _init(nn.Linear(hidden_dim, n_actions), gain=0.01)`
- `self.value = _init(nn.Linear(hidden_dim, 1), gain=1.0)`
- In `forward(self, obs)`: compute `h = self.trunk(obs.to(dtype=torch.float32) / 255.0)` and return `self.policy(h), self.value(h).squeeze(-1)`.

- [ ] **Step 4: Run `pytest tests/test_model.py` to verify all model tests pass**

Run: `pytest -v tests/test_model.py`
Expected: PASS (6/6 passed)

- [ ] **Step 5: Commit Task 2**

```bash
git add mario_ppo/model.py tests/test_model.py
git commit -m "feat(model): implement shared Nature-CNN ActorCritic with orthogonal init"
```

---

### Task 3: Generalized Advantage Estimation (GAE) & Clipped PPO Update (Phân hệ 3 — Quang & Tín)

**Files:**
- Modify: `mario_ppo/ppo.py:8-90`
- Test: `tests/test_gae.py`, `tests/test_ppo.py`

**Interfaces:**
- Produces:
  - `compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam) -> tuple[torch.Tensor, torch.Tensor]`
  - `clipped_policy_loss(ratio, advantages, clip) -> torch.Tensor`
  - `value_loss(values, returns, old_values=None, clip=None) -> torch.Tensor`
  - `approx_kl_and_clipfrac(log_ratio, clip) -> tuple[float, float]`
  - `ppo_loss(new_logp, entropy, values, batch, cfg) -> tuple[torch.Tensor, dict[str, float]]`
  - `ppo_update(model, optimizer, batch, cfg) -> dict[str, float | int]`

- [ ] **Step 1: Add Review Focus test for multi-env mixed termination/truncation at rollout boundary in `tests/test_gae.py`**

Append to `tests/test_gae.py`:
```python
def test_mixed_boundary_truncated_vs_continuing_across_envs():
    # Env 0 is truncated at t=1 (uses final_values[1, 0]=4.0); Env 1 continues at t=1 (uses last_value[1]=10.0)
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    terminated = torch.zeros(2, 2)
    truncated = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    final_values = torch.tensor([[0.0, 0.0], [4.0, 999.0]])
    last_value = torch.tensor([123.0, 10.0])
    adv, ret = compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma=0.9, lam=0.8)
    # Env 0 at t=1: 2 + 0.9 * 4 - 3 = 2.6; Env 1 at t=1: 2 + 0.9 * 10 - 3 = 8.0
    assert adv[1, 0].item() == pytest.approx(2.6, abs=1e-5)
    assert adv[1, 1].item() == pytest.approx(8.0, abs=1e-5)
```

- [ ] **Step 2: Run `pytest tests/test_gae.py tests/test_ppo.py` to verify tests fail**

Run: `pytest -q tests/test_gae.py tests/test_ppo.py`
Expected: FAIL (`NotImplementedError: TODO: compute_gae`)

- [ ] **Step 3: Implement `compute_gae`, `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, and `ppo_update` in `mario_ppo/ppo.py`**

Follow the exact formulas in Spec Section 5.3:
- In `compute_gae`: loop `for t in reversed(range(T)):`
  - `next_val = last_value if t == T - 1 else values[t + 1]`
  - `next_val = torch.where(truncated[t] > 0.5, final_values[t], next_val)`
  - `non_terminal = 1.0 - terminated[t]`
  - `continuing = non_terminal * (1.0 - truncated[t])`
  - `delta = rewards[t] + gamma * non_terminal * next_val - values[t]`
  - `gae = delta + gamma * lam * continuing * gae`
  - `advantages[t] = gae`
  - Return `advantages, advantages + values`.
- In `ppo_loss`: when `cfg.norm_adv` is `True`, normalize `adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)`. Pass `clip = cfg.clip_coef if getattr(cfg, "clip_vloss", False) else None` to `value_loss`.
- In `ppo_update`: shuffle indices with `torch.randperm(B, device=batch["obs"].device)` each epoch, split into `cfg.num_minibatches` slices, update parameters with `clip_grad_norm_(model.parameters(), cfg.max_grad_norm)`, and break out of the epoch loop if `cfg.target_kl is not None and epoch_kl_mean > cfg.target_kl`.

- [ ] **Step 4: Run `pytest tests/test_gae.py tests/test_ppo.py` to verify all GAE and PPO tests pass**

Run: `pytest -v tests/test_gae.py tests/test_ppo.py`
Expected: PASS (19/19 passed)

- [ ] **Step 5: Commit Task 3**

```bash
git add mario_ppo/ppo.py tests/test_gae.py tests/test_ppo.py
git commit -m "feat(ppo): implement GAE with truncation bootstrap and clipped PPO update"
```

---

### Task 4: Cross-Platform Training, Transfer Learning, Stage Configs & Report Tooling (Phân hệ 4 — Tín làm riêng)

**Files:**
- Modify: `mario_ppo/train.py:27-34,88-96`
- Modify: `configs/default.yaml` (keep defaults intact, add `init_ckpt: null` optional key)
- Create: `configs/1-1.yaml`, `configs/1-4.yaml`, `configs/4-1.yaml`, `configs/7-1.yaml`, `configs/ablation_gamma099.yaml`, `configs/ablation_notricks.yaml`
- Create: `scripts/plot_report.py`
- Create: `notebooks/train_colab_kaggle.ipynb`
- Test: `tests/test_pipeline.py`, `tests/test_plot_report.py`

**Interfaces:**
- Consumes: `mario_ppo.env.make_env`, `mario_ppo.model.ActorCritic`, `mario_ppo.ppo.compute_gae`, `mario_ppo.ppo.ppo_update`
- Produces:
  - Cross-platform `_make_vec_env` (`"spawn"` on Windows, `"forkserver"` on POSIX)
  - Transfer learning key `init_ckpt` in `train(cfg)`: loads `ck["model"]` weights while keeping `global_step = 0` and fresh optimizer
  - `scripts/plot_report.py`: CLI and callable functions `plot_learning_curves(progress_csv, out_path)`, `plot_ablation_comparison(run_dirs_a, run_dirs_b, label_a, label_b, out_path)`, `plot_failure_x_distribution(eval_jsons, level, out_path)`

- [ ] **Step 1: Add Review Focus test for `init_ckpt` transfer learning in `tests/test_pipeline.py` and plotting test in `tests/test_plot_report.py`**

Append to `tests/test_pipeline.py`:
```python
def test_train_init_ckpt_loads_weights_at_step_zero(tmp_path):
    from mario_ppo.train import train
    run1 = train(_tiny_cfg(tmp_path / "src"))
    ckpt1 = sorted((run1 / "checkpoints").glob("*.pt"))[-1]

    cfg2 = _tiny_cfg(tmp_path / "transfer")
    cfg2.init_ckpt = str(ckpt1)
    cfg2.algo.total_steps = 32
    run2 = train(cfg2)
    ck2 = torch.load(sorted((run2 / "checkpoints").glob("*.pt"))[-1], weights_only=False)
    assert ck2["global_step"] == 32
```

Create `tests/test_plot_report.py`:
```python
import json
from pathlib import Path
from scripts.plot_report import (
    plot_ablation_comparison,
    plot_failure_x_distribution,
    plot_learning_curves,
)


def _write_dummy_progress(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "global_step,sps,lr,policy_loss,value_loss,entropy,approx_kl,clipfrac,minibatch_updates,episodes,ep_return,ep_length,x_pos,flag_rate\n"
        "4096,300.0,0.00025,-0.01,1.2,1.8,0.005,0.05,32,10,15.0,120.0,850.0,0.0\n"
        "8192,310.0,0.00020,-0.02,0.8,1.5,0.008,0.08,32,20,42.0,200.0,3161.0,0.85\n",
        encoding="utf-8",
    )


def test_plot_functions_create_image_files(tmp_path):
    p1 = tmp_path / "r1" / "progress.csv"
    p2 = tmp_path / "r2" / "progress.csv"
    _write_dummy_progress(p1)
    _write_dummy_progress(p2)

    out_curves = tmp_path / "curves.png"
    plot_learning_curves(p1, out_curves)
    assert out_curves.exists() and out_curves.stat().st_size > 0

    out_ablation = tmp_path / "ablation.png"
    plot_ablation_comparison([p1], [p2], "gamma=0.90", "gamma=0.99", out_ablation)
    assert out_ablation.exists() and out_ablation.stat().st_size > 0

    eval_json = tmp_path / "eval.json"
    eval_json.write_text(
        json.dumps({"level": "1-1", "flag_rate": 0.85, "x_pos": [1200, 3161, 3161, 1800]}),
        encoding="utf-8",
    )
    out_fail = tmp_path / "failure_x.png"
    plot_failure_x_distribution([eval_json], "1-1", out_fail)
    assert out_fail.exists() and out_fail.stat().st_size > 0
```

- [ ] **Step 2: Run `pytest tests/test_pipeline.py tests/test_plot_report.py` to verify failure**

Run: `pytest -q tests/test_plot_report.py`
Expected: FAIL (`ModuleNotFoundError: No module named 'scripts'`)

- [ ] **Step 3: Update `mario_ppo/train.py`, create `configs/*.yaml`, `scripts/plot_report.py`, and `notebooks/train_colab_kaggle.ipynb`**

1. In `mario_ppo/train.py`:
   - In `_make_vec_env`: use `ctx = "spawn" if sys.platform == "win32" else "forkserver"` and pass `context=ctx` to `AsyncVectorEnv`.
   - After `global_step = 0`: if `cfg.get("init_ckpt") and not cfg.resume`, load `ck = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)` and call `model.load_state_dict(ck["model"])`.
2. In `configs/default.yaml`: add `init_ckpt: null` under `resume: null`.
3. Create stage and ablation YAML files in `configs/`:
   - `configs/1-1.yaml` (`level: "1-1"`, `num_envs: 8`, `num_steps: 512`, `total_steps: 4_000_000`, `gamma: 0.9`, `max_no_progress_steps: 180`, `flag_bonus: 5.0`)
   - `configs/1-4.yaml` (`level: "1-4"`, `num_envs: 8`, `num_steps: 512`, `total_steps: 6_000_000`, `gamma: 0.95`, `max_no_progress_steps: 220`, `flag_bonus: 8.0`)
   - `configs/4-1.yaml` (`level: "4-1"`, `num_envs: 8`, `num_steps: 512`, `total_steps: 6_000_000`, `gamma: 0.95`, `max_no_progress_steps: 180`, `flag_bonus: 6.0`)
   - `configs/7-1.yaml` (`level: "7-1"`, `num_envs: 8`, `num_steps: 512`, `total_steps: 8_000_000`, `gamma: 0.95`, `max_no_progress_steps: 200`, `flag_bonus: 8.0`)
   - `configs/ablation_gamma099.yaml` (identical to `configs/1-1.yaml` except `algo.gamma: 0.99`, `algo.total_steps: 3_000_000`)
   - `configs/ablation_notricks.yaml` (identical to `configs/1-1.yaml` except `env.max_no_progress_steps: 0`, `env.flag_bonus: 0.0`, `algo.total_steps: 3_000_000`)
4. Create `scripts/plot_report.py` using `csv`, `json`, `numpy`, and `matplotlib` (`Agg` backend) implementing `plot_learning_curves`, `plot_ablation_comparison`, `plot_failure_x_distribution`, and a CLI `main()`.
5. Create `notebooks/train_colab_kaggle.ipynb` with ready-to-run cells for each of the 4 members (Việt: `1-1`, Sơn: `1-4`/`4-1`, Quang: Ablation 2 seeds, Tín: `7-1` + evaluation & report plots).

- [ ] **Step 4: Run the full `pytest` suite across the entire repository**

Run: `pytest -v`
Expected: PASS (100% of tests in `test_wrappers.py`, `test_model.py`, `test_gae.py`, `test_ppo.py`, `test_pipeline.py`, `test_plot_report.py` pass)

- [ ] **Step 5: Commit Task 4**

```bash
git add mario_ppo/train.py configs/ scripts/plot_report.py notebooks/train_colab_kaggle.ipynb tests/test_pipeline.py tests/test_plot_report.py
git commit -m "feat(infra): add cross-platform vec env, transfer init_ckpt, stage configs, and report plotter"
```
