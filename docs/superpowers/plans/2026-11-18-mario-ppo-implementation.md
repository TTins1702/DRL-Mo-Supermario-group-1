# Kế hoạch Triển khai Đồ án: Chinh phục Super Mario Bros bằng PPO (`mario_ppo`)

> **Hướng dẫn thực hiện cho Nhóm 1 (Việt, Sơn, Quang, Tín):** Triển khai tuần tự theo từng đầu việc (Task) bên dưới bằng quy trình kiểm thử trước (TDD). Hoàn thành bước nào đánh dấu `- [x]` bước đó và kiểm tra toàn bộ `pytest` trước khi chuyển sang giai đoạn huấn luyện trên Colab/Kaggle.

- **Mục tiêu:** Hoàn thiện trọn vẹn gói `mario_ppo` (các wrapper tiền xử lý & mẹo huấn luyện, mạng Actor-Critic Nature-CNN, GAE, vòng cập nhật PPO, hỗ trợ đa nền tảng Windows/Linux, học chuyển giao `init_ckpt`, bộ cấu hình Colab/Kaggle và script vẽ biểu đồ báo cáo) để vượt qua 100% bộ kiểm thử `pytest` và đạt tỉ lệ chạm cờ `flag_rate >= 0.8` ở các màn `1-1` (25đ), `1-4`/`4-1` (15đ) và màn thưởng ngoài danh sách `7-1` (+10đ).
- **Kiến trúc tổng thể:** Phát triển trực tiếp trên kho mã khởi đầu `mario_ppo`. Cài đặt `FrameSkip`, `GrayResize`, `ScalarizeReward` cùng 2 wrapper mẹo huấn luyện (`NoProgressTruncation`, `MarioRewardShaping`) đặt trong nhánh `if not eval_mode:` của `mario_ppo/env.py`; xây dựng mạng chia sẻ thân CNN `ActorCritic` trong `mario_ppo/model.py`; cài đặt GAE (tách riêng `terminated` và `truncated`) và hàm mất mát cắt ngưỡng PPO trong `mario_ppo/ppo.py`; hoàn thiện hạ tầng chạy song song đa nền tảng, học chuyển giao `init_ckpt`, notebook Colab/Kaggle và công cụ vẽ biểu đồ báo cáo.
- **Công nghệ sử dụng:** Python $\ge 3.10$, PyTorch, Gymnasium (`mo-gymnasium==1.3.2`, `gym-super-mario-bros==7.4.0`, `nes-py==8.2.1`), OpenCV (`opencv-python-headless`), NumPy $< 2$, OmegaConf $\ge 2.3$, Matplotlib, Pytest.
- **Tài liệu thiết kế tham chiếu:** [2026-11-18-mario-ppo-official-design.md](../specs/2026-11-18-mario-ppo-official-design.md)

---

## Ràng buộc Toàn dự án

- **Không sửa** `mario_ppo/evaluate.py` và nhánh `eval_mode=True` trong `mario_ppo/env.py`.
- Toàn bộ mẹo huấn luyện (`NoProgressTruncation`, `MarioRewardShaping`) **chỉ được bật** bên trong khối `if not eval_mode:` của `mario_ppo/env.py` và mặc định không kích hoạt nếu không khai báo trong file cấu hình, đảm bảo `configs/default.yaml` giữ nguyên hành vi gốc.
- Đầu vào của mạng `ActorCritic` **chỉ là tensor khung hình đã tiền xử lý** có shape `(B, C, 84, 84)`; tuyệt đối không đọc RAM, tọa độ `x_pos` hay `info` đưa vào mạng.
- Giữ nguyên tên lớp, tên hàm và chữ ký tham số mà `train.py`, `evaluate.py` và bộ `tests/test_*.py` đang gọi.
- Hỗ trợ đa nền tảng: `mario_ppo/train.py` tự động chọn `context="spawn"` trên Windows (`sys.platform == "win32"`) và `"forkserver"` trên Linux/Colab/Kaggle để bộ kiểm thử đầu-cuối chạy thông suốt trên mọi máy của 4 thành viên.

---

## Các Tình huống Biên cần Kiểm soát Đặc biệt

1. **`NoProgressTruncation` khi lượt kết thúc thật ở đúng bước kẹt cuối:** Nếu ở bước thứ `max_steps` bị đứng im mà Mario đồng thời chết hoặc chạm cờ (`terminated=True`), biến `truncated` bắt buộc phải giữ nguyên là `False` (`not terminated`) để `compute_gae` không bootstrap nhầm trạng thái kết thúc từ `final_values`.
2. **Tính bất biến của véc-tơ phần thưởng trong `MarioRewardShaping`:** Khi cộng `flag_bonus` hoặc `stuck_penalty` vào `reward[0]`, phải tạo bản sao `np.array(reward, dtype=np.float32, copy=True)` trước khi sửa để không làm sai lệch dữ liệu gốc, và phải đặt wrapper này trước `ScalarizeReward`.
3. **Truyền tham số `hidden_dim` tùy chỉnh từ `cfg.model` vào `ActorCritic`:** Do `train.py` và `evaluate.py` gọi `ActorCritic(n_actions, in_channels=..., **cfg.get("model", {}))`, hàm `__init__` cần hỗ trợ `hidden_dim=512` nhưng vẫn đảm bảo xác suất chính sách khởi tạo nằm trong sai số `< 0.05` quanh phân phối đều `1 / n_actions`.
4. **Bootstrap tại biên đoạn thu `t = T - 1` trong `compute_gae`:** Khi `truncated[T - 1, n] == 1`, `compute_gae` phải bootstrap từ `final_values[T - 1, n]` (quan sát cuối của lượt vừa bị cắt), tuyệt đối không dùng `last_value[n]` (vì với `AutoresetMode.SAME_STEP`, `last_value` thuộc về lượt mới vừa reset).
5. **Phân biệt `init_ckpt` (học chuyển giao) và `resume` (chạy tiếp) trong `train.py`:** Khi huấn luyện chuyển giao sang màn `4-1` hoặc `7-1` từ checkpoint `1-1` qua `init_ckpt`, `train.py` chỉ nạp trọng số mô hình `model.load_state_dict(ck["model"])`, giữ `global_step = 0` và khởi tạo bộ tối ưu Adam mới hoàn toàn để lịch giảm tốc độ học (`linear_schedule`) chạy đủ từ đầu.

---

## Sơ đồ Cấu trúc Tệp & Phân công Nhóm 4 Thành viên

| Phân hệ | Thành viên phụ trách | Tệp chỉnh sửa / tạo mới | Bộ kiểm thử |
|---|---|---|---|
| **Phân hệ 1: Wrapper & Mẹo huấn luyện** | **Việt & Tín** | Chỉnh sửa: `mario_ppo/wrappers.py`, `mario_ppo/env.py` | `tests/test_wrappers.py` |
| **Phân hệ 2: Mạng Actor-Critic** | **Sơn & Tín** | Chỉnh sửa: `mario_ppo/model.py` | `tests/test_model.py` |
| **Phân hệ 3: GAE & Vòng cập nhật PPO** | **Quang & Tín** | Chỉnh sửa: `mario_ppo/ppo.py` | `tests/test_gae.py`, `tests/test_ppo.py` |
| **Phân hệ 4: Hạ tầng Colab/Kaggle, Cấu hình & Biểu đồ** | **Tín (làm riêng)** | Chỉnh sửa: `mario_ppo/train.py`, `configs/default.yaml`<br>Tạo mới: `configs/1-1.yaml`, `configs/1-4.yaml`, `configs/4-1.yaml`, `configs/7-1.yaml`, `configs/ablation_gamma099.yaml`, `configs/ablation_notricks.yaml`, `scripts/plot_report.py`, `notebooks/train_colab_kaggle.ipynb` | `tests/test_pipeline.py`, `tests/test_plot_report.py` |

---

### Task 1: Hoàn thiện Chuỗi Wrapper & Mẹo Huấn luyện (Phân hệ 1 — Việt & Tín)

**Tệp liên quan:**
- Chỉnh sửa: `mario_ppo/wrappers.py:122-179`
- Chỉnh sửa: `mario_ppo/env.py:15-16,41-47`
- Kiểm thử: `tests/test_wrappers.py`

**Giao diện đầu vào / đầu ra:**
- Đầu vào: `gym.Wrapper`, `gym.ObservationWrapper`, `cv2.cvtColor`, `cv2.resize`
- Đầu ra:
  - `FrameSkip(env, skip=4)` $\rightarrow$ `step(action)` trả `(obs: np.ndarray[uint8], total_reward: np.ndarray[float32], terminated: bool, truncated: bool, info: dict)`
  - `GrayResize(env, size=84)` $\rightarrow$ `observation(obs)` trả `np.ndarray[(size, size), uint8]`
  - `ScalarizeReward(env, weights)` $\rightarrow$ `step(action)` trả `(obs, reward: float, terminated, truncated, info)` kèm `info["vector_reward"]: np.ndarray[float32]`
  - `NoProgressTruncation(env, max_steps=180)` $\rightarrow$ đặt `truncated = True` khi `x_pos` không vượt `self._max_x` trong `max_steps` bước liên tiếp và `not terminated`
  - `MarioRewardShaping(env, flag_bonus=0.0, stuck_penalty=0.0)` $\rightarrow$ cộng `flag_bonus` vào `reward[0]` khi `info.get("flag_get", False)` là `True`, và cộng `stuck_penalty` vào `reward[0]` khi `reward[0] <= 0 and not terminated`

- [ ] **Bước 1: Viết bài kiểm thử cho `NoProgressTruncation` và `MarioRewardShaping` trong `tests/test_wrappers.py`**

Thêm vào cuối tệp `tests/test_wrappers.py`:
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
```

- [ ] **Bước 2: Chạy thử `pytest tests/test_wrappers.py` để xác nhận bài test báo lỗi trước khi cài đặt**

Lệnh chạy: `pytest -q tests/test_wrappers.py`
Kết quả kỳ vọng: FAIL (`ImportError` hoặc `NotImplementedError: TODO: FrameSkip.step`)

- [ ] **Bước 3: Cài đặt `FrameSkip`, `GrayResize`, `ScalarizeReward`, `NoProgressTruncation`, `MarioRewardShaping` trong `mario_ppo/wrappers.py` và nối vào `mario_ppo/env.py`**

Trong `mario_ppo/wrappers.py`:
- `FrameSkip.step(self, action)`: lặp tối đa `self.skip` lần gọi `self.env.step(action)`, cộng dồn `total_reward` kiểu `np.float32`, lưu 2 khung hình gần nhất (`prev_obs`, `last_obs`), dừng sớm khi `terminated or truncated`, trả về `np.maximum(prev_obs, last_obs)` nếu chạy $\ge 2$ khung hoặc `last_obs` nếu chỉ chạy 1 khung.
- `GrayResize.__init__(self, env, size=84)`: gán `self.observation_space = gym.spaces.Box(0, 255, (size, size), dtype=np.uint8)`.
- `GrayResize.observation(self, obs)`: chuyển xám theo độ sáng bằng `cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)` và co giãn về `(self.size, self.size)` bằng `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
- `ScalarizeReward.__init__(self, env, weights)`: lấy số mục tiêu `n_obj = env.unwrapped.reward_space.shape[0]` nếu `getattr(env.unwrapped, "reward_space", None) is not None` else `5`; ném `ValueError` nếu `len(weights) != n_obj`; lưu `self.weights = np.asarray(weights, dtype=np.float32)`.
- `ScalarizeReward.step(self, action)`: lưu `vec = np.asarray(reward, dtype=np.float32)` vào `info["vector_reward"]` và trả về `float(np.dot(self.weights, vec))`.
- Bổ sung `NoProgressTruncation(env, max_steps=180)` và `MarioRewardShaping(env, flag_bonus=0.0, stuck_penalty=0.0)` ở cuối `mario_ppo/wrappers.py`, rồi bật có điều kiện bên trong khối `if not eval_mode:` của `make_env` trong `mario_ppo/env.py` (trước `RandomNoopStart`).

- [ ] **Bước 4: Chạy lại `pytest tests/test_wrappers.py` để kiểm tra toàn bộ bài test đều qua**

Lệnh chạy: `pytest -v tests/test_wrappers.py`
Kết quả kỳ vọng: PASS (9/9 bài kiểm tra đạt)

- [ ] **Bước 5: Lưu thay đổi vào Git (Commit Task 1)**

```bash
git add mario_ppo/wrappers.py mario_ppo/env.py tests/test_wrappers.py
git commit -m "feat(wrappers): implement FrameSkip, GrayResize, ScalarizeReward and training tricks"
```

---

### Task 2: Cài đặt Mạng `ActorCritic` Nature-CNN (Phân hệ 2 — Sơn & Tín)

**Tệp liên quan:**
- Chỉnh sửa: `mario_ppo/model.py:21-37`
- Kiểm thử: `tests/test_model.py`

**Giao diện đầu vào / đầu ra:**
- Sử dụng sẵn: hàm `_init(layer, gain=np.sqrt(2))` trong `mario_ppo/model.py`
- Cung cấp cho `train.py` & `evaluate.py`:
  - `ActorCritic(n_actions: int, in_channels: int = 4, hidden_dim: int = 512)`
  - `ActorCritic.forward(obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` trả về `(logits (B, n_actions) float32, value (B,) float32)`

- [ ] **Bước 1: Thêm bài kiểm thử `hidden_dim` tùy chỉnh vào `tests/test_model.py`**

Thêm vào cuối tệp `tests/test_model.py`:
```python
def test_custom_hidden_dim_and_uniform_init():
    model = ActorCritic(n_actions=7, in_channels=4, hidden_dim=256)
    logits, value = model(_obs(batch=8))
    assert logits.shape == (8, 7) and value.shape == (8,)
    probs = torch.softmax(logits, dim=-1)
    assert float((probs - 1 / 7).abs().max()) < 0.05
```

- [ ] **Bước 2: Chạy `pytest tests/test_model.py` để xác nhận bài test đang báo lỗi**

Lệnh chạy: `pytest -q tests/test_model.py`
Kết quả kỳ vọng: FAIL (`NotImplementedError: TODO: ActorCritic.__init__`)

- [ ] **Bước 3: Cài đặt `ActorCritic.__init__` và `ActorCritic.forward` trong `mario_ppo/model.py`**

Trong `mario_ppo/model.py`:
- Xây dựng `self.trunk = nn.Sequential(_init(nn.Conv2d(in_channels, 32, 8, stride=4)), nn.ReLU(), _init(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(), _init(nn.Conv2d(64, 64, 3, stride=1)), nn.ReLU(), nn.Flatten(), _init(nn.Linear(64 * 7 * 7, hidden_dim)), nn.ReLU())`.
- Đầu chính sách: `self.policy = _init(nn.Linear(hidden_dim, n_actions), gain=0.01)`.
- Đầu giá trị: `self.value = _init(nn.Linear(hidden_dim, 1), gain=1.0)`.
- Trong `forward(self, obs)`: chuẩn hóa `h = self.trunk(obs.to(dtype=torch.float32) / 255.0)` và trả về `self.policy(h), self.value(h).squeeze(-1)`.

- [ ] **Bước 4: Chạy `pytest tests/test_model.py` để xác nhận mọi bài test đều qua**

Lệnh chạy: `pytest -v tests/test_model.py`
Kết quả kỳ vọng: PASS (6/6 bài kiểm tra đạt)

- [ ] **Bước 5: Lưu thay đổi vào Git (Commit Task 2)**

```bash
git add mario_ppo/model.py tests/test_model.py
git commit -m "feat(model): implement shared Nature-CNN ActorCritic with orthogonal init"
```

---

### Task 3: Cài đặt GAE & Vòng cập nhật PPO (Phân hệ 3 — Quang & Tín)

**Tệp liên quan:**
- Chỉnh sửa: `mario_ppo/ppo.py:8-90`
- Kiểm thử: `tests/test_gae.py`, `tests/test_ppo.py`

**Giao diện đầu vào / đầu ra:**
- Cung cấp các hàm cho `train.py`:
  - `compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam) -> tuple[torch.Tensor, torch.Tensor]`
  - `clipped_policy_loss(ratio, advantages, clip) -> torch.Tensor`
  - `value_loss(values, returns, old_values=None, clip=None) -> torch.Tensor`
  - `approx_kl_and_clipfrac(log_ratio, clip) -> tuple[float, float]`
  - `ppo_loss(new_logp, entropy, values, batch, cfg) -> tuple[torch.Tensor, dict[str, float]]`
  - `ppo_update(model, optimizer, batch, cfg) -> dict[str, float | int]`

- [ ] **Bước 1: Thêm bài kiểm thử hỗn hợp cắt lượt và tiếp tục tại biên đoạn thu vào `tests/test_gae.py`**

Thêm vào cuối tệp `tests/test_gae.py`:
```python
def test_mixed_boundary_truncated_vs_continuing_across_envs():
    # Môi trường 0 bị cắt ở t=1 (dùng final_values[1, 0]=4.0); môi trường 1 chạy tiếp (dùng last_value[1]=10.0)
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[2.0, 2.0], [3.0, 3.0]])
    terminated = torch.zeros(2, 2)
    truncated = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    final_values = torch.tensor([[0.0, 0.0], [4.0, 999.0]])
    last_value = torch.tensor([123.0, 10.0])
    adv, ret = compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma=0.9, lam=0.8)
    # Env 0 tại t=1: 2 + 0.9 * 4 - 3 = 2.6; Env 1 tại t=1: 2 + 0.9 * 10 - 3 = 8.0
    assert adv[1, 0].item() == pytest.approx(2.6, abs=1e-5)
    assert adv[1, 1].item() == pytest.approx(8.0, abs=1e-5)
```

- [ ] **Bước 2: Chạy `pytest tests/test_gae.py tests/test_ppo.py` để xác nhận bài test báo lỗi**

Lệnh chạy: `pytest -q tests/test_gae.py tests/test_ppo.py`
Kết quả kỳ vọng: FAIL (`NotImplementedError: TODO: compute_gae`)

- [ ] **Bước 3: Cài đặt 6 hàm cốt lõi trong `mario_ppo/ppo.py`**

Thực hiện theo đúng công thức ở Mục 5.3 của tài liệu thiết kế:
- Trong `compute_gae`: duyệt ngược `for t in reversed(range(T)):`
  - `next_val = last_value if t == T - 1 else values[t + 1]`
  - `next_val = torch.where(truncated[t] > 0.5, final_values[t], next_val)`
  - `non_terminal = 1.0 - terminated[t]`
  - `continuing = non_terminal * (1.0 - truncated[t])`
  - `delta = rewards[t] + gamma * non_terminal * next_val - values[t]`
  - `gae = delta + gamma * lam * continuing * gae`
  - Gán `advantages[t] = gae`, cuối cùng trả về `(advantages, advantages + values)`.
- Trong `clipped_policy_loss`: tính `-torch.minimum(ratio * advantages, torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages).mean()`.
- Trong `value_loss`: tính `(values - returns).pow(2).mean()`; nếu `old_values is not None and clip is not None`, tính `v_clip = old_values + torch.clamp(values - old_values, -clip, clip)` và trả về `torch.maximum((values - returns).pow(2), (v_clip - returns).pow(2)).mean()`.
- Trong `approx_kl_and_clipfrac`: không tính gradient, trả về 2 giá trị `float`: `((ratio - 1.0) - log_ratio).mean().item()` và `((ratio - 1.0).abs() > clip).float().mean().item()`.
- Trong `ppo_loss`: nếu `cfg.norm_adv` là `True`, chuẩn hóa lợi thế bằng `(adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)`. Truyền `clip = cfg.clip_coef if getattr(cfg, "clip_vloss", False) else None` vào `value_loss`.
- Trong `ppo_update`: lặp `cfg.update_epochs` epoch, xáo trộn chỉ số bằng `torch.randperm(B, device=batch["obs"].device)`, chia thành `cfg.num_minibatches` mini-batch, cắt chuẩn gradient với `cfg.max_grad_norm`, và dừng sớm các epoch còn lại nếu `cfg.target_kl is not None and epoch_kl_mean > cfg.target_kl`.

- [ ] **Bước 4: Chạy `pytest tests/test_gae.py tests/test_ppo.py` để xác nhận tất cả bài test GAE & PPO đều qua**

Lệnh chạy: `pytest -v tests/test_gae.py tests/test_ppo.py`
Kết quả kỳ vọng: PASS (19/19 bài kiểm tra đạt)

- [ ] **Bước 5: Lưu thay đổi vào Git (Commit Task 3)**

```bash
git add mario_ppo/ppo.py tests/test_gae.py tests/test_ppo.py
git commit -m "feat(ppo): implement GAE with truncation bootstrap and clipped PPO update"
```

---

### Task 4: Hạ tầng Đa nền tảng, Học chuyển giao, Bộ cấu hình Colab/Kaggle & Vẽ biểu đồ Báo cáo (Phân hệ 4 — Tín làm riêng)

**Tệp liên quan:**
- Chỉnh sửa: `mario_ppo/train.py:27-34,88-96`
- Chỉnh sửa: `configs/default.yaml` (giữ nguyên mặc định, thêm khóa tùy chọn `init_ckpt: null`)
- Tạo mới: `configs/1-1.yaml`, `configs/1-4.yaml`, `configs/4-1.yaml`, `configs/7-1.yaml`, `configs/ablation_gamma099.yaml`, `configs/ablation_notricks.yaml`
- Tạo mới: `scripts/plot_report.py`
- Tạo mới: `notebooks/train_colab_kaggle.ipynb`
- Kiểm thử: `tests/test_pipeline.py`, `tests/test_plot_report.py`

**Giao diện đầu vào / đầu ra:**
- Sử dụng từ Task 1–3: `mario_ppo.env.make_env`, `mario_ppo.model.ActorCritic`, `mario_ppo.ppo.compute_gae`, `mario_ppo.ppo.ppo_update`
- Cung cấp:
  - `_make_vec_env` tương thích cả Windows (`"spawn"`) và Linux/Colab/Kaggle (`"forkserver"`)
  - Khóa `init_ckpt` trong `train(cfg)`: chỉ nạp `ck["model"]`, giữ `global_step = 0` và `Adam` mới tinh để học chuyển giao từ `1-1` sang `4-1` hoặc `7-1`
  - `scripts/plot_report.py`: các hàm `plot_learning_curves(progress_csv, out_path)`, `plot_ablation_comparison(csvs_a, csvs_b, label_a, label_b, out_path)`, `plot_failure_x_distribution(eval_jsons, level, out_path)` và giao diện dòng lệnh `main()`

- [ ] **Bước 1: Thêm bài kiểm thử học chuyển giao `init_ckpt` vào `tests/test_pipeline.py` và kiểm thử vẽ biểu đồ vào `tests/test_plot_report.py`**

Thêm vào cuối `tests/test_pipeline.py`:
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

Tạo tệp `tests/test_plot_report.py`:
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

- [ ] **Bước 2: Chạy `pytest tests/test_plot_report.py` để xác nhận lỗi trước khi tạo script**

Lệnh chạy: `pytest -q tests/test_plot_report.py`
Kết quả kỳ vọng: FAIL (`ModuleNotFoundError: No module named 'scripts'`)

- [ ] **Bước 3: Cập nhật `mario_ppo/train.py`, tạo bộ cấu hình `configs/*.yaml`, `scripts/plot_report.py` và `notebooks/train_colab_kaggle.ipynb`**

1. Trong `mario_ppo/train.py`:
   - Tại `_make_vec_env`: dùng `ctx = "spawn" if sys.platform == "win32" else "forkserver"` truyền vào `AsyncVectorEnv(..., context=ctx)`.
   - Sau dòng `global_step = 0`: nếu `cfg.get("init_ckpt") and not cfg.resume`, nạp `ck = torch.load(cfg.init_ckpt, map_location=device, weights_only=False)` và gọi `model.load_state_dict(ck["model"])`.
2. Trong `configs/default.yaml`: thêm `init_ckpt: null` ngay dưới `resume: null`.
3. Tạo 6 tệp cấu hình trong `configs/` tối ưu cho Colab/Kaggle (`num_envs: 8`, `num_steps: 512`):
   - `configs/1-1.yaml` (`level: "1-1"`, `total_steps: 4_000_000`, `gamma: 0.9`, `max_no_progress_steps: 180`, `flag_bonus: 5.0`)
   - `configs/1-4.yaml` (`level: "1-4"`, `total_steps: 6_000_000`, `gamma: 0.95`, `max_no_progress_steps: 220`, `flag_bonus: 8.0`)
   - `configs/4-1.yaml` (`level: "4-1"`, `total_steps: 6_000_000`, `gamma: 0.95`, `max_no_progress_steps: 180`, `flag_bonus: 6.0`)
   - `configs/7-1.yaml` (`level: "7-1"`, `total_steps: 8_000_000`, `gamma: 0.95`, `max_no_progress_steps: 200`, `flag_bonus: 8.0`)
   - `configs/ablation_gamma099.yaml` (giống `configs/1-1.yaml`, đổi `algo.gamma: 0.99`, `algo.total_steps: 3_000_000`)
   - `configs/ablation_notricks.yaml` (giống `configs/1-1.yaml`, tắt `max_no_progress_steps: 0`, `flag_bonus: 0.0`, `algo.total_steps: 3_000_000`)
4. Tạo `scripts/plot_report.py` sử dụng `matplotlib` (backend `Agg`) để vẽ 3 loại biểu đồ chuẩn báo cáo 300 DPI:
   - 6 đường cong học theo `global_step` (`ep_return`, `x_pos`, `flag_rate`, `entropy`, `approx_kl`, `clipfrac`).
   - Biểu đồ so sánh thí nghiệm đối chứng (trung bình $\pm$ độ lệch chuẩn qua 2 seed).
   - Biểu đồ phân bố tọa độ $x$ kết thúc lượt để phân tích điểm chết.
5. Tạo `notebooks/train_colab_kaggle.ipynb` với các cell sẵn sàng để 4 thành viên chạy song song trên Colab/Kaggle (tự động mount Google Drive, tự động `resume` checkpoint mới nhất nếu đứt phiên 12h, và tự động chạy `evaluate.py` xuất JSON + MP4).

- [ ] **Bước 4: Chạy toàn bộ bộ kiểm thử `pytest` trên toàn dự án**

Lệnh chạy: `pytest -v`
Kết quả kỳ vọng: PASS (100% bài kiểm tra trong `test_wrappers.py`, `test_model.py`, `test_gae.py`, `test_ppo.py`, `test_pipeline.py`, `test_plot_report.py` đều đạt)

- [ ] **Bước 5: Lưu thay đổi vào Git (Commit Task 4)**

```bash
git add mario_ppo/train.py configs/ scripts/plot_report.py notebooks/train_colab_kaggle.ipynb tests/test_pipeline.py tests/test_plot_report.py
git commit -m "feat(infra): add cross-platform vec env, transfer init_ckpt, stage configs, and report plotter"
```
