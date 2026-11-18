# Tài liệu Thiết kế (Design Spec): Hệ thống PPO Chinh phục Super Mario Bros (`mario_ppo`)

- **Ngày tạo:** 2026-11-18
- **Nhóm thực hiện:** Group 1 (4 thành viên: Việt, Sơn, Quang, Tín)
- **Kho mã nền tảng:** `mario_ppo` (`mo-supermario-v0`, `mo-gymnasium==1.3.2`, `gym-super-mario-bros==7.4.0`)
- **Nền tảng huấn luyện:** Google Colab & Kaggle Notebooks (GPU T4/P100 + 2–4 vCPU, đồng bộ qua GitHub & Google Drive)
- **Mục tiêu điểm số:** **110 / 100 điểm** (100 điểm chuẩn + 10 điểm thưởng ngoài danh sách) & đạt mức *Exceeds Mastery (4/4)* ở cả 4 chuẩn đầu ra CLO1–CLO4 khi vấn đáp.

---

## 1. Mục đích & Bối cảnh (Intent and the Why)

Đồ án cuối kỳ học phần **Học tăng cường (Học kỳ 1, 2026–2027)** yêu cầu nhóm 4 sinh viên tự cài đặt các thành phần cốt lõi của thuật toán **PPO (Proximal Policy Optimization)** kiểu Actor–Critic trên kho mã khởi đầu `mario_ppo/`, huấn luyện tác tử chơi Super Mario Bros từ điểm ảnh thô, và đánh giá bằng giao thức cố định công khai (`--episodes 20 --seed 2026`).

Để đạt **điểm tuyệt đối (110/100)** và đảm bảo cả 4 thành viên trả lời xuất sắc mọi câu hỏi vấn đáp độc lập của giảng viên, bản thiết kế này giải quyết trọn vẹn 6 hạng mục chấm điểm trong [de-bai.md](../../../de-bai.md) và [projectdetail.pdf](../../projectdetail.pdf):

| Hạng mục | Điểm tối đa | Điều kiện đạt điểm tuyệt đối |
|---|---|---|
| **A. Cài đặt** | **30 đ** | Vượt qua 100% bài kiểm tra `pytest` công khai: `wrappers` (6đ), `model` (6đ), `GAE` (9đ), `PPO` (9đ) và `pipeline` đầu-cuối. |
| **B. Màn 1-1** | **25 đ** | Tỉ lệ chạm cờ `flag_rate` $\ge 0.8$ ($\ge 16/20$ lượt với `--seed 2026`). Mục tiêu của nhóm: $\ge 0.90$ ($18/20$ lượt). |
| **C. Màn thưởng (trong danh sách)** | **15 đ** | Chọn màn **`1-4`** (kèm nhánh dự phòng/đối chứng **`4-1`**) trong danh sách `{1-2, 1-4, 2-1, 3-1, 3-3, 4-1, 5-2, 6-1}`, ngân sách $\le 10$ triệu bước, đạt `flag_rate` $\ge 0.8$. |
| **Điểm thưởng (ngoài danh sách)** | **+10 đ** | Chọn màn **`7-1`** (ngoài danh sách mục 3.3), ngân sách $\le 10$ triệu bước, đạt `flag_rate` $\ge 0.8$. |
| **D. Báo cáo PDF ($\le 8$ trang)** | **25 đ** | Đủ 6 mục bắt buộc: mô tả cài đặt & bảng `pytest`, bảng thiết lập & tốc độ chạy, 6 đường cong học kèm nhận xét, thí nghiệm so sánh (Ablation Study chạy 2 seed), biểu đồ phân tích vị trí chết $x$, và bảng phân công. |
| **E. Trình bày & Vấn đáp** | **5 đ** | 10 phút trình bày + 5 phút hỏi đáp; cả 4 thành viên đều nắm vững toán học, mã nguồn và chẩn đoán thực nghiệm (đạt 4/4 điểm ở mỗi tiêu chí CLO1–CLO4). |

---

## 2. Mục tiêu, Phi mục tiêu & Điều cần tránh (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
1. **Đạt 100% `pytest` ngay trong Tuần 1:** Cài đặt chính xác, ngắn gọn, dễ đọc cho toàn bộ các hàm `TODO` trong `mario_ppo/wrappers.py`, `mario_ppo/model.py`, và `mario_ppo/ppo.py`.
2. **Tối ưu hiệu năng cho Google Colab & Kaggle:** Cấu hình song song phù hợp với giới hạn 2–4 lõi CPU của Colab/Kaggle (`num_envs: 8`, `num_steps: 512`, giữ nguyên cỡ batch $8 \times 512 = 4096$), tận dụng GPU (`device: auto`) cho mạng CNN và tự động lưu/khôi phục (`resume`) qua Google Drive hoặc `/kaggle/working`.
3. **Bổ sung mẹo huấn luyện hợp lệ (`if not eval_mode:`):** Cài đặt các wrapper cắt lượt khi không tiến triển (`NoProgressTruncation`) và định hình phần thưởng (`MarioRewardShaping`) chỉ kích hoạt lúc huấn luyện, giúp Mario vượt qua các điểm nghẽn (ống cống cao, hố rộng, thanh lửa lâu đài `1-4`, súng đại bác `7-1`) nhanh gấp 2–3 lần.
4. **Chuẩn hóa dữ liệu cho Báo cáo 8 trang:** Tự động hóa việc trích xuất bảng số liệu và vẽ biểu đồ chất lượng cao từ `progress.csv`, `evals.csv` và `ket-qua-*.json`.

### 2.2. Phi mục tiêu (Non-Goals)
- Không viết lại vòng lặp thu thập dữ liệu `mario_ppo/train.py` hay cấu trúc thư mục ngoài gói `mario_ppo/`.
- Không sử dụng thư viện RL bên ngoài (`stable-baselines3`, `cleanrl`, `tianshou`).

### 2.3. Điều cần tránh (Anti-Goals — Quy định bắt buộc Mục 3.5)
- **Tuyệt đối không sửa `mario_ppo/evaluate.py` và phần `eval_mode=True` trong `mario_ppo/env.py`:** Mọi wrapper định hình phần thưởng hoặc cắt lượt sớm phải nằm hoàn toàn bên trong khối `if not eval_mode:` của `make_env`.
- **Tuyệt đối không gian lận đầu vào của mạng:** Mạng `ActorCritic` chỉ nhận tensor khung hình đã tiền xử lý `(B, C, 84, 84)`; không đọc RAM, tọa độ `x_pos` hay `info` đưa vào mạng.
- **Không để lệch không gian quan sát giữa lúc học và lúc chấm:** `FrameSkip`, `GrayResize`, `FrameStackObservation` và tập hành động `env.actions` phải đồng nhất giữa huấn luyện và đánh giá.
- **Không dùng trọng số từ bên ngoài nhóm:** Mọi checkpoint nộp đều được sinh ra từ mã nguồn và file `config.yaml` của chính nhóm (chỉ dùng học chuyển giao từ checkpoint `1-1` của chính nhóm).

---

## 3. Ràng buộc Kỹ thuật (Constraints)

1. **Ràng buộc từ bộ kiểm tra tự động `tests/`:**
   - `tests/test_wrappers.py`:
     - `FrameSkip`: lặp hành động tối đa `skip` lần, dừng ngay khi `terminated` hoặc `truncated`; trả tổng phần thưởng véc-tơ kiểu `np.float32`; quan sát trả về là `np.maximum` theo từng điểm ảnh của **2 khung hình cuối** (hoặc khung duy nhất nếu lượt kết thúc ngay ở bước đầu).
     - `GrayResize`: `observation_space` phải là `Box(0, 255, (size, size), np.uint8)`; phải chuyển ảnh xám theo công thức độ sáng (luminance `cv2.COLOR_RGB2GRAY`, $Y \approx 0.299R + 0.587G + 0.114B$), không được lấy trung bình cộng 3 kênh.
     - `ScalarizeReward`: kiểm tra `len(weights)` khớp với số chiều phần thưởng (`env.unwrapped.reward_space.shape[0]` nếu tồn tại, mặc định là `5`), nếu sai phải ném `ValueError`; trả phần thưởng kiểu `float` chuẩn của Python và lưu véc-tơ thưởng gốc (`np.float32`) vào `info["vector_reward"]`.
   - `tests/test_model.py`:
     - `ActorCritic(n_actions, in_channels=4)` phải chấp nhận `n_actions` và `in_channels` bất kỳ (ví dụ `n_actions=12, in_channels=2`); đầu ra `logits` có shape `(B, n_actions)` và `value` có shape `(B,)` kiểu `torch.float32`.
     - Chính sách khởi tạo phải gần phân phối đều: `max(|softmax(logits) - 1/n_actions|) < 0.05` (đạt được bằng khởi tạo trực giao `gain=0.01` cho lớp tuyến tính `self.policy`).
     - Mọi tham số trong `model.named_parameters()` phải nhận được gradient hữu hạn khác `None` trong một lượt `backward()`.
   - `tests/test_gae.py`:
     - `compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam)` nhận các tensor `(T, N)` và `last_value` `(N,)`.
     - Nếu bước cuối `t = T - 1` có `terminated == 0` và `truncated == 0`, coi như bị cắt ngắn với giá trị bootstrap là `last_value`.
     - Tại mọi bước $t$:
       - Giá trị kế tiếp $V_t^{\text{next}}$: lấy `final_values[t]` nếu `truncated[t] == 1`, ngược lại lấy `values[t + 1]` (riêng tại bước cuối `t = T - 1` nếu không bị cắt thì dùng `last_value`).
       - Sai số TD (ký hiệu $d_t = \text{terminated}[t]$ và $c_t = \text{truncated}[t]$):
         $$\delta_t = r_t + \gamma (1 - d_t) V_t^{\text{next}} - V_t$$
       - Truy hồi lùi GAE:
         $$A_t = \delta_t + \gamma \lambda (1 - d_t)(1 - c_t) A_ {t+1}$$
       - Mục tiêu của Critic:
         $$R_t = A_t + V_t$$
   - `tests/test_ppo.py`:
     - `clipped_policy_loss`:
       $$L^{\text{CLIP}}(\theta) = -\operatorname{mean}\!\left(\min\!\left(r_t(\theta) A_t,\ \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\right)\right)$$
     - `value_loss`: tính $\operatorname{mean}((V_t - R_t)^2)$ (không có hệ số $1/2$); nếu `old_values` và `clip` khác `None` thì tính $V_t^{\text{clip}} = V_t^{\text{old}} + \operatorname{clamp}(V_t - V_t^{\text{old}}, -\epsilon, +\epsilon)$ và lấy:
       $$L^{\text{VF}}(\theta) = \operatorname{mean}\!\left(\max\!\left((V_t - R_t)^2,\ (V_t^{\text{clip}} - R_t)^2\right)\right)$$
     - `approx_kl_and_clipfrac`: `approx_kl = mean((exp(log_ratio) - 1) - log_ratio)`, `clipfrac = mean((|exp(log_ratio) - 1| > clip).float())`, trả về 2 số thực `float`.
     - `ppo_loss`: khi `cfg.norm_adv == True`, chuẩn hóa lợi thế trên mini-batch bằng **độ lệch chuẩn tổng thể** `adv.std(unbiased=False) + 1e-8` (giữ nguyên `returns` không đổi).
     - `ppo_update`: lặp `cfg.update_epochs` epoch, xáo trộn mẫu, chia thành `cfg.num_minibatches` mini-batch, cắt chuẩn gradient `cfg.max_grad_norm`, kiểm tra dừng sớm sau mỗi epoch nếu `cfg.target_kl is not None` và `epoch_mean_approx_kl > cfg.target_kl`, trả về `dict` trung bình các thống kê kèm `"minibatch_updates"`.
2. **Ràng buộc đa nền tảng (Windows local vs Linux Colab/Kaggle):**
   - Trong `mario_ppo/train.py`, hàm `_make_vec_env` gọi `AsyncVectorEnv(..., context="forkserver")`. Trên hệ điều hành Windows (`sys.platform == "win32"`), Python chỉ hỗ trợ `"spawn"`. Do đó cần chọn động `"spawn" if sys.platform == "win32" else "forkserver"` để bộ kiểm tra `tests/test_pipeline.py` chạy qua trên cả máy Windows cá nhân lẫn Linux Colab/Kaggle.

---

## 4. Các Phương án Đã Cân nhắc & Lý do Lựa chọn (Approaches)

| Phương án | Nội dung chính | Đánh đổi (Trade-offs) | Quyết định |
|---|---|---|---|
| **Phương án A (Đã chọn)** | **Nature-CNN chuẩn + Mẹo huấn luyện trong `if not eval_mode:` (`NoProgressTruncation`, `MarioRewardShaping`) + Cấu hình song song tối ưu cho Colab/Kaggle (`num_envs=8, num_steps=512`)** | **Ưu điểm:** Tốc độ thu mẫu trên Colab/Kaggle cao nhất (~250–400 bước/giây); cắt sớm các lượt Mario bị kẹt giúp tiết kiệm 40% số bước huấn luyện; dễ dàng vượt mốc `flag_rate >= 0.8` ở cả `1-1`, `1-4` và `7-1`; code cực kỳ rõ ràng khi vấn đáp.<br>**Nhược điểm:** Cần quản lý các file cấu hình YAML riêng cho từng màn và thí nghiệm. | **CHỌN** |
| **Phương án B** | **Thay bằng mạng sâu IMPALA-ResNet15 + Huấn luyện thuần (không dùng wrapper cắt lượt hay định hình phần thưởng)** | **Ưu điểm:** Mạng sâu hơn.<br>**Nhược điểm:** Tốc độ huấn luyện chậm trên 2–4 vCPU của Colab; không cắt lượt khi kẹt trước ống cống làm lãng phí hàng triệu bước và dễ hết phiên 12h trước khi hội tụ ở màn `7-1`. | Loại |
| **Phương án C** | **Cài đặt tối thiểu theo nguyên bản `default.yaml`, chạy 1 seed duy nhất và không thêm wrapper phụ** | **Ưu điểm:** Ít phải cấu hình.<br>**Nhược điểm:** `num_envs=16` gây nghẽn cổ chai CPU trên Colab Free; thiếu dữ liệu 2 seed và phân tích sâu cho báo cáo 25 điểm. | Loại |

---

## 5. Thiết kế Kỹ thuật Chi tiết (Detailed Technical Design)

### 5.1. Phân hệ 1: Chuỗi Wrapper & Mẹo Huấn luyện (`mario_ppo/wrappers.py`, `mario_ppo/env.py`) — **Phụ trách: Việt & Tín**

1. **Ba Wrapper bắt buộc (chấm điểm tự động 6/30đ):**
   - **`FrameSkip(gym.Wrapper)`:**
     - Khởi tạo `self.skip = skip`.
     - Trong `step(action)`: lặp tối đa `self.skip` lần gọi `self.env.step(action)`.
     - Cộng dồn phần thưởng vào biến `total_reward` có kiểu `np.float32` (khởi tạo bằng `np.zeros_like(reward, dtype=np.float32)` ở bước đầu).
     - Lưu tối đa 2 khung hình gần nhất (`prev_obs`, `last_obs`). Nếu số bước thực hiện $\ge 2$, trả về `np.maximum(prev_obs, last_obs)`; nếu chỉ chạy 1 bước (do kết thúc sớm), trả về `last_obs`.
     - Dừng vòng lặp ngay khi `terminated or truncated`.
   - **`GrayResize(gym.ObservationWrapper)`:**
     - Trong `__init__(self, env, size=84)`: lưu `self.size = size` và đặt `self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size, size), dtype=np.uint8)`.
     - Trong `observation(self, obs)`: chuyển sang ảnh xám bằng `gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)` và co giãn bằng `cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_AREA)`.
   - **`ScalarizeReward(gym.Wrapper)`:**
     - Trong `__init__(self, env, weights)`: lấy số mục tiêu `n_obj = env.unwrapped.reward_space.shape[0] if hasattr(env.unwrapped, "reward_space") and env.unwrapped.reward_space is not None else 5`. Chuyển `w = np.asarray(weights, dtype=np.float32)`. Nếu `w.shape != (n_obj,)` thì `raise ValueError(f"Expected {n_obj} weights, got {len(weights)}")`. Lưu `self.weights = w`.
     - Trong `step(action)`: gọi `obs, reward, terminated, truncated, info = self.env.step(action)`. Lưu `vec = np.asarray(reward, dtype=np.float32)`, gán `info = dict(info, vector_reward=vec)`, và trả phần thưởng vô hướng `float(np.dot(self.weights, vec))`.

2. **Hai Wrapper mẹo huấn luyện bổ sung (chỉ bật khi `not eval_mode`):**
   - **`NoProgressTruncation(gym.Wrapper)`:**
     - Mục đích: Khi Mario đứng kẹt trước ống cống cao, tường gạch hoặc thanh lửa lâu đài mà không tiến thêm theo trục $x$ trong `max_no_progress_steps` bước tác tử liên tiếp, đánh dấu `truncated = True` để kết thúc lượt sớm và bootstrap giá trị trạng thái cuối qua `compute_gae`.
     - Hoạt động:
       - Trong `reset(**kwargs)`: đặt `self._max_x = -1`, `self._stalled = 0`.
       - Trong `step(action)`: đọc `x = int(info.get("x_pos", 0))`. Nếu `x > self._max_x`: cập nhật `self._max_x = x` và `self._stalled = 0`; ngược lại tăng `self._stalled += 1`. Nếu `not terminated and self._stalled >= self.max_steps`: đặt `truncated = True`.
   - **`MarioRewardShaping(gym.Wrapper)`:**
     - Mục đích: Đặt trước `ScalarizeReward` để củng cố tín hiệu chạm cờ cuối màn và giảm hành vi đứng im câu giờ.
     - Hoạt động:
       - Khi `info.get("flag_get", False)` là `True`: cộng thêm `self.flag_bonus` (mặc định `5.0`) vào thành phần `reward[0]` (`x_pos`).
       - Khi `reward[0] <= 0` và chưa kết thúc lượt: cộng `self.stuck_penalty` (mặc định `0.0` hoặc `-0.02` tùy cấu hình) vào `reward[0]`.
   - **Kết nối vào `mario_ppo/env.py` (trong khối `if not eval_mode:`):**
     ```python
     if not eval_mode:
         if tricks.get("max_no_progress_steps"):
             env = NoProgressTruncation(env, int(tricks["max_no_progress_steps"]))
         if tricks.get("flag_bonus") or tricks.get("stuck_penalty"):
             env = MarioRewardShaping(
                 env,
                 flag_bonus=float(tricks.get("flag_bonus", 0.0)),
                 stuck_penalty=float(tricks.get("stuck_penalty", 0.0)),
             )
         if noop_max:
             env = RandomNoopStart(env, noop_max)
     ```
     *Lưu ý quan trọng:* Khi không truyền các khóa này (như trong `configs/default.yaml` hoặc khi chạy `pytest`), chuỗi wrapper hoạt động giống hệt mặc định ban đầu.

---

### 5.2. Phân hệ 2: Kiến trúc Mạng `ActorCritic` (`mario_ppo/model.py`) — **Phụ trách: Sơn & Tín**

1. **Cấu trúc mạng CNN chia sẻ thân (Shared Nature-CNN Backbone):**
   - Đầu vào: Tensor `obs` có shape `(B, in_channels, 84, 84)` kiểu `uint8` (hoặc `float32`).
   - Tiền xử lý trong `forward(obs)`: `x = obs.to(dtype=torch.float32) / 255.0`.
   - Thân trích xuất đặc trưng `self.trunk` (khởi tạo mọi lớp bằng `_init(layer, gain=np.sqrt(2))`):
     - `Conv2d(in_channels, 32, kernel_size=8, stride=4)` $\rightarrow$ `(B, 32, 20, 20)` + `ReLU()`
     - `Conv2d(32, 64, kernel_size=4, stride=2)` $\rightarrow$ `(B, 64, 9, 9)` + `ReLU()`
     - `Conv2d(64, 64, kernel_size=3, stride=1)` $\rightarrow$ `(B, 64, 7, 7)` + `ReLU()`
     - `Flatten()` $\rightarrow$ `(B, 3136)`
     - `Linear(3136, hidden_dim)` (mặc định `hidden_dim=512`) + `ReLU()`
   - Hai nhánh đầu ra tách biệt:
     - **Actor (`self.policy`):** `_init(nn.Linear(hidden_dim, n_actions), gain=0.01)` $\rightarrow$ trả về `logits` shape `(B, n_actions)`. Giá trị `gain=0.01` đảm bảo xác suất ban đầu của mọi hành động xấp xỉ `1 / n_actions` với sai số $< 0.01 < 0.05$.
     - **Critic (`self.value`):** `_init(nn.Linear(hidden_dim, 1), gain=1.0)` $\rightarrow$ `squeeze(-1)` trả về `value` shape `(B,)`.

---

### 5.3. Phân hệ 3: Toán học GAE & Vòng cập nhật PPO (`mario_ppo/ppo.py`) — **Phụ trách: Quang & Tín**

1. **`compute_gae(rewards, values, terminated, truncated, final_values, last_value, gamma, lam)`:**
   - Kích thước: `T, N = rewards.shape`. Khởi tạo `advantages = torch.zeros_like(rewards)` và `gae = torch.zeros(N, dtype=rewards.dtype, device=rewards.device)`.
   - Duyệt ngược $t = T - 1, T - 2, \dots, 0$:
     - Nếu $t == T - 1$:
       - Với những môi trường bị cắt ngay tại $t = T - 1$ (`truncated[t] == 1`), giá trị kế tiếp là `final_values[t]`; ngược lại dùng `last_value`.
       - Sau bước $T - 1$ không có $A_T$ trong đoạn thu nên `gae` khởi đầu bằng $0$.
     - Nếu $t < T - 1$:
       - Giá trị kế tiếp: `next_val = torch.where(truncated[t] > 0.5, final_values[t], values[t + 1])`.
     - Tính hệ số không kết thúc thật: `non_terminal = 1.0 - terminated[t]`.
     - Tính hệ số tiếp tục cùng lượt sang bước $t + 1$: `continuing = non_terminal * (1.0 - truncated[t])`.
     - Sai số TD: `delta = rewards[t] + gamma * non_terminal * next_val - values[t]`.
     - Truy hồi GAE: `gae = delta + gamma * lam * continuing * gae`.
     - Lưu `advantages[t] = gae`.
   - Tính `returns = advantages + values` và trả về `(advantages, returns)`.

2. **`clipped_policy_loss(ratio, advantages, clip)`:**
   - `unclipped = ratio * advantages`
   - `clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantages`
   - Trả về `-torch.minimum(unclipped, clipped).mean()`.

3. **`value_loss(values, returns, old_values=None, clip=None)`:**
   - `v_loss_unclipped = (values - returns).pow(2)`
   - Nếu `old_values is not None and clip is not None`:
     - `v_clipped = old_values + torch.clamp(values - old_values, -clip, clip)`
     - `v_loss_clipped = (v_clipped - returns).pow(2)`
     - Trả về `torch.maximum(v_loss_unclipped, v_loss_clipped).mean()`.
   - Ngược lại trả về `v_loss_unclipped.mean()`.

4. **`approx_kl_and_clipfrac(log_ratio, clip)`:**
   - Trong `torch.no_grad()`:
     - `ratio = torch.exp(log_ratio)`
     - `approx_kl = ((ratio - 1.0) - log_ratio).mean().item()`
     - `clipfrac = ((ratio - 1.0).abs() > clip).float().mean().item()`
   - Trả về `(float(approx_kl), float(clipfrac))`.

5. **`ppo_loss(new_logp, entropy, values, batch, cfg)`:**
   - Lấy `adv = batch["advantages"]`. Nếu `cfg.norm_adv`:
     - `adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)`.
   - `log_ratio = new_logp - batch["old_logp"]`, `ratio = torch.exp(log_ratio)`.
   - `p_loss = clipped_policy_loss(ratio, adv, cfg.clip_coef)`.
   - `v_clip = cfg.clip_coef if getattr(cfg, "clip_vloss", False) else None`.
   - `v_loss = value_loss(values, batch["returns"], old_values=batch.get("old_values"), clip=v_clip)`.
   - `ent_mean = entropy.mean()`.
   - `loss = p_loss + cfg.vf_coef * v_loss - cfg.ent_coef * ent_mean`.
   - `approx_kl, clipfrac = approx_kl_and_clipfrac(log_ratio, cfg.clip_coef)`.
   - Trả về `(loss, {"policy_loss": float(p_loss.item()), "value_loss": float(v_loss.item()), "entropy": float(ent_mean.item()), "approx_kl": approx_kl, "clipfrac": clipfrac})`.

6. **`ppo_update(model, optimizer, batch, cfg)`:**
   - Số mẫu `B = batch["obs"].shape[0]`, kích thước mini-batch `mb_size = B // cfg.num_minibatches`.
   - Khởi tạo bộ tích lũy tổng các thống kê và bộ đếm `minibatch_updates = 0`.
   - Lặp `epoch` trong `range(cfg.update_epochs)`:
     - Sinh hoán vị ngẫu nhiên `perm = torch.randperm(B, device=batch["obs"].device)`.
     - Theo dõi `epoch_kls = []`.
     - Chia `perm` thành `cfg.num_minibatches` đoạn chỉ số `idx`:
       - Tạo `mb = {k: v[idx] for k, v in batch.items()}`.
       - Gọi `logits, values = model(mb["obs"])`.
       - Tạo `dist = torch.distributions.Categorical(logits=logits)`.
       - Tính `new_logp = dist.log_prob(mb["actions"])`, `entropy = dist.entropy()`.
       - Gọi `loss, stats = ppo_loss(new_logp, entropy, values, mb, cfg)`.
       - Thực hiện `optimizer.zero_grad()`, `loss.backward()`, `torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)`, `optimizer.step()`.
       - Cộng dồn `stats`, thêm `stats["approx_kl"]` vào `epoch_kls`, tăng `minibatch_updates += 1`.
     - Nếu `cfg.target_kl is not None and (sum(epoch_kls) / len(epoch_kls)) > cfg.target_kl`: `break`.
   - Trả về `dict` trung bình của 5 khóa thống kê (chia cho `minibatch_updates`) kèm `"minibatch_updates": minibatch_updates`.

---

### 5.4. Phân hệ 4: Vận hành Colab/Kaggle, Đánh giá & Tự động hóa Báo cáo (`configs/`, `scripts/`, `notebooks/`) — **Phụ trách riêng: Tín**

1. **Tương thích Đa nền tảng trong `mario_ppo/train.py`:**
   - Tại hàm `_make_vec_env` (dòng 33), cập nhật:
     ```python
     ctx = "spawn" if sys.platform == "win32" else "forkserver"
     return gym.vector.AsyncVectorEnv(fns, context=ctx, autoreset_mode=mode)
     ```
2. **Bộ cấu hình YAML tối ưu cho Colab/Kaggle (`configs/`):**
   - Giữ nguyên `configs/default.yaml` để mọi bài test mặc định luôn nhất quán.
   - Tạo các cấu hình chuyên biệt:
     - **`configs/1-1.yaml` (Màn bắt buộc 1-1 — 25đ):**
       - `level: "1-1"`, `algo.num_envs: 8`, `algo.num_steps: 512`, `algo.total_steps: 4_000_000`, `algo.gamma: 0.90`, `env.actions: simple`, `env.max_no_progress_steps: 180`, `env.flag_bonus: 5.0`.
     - **`configs/1-4.yaml` (Màn thưởng trong danh sách 1-4 — 15đ):**
       - `level: "1-4"`, `algo.num_envs: 8`, `algo.num_steps: 512`, `algo.total_steps: 6_000_000`, `algo.gamma: 0.95` (tăng tầm nhìn xa để nhảy qua hồ dung nham và vượt Bowser ở cuối màn), `env.actions: simple`, `env.max_no_progress_steps: 220` (đủ thời gian chờ thanh lửa xoay qua), `env.flag_bonus: 8.0`.
     - **`configs/4-1.yaml` (Màn thưởng dự phòng/đối chứng trong danh sách 4-1):**
       - `level: "4-1"`, hỗ trợ nạp trọng số khởi tạo từ checkpoint `1-1` tốt nhất (Transfer Learning) hoặc huấn luyện từ đầu 6 triệu bước.
     - **`configs/7-1.yaml` (Màn thưởng ngoài danh sách 7-1 — +10đ):**
       - `level: "7-1"`, `algo.num_envs: 8`, `algo.num_steps: 512`, `algo.total_steps: 8_000_000`, `algo.gamma: 0.95`, `env.actions: simple`, `env.max_no_progress_steps: 200`, `env.flag_bonus: 8.0`.
   - **Tùy chọn Khởi tạo Trọng số Học chuyển giao (`init_weights`):**
     - Trong `mario_ppo/train.py`, ngoài `cfg.resume` (khôi phục cả `global_step` và `optimizer` để chạy tiếp cùng 1 màn), hỗ trợ thêm khóa tùy chọn `cfg.get("init_ckpt")`: chỉ nạp `model.load_state_dict(ck["model"])` với `global_step = 0` và `optimizer` mới tinh khi chuyển giao từ màn `1-1` sang `4-1` hoặc `7-1`.
3. **Script Vẽ biểu đồ & Phân tích lỗi cho Báo cáo (`scripts/plot_report.py`):**
   - Đọc `progress.csv` và `evals.csv` của các lần chạy để xuất hình vector PDF/PNG độ phân giải 300 DPI:
     1. **6 đường cong học (Mục 6.3):** `ep_return`, `x_pos`, `flag_rate`, `entropy`, `approx_kl`, `clipfrac` theo `global_step` (làm mượt EMA kèm dải mờ dữ liệu gốc).
     2. **Biểu đồ Thí nghiệm so sánh 2 seed (Mục 6.4):** Đường trung bình $\pm$ độ lệch chuẩn giữa 2 cấu hình đối chứng trên 2 seed (`seed=1`, `seed=2`).
     3. **Biểu đồ Phân tích thất bại (Mục 6.5):** Histogram / Strip-plot phân bố tọa độ `x_pos` cuối mỗi lượt từ file `ket-qua-*.json` và `evals.csv`, chú thích rõ các tọa độ chướng ngại vật nơi Mario hay chết nhất (ví dụ hố nhảy và ống cống cao ở $x \approx 900\text{--}1800$ màn `1-1`, hồ lửa/Bowser ở $x \approx 1400\text{--}2100$ màn `1-4`, súng đại bác/Hammer Bros ở màn `7-1`).
4. **Notebook chạy tự động trên Colab/Kaggle (`notebooks/train_colab_kaggle.ipynb`):**
   - Tự động mount Google Drive (nếu trên Colab) hoặc trỏ vào `/kaggle/working` (nếu trên Kaggle), cài đặt gói `pip install -e .`, kiểm tra `pytest -q`, tự động tìm checkpoint mới nhất trong thư mục `run_dir` để gắn `resume=...` khi chạy lại sau khi hết phiên 12h, và tự động chạy lệnh `python -m mario_ppo.evaluate --seed 2026 --episodes 20` xuất JSON + MP4 ngay khi hoàn tất.

---

## 6. Kế hoạch Thực nghiệm & Phân công Nhóm 4 Thành viên (Việt, Sơn, Quang, Tín)

Để đạt tối đa điểm **Làm việc nhóm (CLO4)** và đảm bảo tiến độ vững chắc nhất:
- **Tín** đồng phụ trách cài đặt, tích hợp và kiểm thử **cả 3 phân hệ cốt lõi (Phân hệ 1, 2, 3)** cùng với **Việt, Sơn, Quang**, đồng thời **độc lập thực hiện riêng toàn bộ Phân hệ 4** (hạ tầng huấn luyện đa nền tảng, cấu hình YAML, tự động hóa Colab/Kaggle, vẽ biểu đồ báo cáo và chinh phục màn điểm thưởng `7-1`).
- Cả 4 thành viên đều vận hành song song các nhánh huấn luyện trên Colab/Kaggle:

| Thành viên | Phân hệ Code & Kiểm thử (Tuần 1) | Nhánh Huấn luyện trên Colab/Kaggle (Tuần 2–3) | Phần Báo cáo & Vấn đáp phụ trách chính (Tuần 4) |
|---|---|---|---|
| **1. Việt** *(cùng Tín)* | **Phân hệ 1:** `mario_ppo/wrappers.py` (`FrameSkip`, `GrayResize`, `ScalarizeReward`, `NoProgressTruncation`, `MarioRewardShaping`) & `mario_ppo/env.py` + `tests/test_wrappers.py` | **Màn 1-1 chính thức** (`seed=1`, `seed=2` với cấu hình `configs/1-1.yaml`) $\rightarrow$ Chọn ra checkpoint `1-1` đạt `flag_rate` cao nhất ($\ge 0.90$). | **Mục 1 & Mục 2 Báo cáo:** Tiền xử lý ảnh, cơ chế FrameSkip/Max-pool, ScalarizeReward 5 chiều, và thiết lập môi trường. |
| **2. Sơn** *(cùng Tín)* | **Phân hệ 2:** `mario_ppo/model.py` (`ActorCritic.__init__`, `ActorCritic.forward`, khởi tạo trực giao) + `tests/test_model.py` | **Màn thưởng trong danh sách (`1-4` & `4-1`):** Chạy `configs/1-4.yaml` (từ đầu) và `configs/4-1.yaml` (transfer từ checkpoint `1-1` của Việt) $\rightarrow$ Đảm bảo `flag_rate` $\ge 0.80$ (15/15đ). | **Mục 1 (Model) & Mục 5 Báo cáo:** Kiến trúc mạng Actor-Critic, ý nghĩa khởi tạo trực giao `gain=0.01`, và Phân tích thất bại theo tọa độ $x$ ở các màn. |
| **3. Quang** *(cùng Tín)* | **Phân hệ 3:** `mario_ppo/ppo.py` (`compute_gae`, `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update`) + `tests/test_gae.py`, `tests/test_ppo.py` | **Thí nghiệm So sánh (Ablation Study — Mục 6.4):** Trên màn `1-1`, chạy đối chứng **2 seed (`seed=1, 2`)** khi thay đổi hệ số chiết khấu $\gamma = 0.90$ so với $\gamma = 0.99$ (và đối chứng có/không có `NoProgressTruncation`). | **Mục 3 & Mục 4 Báo cáo:** Giải thích toán học GAE (xử lý `terminated` vs `truncated`), mục tiêu cắt PPO, chẩn đoán 6 đường cong học và kết luận Thí nghiệm so sánh. |
| **4. Tín** *(Toàn diện 1–3 & làm riêng Phân hệ 4)* | **Đồng phụ trách Phân hệ 1, 2, 3** (cài đặt, rà soát code & đảm bảo pass 100% `pytest` toàn cục) + **Làm riêng Phân hệ 4:** `train.py`, `configs/*.yaml`, `scripts/plot_report.py`, `notebooks/train_colab_kaggle.ipynb` + `tests/test_pipeline.py` | **Màn điểm thưởng ngoài danh sách (`7-1` — +10đ) & Hỗ trợ huấn luyện toàn cục:** Chạy `configs/7-1.yaml` (cả từ đầu và transfer từ checkpoint `1-1`), xuất toàn bộ JSON + Video đánh giá chuẩn `--seed 2026`. | **Nắm toàn bộ 4 phân hệ khi vấn đáp; phụ trách Mục 2, Mục 6 & Tổng hợp PDF + Slide:** Bảng tổng hợp cấu hình/tốc độ SPS, biểu đồ báo cáo 8 trang, đóng gói `nhom-1.zip` và điều phối demo. |

---

## 7. Những phần để Người cài đặt Tự quyết định (What's Left to the Builder)

1. Cách đặt tên các biến cục bộ bên trong thân hàm của `wrappers.py`, `model.py`, `ppo.py` miễn tuân thủ đúng chữ ký hàm, docstring và vượt qua 100% `pytest`.
2. Bố cục chi tiết, bảng màu và kích thước font chữ trong script vẽ biểu đồ `scripts/plot_report.py`.
3. Tinh chỉnh nhẹ các hệ số trong file cấu hình YAML (`max_no_progress_steps` trong khoảng `160..240`, `flag_bonus` trong khoảng `3.0..10.0`) dựa trên quan sát thực tế từ video checkpoint mỗi 500 nghìn bước.
