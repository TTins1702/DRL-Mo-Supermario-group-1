# Tài liệu Thiết kế (Design Spec): Hệ thống Multi-Objective Speedrun PPO (MO-PPO) cho `mo-supermario-v0`

- **Ngày tạo:** 2026-11-12
- **Nhóm thực hiện:** Group 1 (4 thành viên)
- **Môi trường mục tiêu:** `mo-supermario-v0` (`mo-gymnasium` & `gym-super-mario-bros`)
- **Nền tảng huấn luyện:** Kaggle & Google Colab (đồng bộ qua GitHub, Google Drive, và Weights & Biases)

---

## 1. Mục đích & Bối cảnh (Intent and the Why)

Dự án này là đồ án cuối kỳ môn **Học tăng cường (Reinforcement Learning)** của nhóm 4 sinh viên, giải quyết bài toán điều khiển nhân vật Mario trên môi trường đa mục tiêu `mo-supermario-v0` với hai cột mốc chấm điểm và xếp hạng (Ranking):

1. **Mục tiêu 8 điểm (Cốt lõi):** Xây dựng và huấn luyện thành công tác tử (Agent) bằng Deep Reinforcement Learning vượt qua các chướng ngại vật và chạm cờ (reaches the flag) tại màn 1-1 một cách ổn định.
2. **Mục tiêu 2 điểm (Đua Top / Ranking giữa các nhóm):** Đáp ứng yêu cầu mở rộng của giảng viên (sẽ công bố sau, bao gồm các kịch bản: thời gian về đích ngắn nhất, tổng điểm số/xu/diệt quái cao nhất, hoặc vượt qua nhiều màn liên tiếp).
3. **Yêu cầu Vấn đáp cá nhân:** Giảng viên sẽ hỏi vấn đáp ngẫu nhiên từng thành viên về bài toán, công thức toán học và từng dòng code. Do đó, toàn bộ nhóm thống nhất **tập trung phát triển duy nhất một mô hình mạnh nhất (MO-PPO) tự cài đặt bằng PyTorch**, chia thành 4 phân hệ rõ ràng để cả 4 thành viên đều nắm vững 100% lý thuyết và code.

---

## 2. Mục tiêu, Phi mục tiêu & Điều cần tránh (Goals, Non-Goals, Anti-Goals)

### 2.1. Mục tiêu (Goals)
- **Tự cài đặt MO-PPO bằng PyTorch từ đầu:** Minh bạch hóa toàn bộ luồng tính toán từ mạng Residual CNN, ước lượng lợi thế 5 chiều (5D GAE), chuẩn hóa lợi thế từng mục tiêu (Per-Objective Advantage Normalization) đến hàm mất mát cắt ngưỡng (Clipped Surrogate Loss).
- **Khai thác trọn vẹn vector phần thưởng 5 chiều:** Sử dụng **Multi-Head Vector Critic** $\mathbf{V}_\phi(s) \in \mathbb{R}^5$ để học riêng giá trị kỳ vọng của từng mục tiêu (`x_pos`, `time`, `death`, `coin`, `enemy`), giúp Mario vừa về đích nhanh vừa ăn xu/diệt quái mà không bị nhiễu gradient.
- **Tối ưu cho Kaggle & Google Colab:** Bộ nhớ đệm `RolloutBuffer` lưu ảnh ở định dạng `uint8` tiết kiệm 4 lần RAM; tự động phát hiện môi trường Colab/Kaggle để lưu và khôi phục (Resume) toàn bộ trạng thái huấn luyện khi hết phiên 12 giờ.
- **Theo dõi thực nghiệm nhóm trực quan:** Ghi log kép ra file CSV/TensorBoard cục bộ và đồng bộ trực tuyến lên **Weights & Biases (WandB)** kèm video `.mp4` và thống kê độc lập cả 5 chiều phần thưởng.

### 2.2. Phi mục tiêu (Non-Goals)
- Không sử dụng các thư viện đóng gói sẵn thuật toán như `stable-baselines3`, `cleanrl` hay `morl-baselines` làmagent chính (vì ẩn công thức toán học, bất lợi khi vấn đáp).
- Không triển khai dàn trải 4 thuật toán rời rạc (như DQN, A2C, REINFORCE riêng lẻ) làm phân tán tài nguyên GPU và khiến thành viên không hiểu hết code của nhau.
- Không huấn luyện trực tiếp trên không gian hành động thô `Discrete(256)` hay ảnh RGB `(240, 256, 3)` chưa qua tiền xử lý.

### 2.3. Điều cần tránh (Anti-Goals)
- **Tránh kiến trúc quá trừu tượng (Over-engineering):** Không dùng chuỗi kế thừa đa tầng phức tạp khiến sinh viên khó chỉ ra công thức toán nằm ở dòng code nào khi vấn đáp.
- **Tránh gộp thô (Naive Scalarization) trước Critic:** Không cộng gộp trực tiếp phần thưởng đồng xu ($+100$) và bước đi ($+1$) vào một Critic đơn lẻ $V(s) \in \mathbb{R}^1$ vì chênh lệch thang đo hàng trăm lần sẽ làm hỏng gradient học đường đi.
- **Tránh xung đột thư viện trên Colab/Kaggle:** Không để `pip` tự nâng cấp `numpy >= 2.0` hoặc `gymnasium` không tương thích với `nes-py` và `gym-super-mario-bros`.

---

## 3. Ràng buộc kỹ thuật (Constraints)

1. **Đặc tả môi trường `mo-supermario-v0`:**
   - **Observation Space gốc:** `Box(0, 255, (240, 256, 3), uint8)`.
   - **Action Space gốc:** `Discrete(256)`.
   - **Reward Space gốc:** `Box([-inf, -inf, -25.0, 0.0, 0.0], [inf, 0.0, 0.0, 100.0, inf], (5,), float32)` ứng với:
     - `r[0]`: Quãng đường tiến theo trục $x$ ($\Delta x$).
     - `r[1]`: Phạt thời gian trôi qua giữa 2 bước ($\Delta t \le 0$).
     - `r[2]`: Phạt tử vong ($-25$ nếu Mario chết, $0$ nếu sống).
     - `r[3]`: Điểm thưởng ăn xu ($+100$ nếu ăn được xu ở bước đó, ngược lại $0$).
     - `r[4]`: Điểm thưởng tiêu diệt quái.
2. **Tương thích thư viện trên Kaggle / Colab:**
   - `nes-py==8.2.1` và `gym-super-mario-bros==7.4.0` sử dụng các kiểu dữ liệu `np.bool8` / `uint8` cũ bị loại bỏ trên `numpy >= 2.0`. Cần khóa cứng `numpy==1.26.4` và bọc lớp tương thích (shim) cho `mo-gymnasium`.
3. **Giới hạn phần cứng:**
   - RAM hệ thống trên Colab miễn phí là ~12.7 GB, VRAM GPU T4/P100 là 15–16 GB; số môi trường chạy song song (`num_envs`) cần đặt mặc định là `8` để cân bằng tốc độ CPU emulation (NES) và GPU.

---

## 4. Các phương án đã cân nhắc & Lý do lựa chọn (Approaches)

| Phương án | Mô tả | Đánh đổi (Trade-offs) | Quyết định |
| :--- | :--- | :--- | :--- |
| **1. Chia 4 mô hình khác nhau cho 4 thành viên** (DQN, DDQN, A2C, PPO) | Mỗi thành viên tự code và train 1 thuật toán riêng biệt trên nền môi trường chung. | Dễ chia việc ban đầu nhưng chia nhỏ thời gian GPU của nhóm thành 4 phần; mỗi thành viên chỉ rành thuật toán của mình nên rủi ro cao khi giảng viên vấn đáp chéo; chất lượng model cuối cùng không đạt đỉnh để đua Ranking. | **Loại bỏ** |
| **2. Cả nhóm làm 1 mô hình Double Dueling DQN** | Tập trung vào thuật toán Value-based với Replay Buffer. | Công thức Bellman dễ hiểu, nhưng Replay Buffer lưu ảnh ngốn RAM cực lớn trên Colab/Kaggle, tốc độ thu thập mẫu chậm và khó tổng quát hóa khi chạy tốc độ cao (Speedrun). | **Loại bỏ** |
| **3. Cả nhóm tập trung 1 mô hình MO-PPO (5-Head Vector Critic + Speedrun Wrapper + Residual CNN)** | Chia dự án thành 4 phân hệ kỹ thuật cho 4 người; dùng PPO đa môi trường kết hợp Critic 5 đầu ra và chuẩn hóa Advantage từng chiều. | Nhẹ RAM, tốc độ hội tụ nhanh nhờ chạy 8 môi trường song song; giải quyết triệt để bài toán lệch thang đo giữa 5 chiều reward; dễ dàng đổi bộ trọng số $\mathbf{w}$ cho 2 điểm mở rộng; tạo lợi thế cạnh tranh lớn khi xếp hạng cuối kỳ. | **Lựa chọn làm Baseline chính** |

---

## 5. Cấu trúc thư mục chung & Phân công 4 thành viên (Directory Structure & Team Roles)

```text
DRL-Mo-Supermario-group-1/
├── configs/
│   ├── base_8pt_clear.yaml         # Cấu hình chuẩn tối ưu qua màn 1-1 ổn định (8 điểm)
│   ├── bonus_speedrun.yaml         # Cấu hình ưu tiên tốc độ tối đa (đón đầu 2 điểm phụ)
│   ├── bonus_highscore.yaml        # Cấu hình ưu tiên ăn xu + diệt quái + qua màn (đón đầu 2 điểm phụ)
│   └── bonus_multistage.yaml       # Cấu hình huấn luyện/fine-tune qua nhiều màn (1-1 -> 1-4)
├── docs/
│   ├── RL-supermario.md            # Đề bài gốc
│   ├── viva_math_code_mapping.md   # Tài liệu đối chiếu Công thức Toán <-> Dòng Code cho cả 4 người ôn vấn đáp
│   └── superpowers/
│       └── specs/
│           └── 2026-11-12-mo-ppo-supermario-design.md
├── notebooks/
│   ├── colab_runner.ipynb          # Notebook mỏng chạy trên Google Colab (Mount Drive + Clone + Train/Eval)
│   └── kaggle_runner.ipynb         # Notebook mỏng chạy trên Kaggle (Setup + Train/Eval + Export)
├── scripts/
│   └── setup_env.sh                # Script cài đặt thư viện khóa phiên bản chuẩn cho Colab/Kaggle
├── src/
│   ├── __init__.py
│   ├── env/                        # === PHÂN HỆ 1 (THÀNH VIÊN 1): MÔI TRƯỜNG & TIỀN XỬ LÝ ===
│   │   ├── __init__.py
│   │   ├── action_wrappers.py      # Rút gọn Discrete(256) -> 7 hành động Speedrun
│   │   ├── obs_wrappers.py         # SkipFrame(4), GrayScale, Resize(84x84), FrameStack(4)
│   │   ├── reward_wrappers.py      # Scaling 5 chiều reward, Anti-Stuck & ghi log 5 chiều gốc (raw)
│   │   └── make_env.py             # Hàm khởi tạo môi trường đơn & VectorEnv (8 envs song song)
│   ├── model/                      # === PHÂN HỆ 2 (THÀNH VIÊN 2): MẠNG CNN & BỘ ĐỆM ROLLOUT ===
│   │   ├── __init__.py
│   │   ├── impala_cnn.py           # Mạng Residual CNN + Orthogonal Init + Actor Head + 5D Critic Head
│   │   └── rollout_buffer.py       # Bộ đệm lưu (s, a, log_prob, r_5d, done, v_5d) & tính 5D GAE
│   ├── agent/                      # === PHÂN HỆ 3 (THÀNH VIÊN 3): THUẬT TOÁN MO-PPO ===
│   │   ├── __init__.py
│   │   └── mo_ppo_agent.py         # Chuẩn hóa Advantage 5 chiều, ghép trọng số w, PPO Clip Loss, cập nhật mạng
│   └── utils/                      # === PHÂN HỆ 4 (THÀNH VIÊN 4): HẠ TẦNG, CHECKPOINT & ĐÁNH GIÁ ===
│       ├── __init__.py
│       ├── config_loader.py        # Đọc/ghi file cấu hình YAML và ghi đè từ CLI
│       ├── checkpoint.py           # Tự động lưu & resume checkpoint trên Colab (Drive) / Kaggle / Local
│       ├── logger.py               # Ghi log đồng thời ra CSV, TensorBoard và Weights & Biases (WandB)
│       └── video_recorder.py       # Xuất video .mp4 quá trình Mario qua màn kèm thông số
├── tests/                          # Kiểm thử tự động từng phân hệ (chạy bằng pytest)
│   ├── test_env_wrappers.py
│   ├── test_model_and_buffer.py
│   ├── test_mo_ppo_agent.py
│   └── test_checkpoint_and_logger.py
├── train.py                        # Điểm vào huấn luyện chính (CLI hỗ trợ --config, --resume, --wandb)
├── evaluate.py                     # Điểm vào đánh giá model (chạy N episodes, tính Win Rate, xuất bảng & video)
└── requirements.txt                # Danh sách thư viện khóa phiên bản
```

---

## 6. Thiết kế Kỹ thuật Chi tiết cho từng Phân hệ (Technical Specification)

### 6.1. Phân hệ 1 (`src/env/`) — Tiền xử lý & Đa mục tiêu
1. **Không gian hành động Speedrun (`SpeedrunActionWrapper`):**
   - Thay thế `Discrete(256)` bằng `Discrete(7)` tối ưu cho tốc độ và vượt chướng ngại vật:
     - `0`: `['NOOP']`
     - `1`: `['right']` (Đi bộ sang phải)
     - `2`: `['right', 'A']` (Nhảy thường sang phải)
     - `3`: `['right', 'B']` (Chạy nước rút sang phải)
     - `4`: `['right', 'A', 'B']` (Chạy nước rút + nhảy xa/cao — hành động chủ lực)
     - `5`: `['A']` (Nhảy thẳng đứng để phá gạch/ăn xu hoặc thoát kẹt sát ống cống)
     - `6`: `['left']` (Lùi nhẹ sang trái để lấy đà khi bị kẹt sát chướng ngại vật)
2. **Xử lý khung hình (`SkipFrame` + `GrayScale` + `Resize` + `FrameStack`):**
   - `SkipFrame(skip=4)`: Lặp lại hành động trong 4 frame NES liên tiếp; **cộng dồn vector phần thưởng 5 chiều** $\mathbf{r}_{\text{step}} = \sum_{k=1}^{4} \mathbf{r}_{t+k} \in \mathbb{R}^5$. Nếu Mario chạm cờ hoặc chết ở frame thứ $k < 4$, dừng vòng lặp ngay.
   - Chuyển ảnh RGB `(240, 256, 3)` sang ảnh xám `(240, 256)`, thu nhỏ bằng phép nội suy song tuyến tính (bilinear/area) về `(84, 84)`, và ghép chồng 4 bước thời gian thành tensor `(4, 84, 84)` kiểu `uint8` (giữ nguyên `0..255` trong bộ đệm để tiết kiệm RAM, chỉ chia `255.0` khi đưa lên GPU trong hàm `forward`).
3. **Chuẩn hóa thang đo 5 chiều & Chống kẹt (`MultiObjectiveRewardWrapper`):**
   - **Lưu giữ thống kê gốc (Raw Metrics):** Trước khi biến đổi, cộng dồn vector gốc $\mathbf{r}_{\text{raw}} \in \mathbb{R}^5$ của toàn bộ episode vào `info["episode_raw_rewards"]` (cùng với `info["flag_get"]`, `info["x_pos"]`, `info["time"]`, `info["coins"]`, `info["score"]`) để phục vụ đánh giá công bằng.
   - **Co giãn thang đo (Component Scaling):** Nhân từng phần tử với hệ số co giãn cấu hình sẵn $\mathbf{s} = [0.05, 0.05, 0.08, 0.01, 0.02]$ để đưa cả 5 thành phần (`x_pos`, `time`, `death`, `coin`, `enemy`) về vùng giá trị xấp xỉ $[-2.0, +2.0]$, giúp mạng Critic hội tụ nhanh và ổn định.
   - **Milestone & Anti-Stuck Shaping (tùy chọn bật/tắt trong config):** Trên chiều 0 (`x_pos`), thưởng thêm $+\beta_{\text{milestone}}$ khi Mario vượt qua tọa độ $x_{\max}$ lớn nhất từ đầu episode, và phạt nhẹ $-\beta_{\text{stuck}}$ nếu sau $K_{\text{stuck}} = 60$ bước (tương đương 240 frames) mà $x_{\max}$ không tăng (tránh đứng im trước ống cống cao).

### 6.2. Phân hệ 2 (`src/model/`) — Mạng IMPALA Residual CNN & 5D Rollout Buffer
1. **Kiến trúc mạng `ActorCriticResNet` (`src/model/impala_cnn.py`):**
   - **Đầu vào:** Tensor trạng thái $s_t \in \mathbb{R}^{B \times 4 \times 84 \times 84}$ (ép kiểu `float32 / 255.0`).
   - **Khối trích xuất đặc trưng (IMPALA-Lite Backbone):**
     - Gồm 3 tầng `ConvSequence` với số kênh lần lượt là `[32, 64, 64]`.
     - Mỗi tầng `ConvSequence` gồm: `Conv2d(kernel=3, stride=1, padding=1)` $\rightarrow$ `MaxPool2d(kernel=3, stride=2, padding=1)` $\rightarrow$ **1 Residual Block** (`ReLU -> Conv2d(3x3) -> ReLU -> Conv2d(3x3) + Skip Connection`).
     - Sau 3 lần giảm chiều (`84 -> 42 -> 21 -> 11`), trải phẳng (Flatten) tensor `(64 * 11 * 11 = 7744)` qua lớp `ReLU` $\rightarrow$ `Linear(7744 -> 512)` $\rightarrow$ `ReLU` tạo ra vector đặc trưng chung $\mathbf{h}_t \in \mathbb{R}^{512}$.
   - **Đầu ra Actor (Policy Head):** `Linear(512 -> 7)` khởi tạo trực giao (`orthogonal_`, `gain=0.01`), tạo phân phối `Categorical(logits=...)`.
   - **Đầu ra Vector Critic (5-Head Value Function):** `Linear(512 -> 5)` khởi tạo trực giao (`orthogonal_`, `gain=1.0`), dự đoán vector giá trị kỳ vọng $\mathbf{V}_\phi(s_t) = [V^{(0)}, V^{(1)}, V^{(2)}, V^{(3)}, V^{(4)}] \in \mathbb{R}^5$.
2. **Bộ đệm `MultiObjectiveRolloutBuffer` (`src/model/rollout_buffer.py`):**
   - Thu thập $T = 512$ bước trên $N = 8$ môi trường song song (tổng cộng $B = T \times N = 4096$ mẫu mỗi lần cập nhật).
   - **Công thức tính 5-Dimensional GAE (`compute_returns_and_advantages`):**
     Với từng bước $t = T-1 \dots 0$ và mặt nạ không kết thúc $m_t = 1.0 - \text{done}_t$, tính đồng thời trên cả 5 chiều bằng phép toán vector hóa:
     $$\boldsymbol{\delta}_t = \mathbf{r}_t + \gamma m_t \mathbf{V}_\phi(s_{t+1}) - \mathbf{V}_\phi(s_t) \quad \in \mathbb{R}^{N \times 5}$$
     $$\hat{\mathbf{A}}_t = \boldsymbol{\delta}_t + \gamma \lambda m_t \hat{\mathbf{A}}_{t+1} \quad \in \mathbb{R}^{N \times 5}$$
     $$\mathbf{R}_t = \hat{\mathbf{A}}_t + \mathbf{V}_\phi(s_t) \quad \in \mathbb{R}^{N \times 5}$$
     Trong đó $\gamma = 0.99$ (hoặc mảng $\boldsymbol{\gamma} \in \mathbb{R}^5$ nếu muốn hệ số chiết khấu riêng) và $\lambda = 0.95$.

### 6.3. Phân hệ 3 (`src/agent/`) — Thuật toán MO-PPO & Hàm Mất Mát
1. **Chuẩn hóa Lợi thế từng chiều & Tổng hợp Trọng số (Per-Objective Advantage Normalization):**
   - Trên toàn bộ batch rollout $4096$ mẫu, mỗi chiều mục tiêu $i \in \{0, 1, 2, 3, 4\}$ được chuẩn hóa độc lập về trung bình $0$ và độ lệch chuẩn $1$:
     $$\tilde{A}_t^{(i)} = \frac{\hat{A}_t^{(i)} - \text{mean}(\hat{A}^{(i)})}{\text{std}(\hat{A}^{(i)}) + 10^{-8}}$$
   - Nhân tích vô hướng với vector trọng số ưu tiên $\mathbf{w} = [w_0, w_1, w_2, w_3, w_4]$ (đọc từ file YAML cấu hình):
     $$\hat{A}_t^{\text{w}} = \sum_{i=0}^{4} w_i \tilde{A}_t^{(i)}$$
2. **Cập nhật qua $K_{\text{epochs}} = 4$ vòng lặp với Mini-batch size $M = 256$ (hoặc $512$):**
   - **Tỷ số xác suất:** $r_t(\theta) = \exp\left(\log \pi_\theta(a_t|s_t) - \log \pi_{\theta_{\text{old}}}(a_t|s_t)\right)$.
   - **Actor Loss (Clipped Surrogate Loss):**
     $$\mathcal{L}^{\text{CLIP}}(\theta) = -\frac{1}{M} \sum_{t=1}^{M} \min\left( r_t(\theta) \hat{A}_t^{\text{w}}, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t^{\text{w}} \right), \quad \epsilon = 0.2$$
   - **Vector Critic Loss (Multi-Head Huber / Smooth L1 Loss):**
     $$\mathcal{L}^{\text{VF}}(\phi) = \frac{1}{5M} \sum_{t=1}^{M} \sum_{i=0}^{4} \text{SmoothL1Loss}\left( V_\phi^{(i)}(s_t), R_t^{(i)} \right)$$
   - **Entropy Regularization:** $\mathcal{L}^{\text{ENT}}(\theta) = \frac{1}{M} \sum_{t=1}^{M} \mathcal{H}(\pi_\theta(\cdot|s_t))$.
   - **Tổng hàm Loss:**
     $$\mathcal{L}^{\text{TOTAL}} = \mathcal{L}^{\text{CLIP}}(\theta) + c_{\text{vf}} \mathcal{L}^{\text{VF}}(\phi) - c_{\text{ent}} \mathcal{L}^{\text{ENT}}(\theta)$$
     với $c_{\text{vf}} = 0.5$, $c_{\text{ent}} = 0.01$, cắt chuẩn gradient `max_grad_norm = 0.5`, và giảm tuyến tính tốc độ học `lr` từ $2.5 \times 10^{-4}$ về $1.0 \times 10^{-5}$.

### 6.4. Phân hệ 4 (`src/utils/`, `train.py`, `evaluate.py`) — Vận hành Kaggle/Colab & Chấm điểm
1. **Quản lý Checkpoint thông minh (`CheckpointManager`):**
   - Tự động phát hiện môi trường chạy:
     - Nếu có `/content/drive/MyDrive`: lưu vào `/content/drive/MyDrive/DRL_Mario_Checkpoints/<exp_name>/`.
     - Nếu có `/kaggle/working`: lưu vào `/kaggle/working/checkpoints/<exp_name>/`.
     - Ngược lại: lưu vào `./checkpoints/<exp_name>/`.
   - Lưu 3 loại file:
     - `latest.pt`: Ghi đè mỗi 25 vòng cập nhật (~100k frames) chứa `model_state_dict`, `optimizer_state_dict`, `global_step`, `update_idx`, `best_metrics`, `config`.
     - `best_winrate.pt`: Lưu khi tỷ lệ qua màn 1-1 (`flag_get`) trung bình 20 episodes gần nhất đạt đỉnh mới.
     - `best_speedrun.pt`: Lưu khi thời gian hoàn thành màn 1-1 trung bình (trong các lần chạm cờ) ngắn nhất.
2. **Chiến lược chuyển đổi từ 8 điểm sang 2 điểm mở rộng:**
   - Giai đoạn 1 (8 điểm): Huấn luyện với `configs/base_8pt_clear.yaml` ($\mathbf{w} = [0.65, 0.10, 0.20, 0.02, 0.03]$) cho đến khi `win_rate >= 90%`.
   - Giai đoạn 2 (2 điểm): Khi giảng viên công bố đề 2 điểm:
     - Chạy lệnh `python train.py --config configs/bonus_<type>.yaml --resume checkpoints/best_winrate.pt --reset-lr` để fine-tune nhanh từ mạng đã biết nhảy qua mọi chướng ngại vật và Critic đã học sẵn 5 chiều giá trị!

---

## 7. Những phần giao quyền quyết định cho người lập trình (Left to the Builder)

Các chi tiết nhỏ sau không làm thay đổi kiến trúc và người lập trình từng phân hệ có thể chủ động quyết định:
- Định dạng bảng in ra màn hình console (dùng `tqdm` hoặc `rich` / `print` chuẩn).
- Các tham số phụ trong hàm xuất video `.mp4` (`imageio-ffmpeg` hoặc `opencv-python`, FPS = 30).
- Cách đặt tên các trường phụ trên dashboard Weights & Biases miễn là giữ đủ 5 chiều reward gốc, `win_rate`, `avg_x_pos`, `avg_time_left`, và `avg_score`.

