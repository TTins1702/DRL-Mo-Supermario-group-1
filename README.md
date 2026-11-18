# mario-ppo: kho mã khởi đầu của đồ án

Đề bài đầy đủ: `de-bai.md`.

## Cài đặt

Cần Python 3.10–3.12 và trình biên dịch C++ (để cài `nes-py`).

```bash
python -m venv .venv && source .venv/bin/activate      # hoặc: uv venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # bỏ --index-url nếu dùng GPU
pip install -e .
pytest -q          # ban đầu hầu hết bài kiểm tra báo lỗi NotImplementedError
```

Trên Colab: `!pip install -q -e .` trong thư mục dự án.

## Cấu trúc

| Tệp | Vai trò | Cần viết? |
|---|---|---|
| `mario_ppo/wrappers.py` | FrameSkip, GrayResize, ScalarizeReward (+ wrapper riêng của nhóm) | **TODO** |
| `mario_ppo/model.py` | Mạng actor-critic | **TODO** |
| `mario_ppo/ppo.py` | GAE, mất mát PPO, vòng cập nhật | **TODO** |
| `mario_ppo/env.py` | Chuỗi wrapper, chỗ thêm mẹo huấn luyện | có sẵn |
| `mario_ppo/train.py` | Vòng huấn luyện, nhật ký, checkpoint, video | có sẵn |
| `mario_ppo/evaluate.py` | Script đánh giá dùng để chấm | có sẵn, không sửa |
| `configs/default.yaml` | Cấu hình mặc định | sửa tùy ý |
| `tests/` | Bài kiểm tra công khai | |

Gợi ý thứ tự: `test_wrappers` → `test_model` → `test_gae` → `test_ppo` → `test_pipeline`.

```bash
pytest -q tests/test_gae.py          # một nhóm bài kiểm tra
pytest -q -k frameskip               # theo tên
```

## Huấn luyện

```bash
python -m mario_ppo.train                                        # 1-1, cấu hình mặc định
python -m mario_ppo.train level=1-4 algo.total_steps=8000000 seed=2
python -m mario_ppo.train algo.num_envs=8 run_dir=runs/thu-nghiem # máy ít lõi
python -m mario_ppo.train resume=runs/1-1/<run>/checkpoints/step_002000000.pt
```

Mỗi lần chạy ghi vào `runs/<màn>/<thời điểm>/`:

| Tệp | Nội dung |
|---|---|
| `config.yaml` | cấu hình đã dùng |
| `progress.csv` | mỗi lần cập nhật: sps, lr, policy_loss, value_loss, entropy, approx_kl, clipfrac, ep_return, x_pos, flag_rate (100 lượt gần nhất) |
| `evals.csv` | mỗi checkpoint: flag_rate của 5 lượt đánh giá |
| `checkpoints/step_*.pt` | trọng số, optimizer, số bước, cấu hình |
| `videos/step_*.mp4` | lượt tốt nhất ở mỗi checkpoint |

## Đánh giá (giống hệt cách chấm)

```bash
python -m mario_ppo.evaluate --ckpt runs/1-1/<run>/checkpoints/step_003000000.pt --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4
```

## Thêm mẹo huấn luyện

1. Viết wrapper mới ở cuối `mario_ppo/wrappers.py`.
2. Bật nó trong khối `if not eval_mode:` của `make_env` (`mario_ppo/env.py`), đọc tham số từ `tricks`.
3. Thêm khóa tương ứng vào mục `env:` của cấu hình, ví dụ `env.max_no_progress_steps=200`.

Mẹo đặt trong khối này không chạy khi đánh giá, nên không ảnh hưởng đến kết quả chấm.
