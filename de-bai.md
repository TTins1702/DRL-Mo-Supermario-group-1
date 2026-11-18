# Đồ án cuối kỳ: Dạy Mario vượt màn bằng PPO

Học phần Học tăng cường, học kỳ 1 năm học 2026–2027. Nhóm 4 sinh viên, thời gian 4 tuần.

## 1. Mục tiêu

Nhóm tự cài đặt các thành phần cốt lõi của thuật toán PPO (Proximal Policy Optimization) kiểu actor–critic, rồi huấn luyện một tác tử chơi Super Mario Bros từ điểm ảnh. Sau đồ án, nhóm phải:

1. biến môi trường trò chơi thành một bài toán học tăng cường dùng được: quan sát, hành động, phần thưởng, kết thúc và cắt ngắn lượt;
2. cài đặt đúng GAE, mục tiêu cắt của PPO, mất mát của critic, entropy và vòng cập nhật nhiều epoch;
3. huấn luyện, theo dõi và chẩn đoán một lần chạy dài (lợi tức, tỉ lệ chạm cờ, approxKL, tỉ lệ cắt, entropy);
4. đánh giá tác tử theo một giao thức cố định và giải thích bằng số liệu vì sao nó thành công hoặc thất bại.

## 2. Môi trường

Môi trường là `mo-supermario-v0` của MO-Gymnasium. Mỗi bước, môi trường trả một phần thưởng véc-tơ 5 thành phần `[x_pos, time, death, coin, enemy]`: tiến sang phải, thời gian trôi, chết, nhặt xu, hạ địch. Nhóm chọn trọng số để đổi véc-tơ này thành một số thực.

| Thành phần | Giá trị gốc | Ghi chú |
|---|---|---|
| Quan sát | ảnh RGB 240×256 | tiền xử lý thành 4 ảnh xám 84×84 xếp chồng |
| Hành động | `simple`: 7 tổ hợp nút (NOOP, sang phải, nhảy, chạy, ...) | có thể chọn `right_only`, `simple_down`, `complex` |
| Kết thúc lượt | Mario chết, chạm cờ hoặc hết giờ | thông tin `flag_get` cho biết có chạm cờ |
| Màn | `"w-s"`, ví dụ `"1-1"` | thế giới 1–8, màn 1–4 |

Kho mã khởi đầu đã sửa sẵn hai lỗi của môi trường gốc (phần phạt khi chết không bao giờ được trả; tọa độ x 16 bit bị tràn). Nhóm không cần xử lý lại.

## 3. Nhiệm vụ

### 3.1 Phần bắt buộc: cài đặt (mốc tuần 1)

Các hàm có `raise NotImplementedError("TODO: ...")` trong kho mã khởi đầu. Mỗi hàm có chuỗi tài liệu mô tả mục đích, đầu vào, đầu ra và yêu cầu.

| Tệp | Cần cài đặt | Bài kiểm tra |
|---|---|---|
| `mario_ppo/wrappers.py` | `FrameSkip`, `GrayResize`, `ScalarizeReward` | `tests/test_wrappers.py` |
| `mario_ppo/model.py` | `ActorCritic.__init__`, `ActorCritic.forward` | `tests/test_model.py` |
| `mario_ppo/ppo.py` | `compute_gae`, `clipped_policy_loss`, `value_loss`, `approx_kl_and_clipfrac`, `ppo_loss`, `ppo_update` | `tests/test_gae.py`, `tests/test_ppo.py` |
| (toàn bộ) | chạy được vòng huấn luyện và script đánh giá | `tests/test_pipeline.py` |

Phần đã có sẵn, không cần viết lại: vòng thu dữ liệu song song (`train.py`), chuỗi wrapper (`env.py`), script đánh giá (`evaluate.py`). Mốc tuần 1 đạt khi `pytest` báo mọi bài kiểm tra đều qua.

### 3.2 Phần bắt buộc: màn 1-1

Huấn luyện tác tử vượt màn 1-1. Ngân sách gợi ý: 2–5 triệu bước tác tử. Màn được coi là **vượt** khi tỉ lệ chạm cờ theo giao thức ở Mục 4 đạt ít nhất 0,8 (16/20 lượt).

### 3.3 Phần bắt buộc: một màn thưởng

Mỗi nhóm chọn **một** màn trong danh sách sau, ghi tên màn trong báo cáo, và huấn luyện tác tử vượt màn đó với ngân sách tối đa 10 triệu bước tác tử:

> 1-2, 1-4, 2-1, 3-1, 3-3, 4-1, 5-2, 6-1

Nhóm được khởi tạo từ checkpoint 1-1 của chính mình (học chuyển giao) hoặc huấn luyện từ đầu; báo cáo phải nêu rõ cách nào. Một màn ngoài danh sách (ví dụ 4-2, 7-1, 8-1) được tính thêm điểm thưởng ở Mục 5.

### 3.4 Được phép làm gì để Mario vượt màn

Nhóm được dùng mọi kỹ thuật dưới đây, miễn ghi lại trong báo cáo và giữ đúng các quy định ở Mục 3.5:

- đổi siêu tham số (γ, λ, ε, tốc độ học, số môi trường, độ dài đoạn thu, số epoch, hệ số entropy, ...);
- đổi trọng số phần thưởng, thêm wrapper định hình phần thưởng chỉ dùng khi huấn luyện, ví dụ thưởng khi chạm cờ, phạt khi đứng yên quá lâu, cắt lượt khi Mario không tiến;
- đổi kiến trúc mạng (sâu hơn, IMPALA-CNN, chuẩn hóa, ...) miễn giữ giao diện `forward`/`act`;
- đổi tập hành động trong số các tập có sẵn;
- học theo chương trình (curriculum), học chuyển giao từ checkpoint của chính nhóm.

### 3.5 Quy định

1. Không sửa `mario_ppo/evaluate.py` và phần `eval_mode` của `mario_ppo/env.py`. Mẹo huấn luyện đặt trong khối `if not eval_mode:` của `make_env`.
2. Đầu vào của mạng chỉ gồm các khung hình đã tiền xử lý; không đọc RAM của trò chơi, tọa độ hay `info` làm đầu vào.
3. Không dùng checkpoint hay trọng số huấn luyện sẵn từ bên ngoài nhóm.
4. Mọi checkpoint nộp phải tái tạo được từ cấu hình đã nộp (cho phép sai khác do ngẫu nhiên).

## 4. Giao thức đánh giá

Điểm phần 3.2 và 3.3 tính bằng script có sẵn, trên màn đã biết và seed công khai:

```bash
python -m mario_ppo.evaluate --ckpt <checkpoint.pt> --level 1-1 --episodes 20 --seed 2026 --out ket-qua-1-1.json --video 1-1.mp4
```

Script chạy 20 lượt. Lượt thứ $i$ bắt đầu sau $n_i$ bước NOOP, với $n_i$ ngẫu nhiên trong $\{0,\dots,30\}$ sinh từ seed 2026, nên mọi nhóm gặp cùng các điểm xuất phát. Chính sách lấy mẫu ngẫu nhiên như khi huấn luyện; không có mẹo huấn luyện, không chuẩn hóa phần thưởng. Kết quả là tỉ lệ chạm cờ `flag_rate` và vị trí x cuối mỗi lượt. Giảng viên chạy lại đúng lệnh này trên checkpoint nhóm nộp.

## 5. Chấm điểm

| Hạng mục | Điểm | Cách tính |
|---|---|---|
| A. Cài đặt | 30 | tỉ lệ bài kiểm tra công khai qua (wrappers 6, model 6, GAE 9, PPO 9) |
| B. Màn 1-1 | 25 | $25\cdot\min(1,\ f/0{,}8)$ với $f$ là `flag_rate` theo Mục 4 |
| C. Màn thưởng | 15 | $15\cdot\min(1,\ f/0{,}8)$ |
| D. Báo cáo | 25 | xem Mục 6 |
| E. Trình bày và demo | 5 | 10 phút trình bày, 5 phút hỏi đáp |
| Điểm thưởng | tối đa +10 | thêm một màn ngoài danh sách, chấm như hạng mục C với hệ số 10 |

## 6. Báo cáo (tối đa 8 trang, PDF)

1. **Cài đặt.** Mô tả ngắn từng thành phần đã viết, các lựa chọn và lỗi đã gặp. Bảng kết quả `pytest`.
2. **Thiết lập huấn luyện.** Bảng cấu hình cuối cùng của mỗi màn, phần cứng, số bước, thời gian chạy, tốc độ (bước mỗi giây).
3. **Đường cong học.** Theo số bước: lợi tức, vị trí x, tỉ lệ chạm cờ, entropy, approxKL, tỉ lệ cắt. Mỗi hình kèm một đoạn nhận xét.
4. **Thí nghiệm so sánh.** Ít nhất một thí nghiệm thay đổi một lựa chọn, giữ nguyên phần còn lại, ví dụ γ = 0,9 so với 0,99, có hay không chuẩn hóa lợi thế, có hay không NOOP ngẫu nhiên, một mẹo định hình phần thưởng. Nêu giả thuyết, kết quả, kết luận; nếu đủ thời gian, chạy 2 seed cho mỗi cấu hình.
5. **Phân tích thất bại.** Tác tử thường chết ở đâu (biểu đồ vị trí x cuối lượt), vì sao, nhóm đã thử gì.
6. **Phân công.** Đóng góp của từng thành viên.

## 7. Nộp bài

Một tệp nén `nhom-<số>.zip` gồm:

- toàn bộ mã nguồn, cấu hình đã dùng (`config.yaml` của mỗi lần chạy nộp), `progress.csv` và `evals.csv` của các lần chạy đó;
- checkpoint tốt nhất cho 1-1 và cho màn thưởng;
- các tệp JSON và video do `evaluate.py` sinh theo Mục 4;
- báo cáo PDF và tệp trình chiếu.

## 8. Tiến độ

| Tuần | Việc | Mốc |
|---|---|---|
| 1 | Cài đặt môi trường; viết wrapper, mạng, GAE, PPO; chạy thử ngắn | mọi bài kiểm tra qua (nộp mã) |
| 2 | Huấn luyện 1-1; đọc chẩn đoán; chọn màn thưởng | checkpoint 1-1 và kết quả đánh giá |
| 3 | Huấn luyện màn thưởng; thí nghiệm so sánh | |
| 4 | Hoàn thiện báo cáo, video, trình bày | nộp bài, trình bày |

## 9. Gợi ý về tài nguyên tính toán

Trình giả lập chạy trên CPU; mạng nhỏ nên GPU giúp ít. Với 8–16 lõi CPU, tốc độ thường khoảng 100–500 bước mỗi giây, tức 1 triệu bước mất từ 30 phút đến 3 giờ. Nên chia việc trong nhóm: mỗi máy chạy một cấu hình, cùng lúc. Lần chạy bị ngắt có thể chạy tiếp bằng `resume=<checkpoint>`. Mỗi checkpoint (mặc định mỗi 500 nghìn bước) kèm 5 lượt đánh giá và một video, giúp phát hiện sớm lần chạy hỏng.
