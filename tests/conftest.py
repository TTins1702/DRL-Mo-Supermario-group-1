import os
from pathlib import Path

# Chạy pytest từ bất kỳ đâu: đường dẫn tương đối (configs/...) tính từ gốc dự án.
os.chdir(Path(__file__).resolve().parents[1])
