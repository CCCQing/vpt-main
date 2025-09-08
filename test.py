
import torch, platform
print("python:", platform.python_version())
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda.is_available:", torch.cuda.is_available())
try:
    print("cudnn:", torch.backends.cudnn.version())
except Exception as e:
    print("cudnn error:", e)
