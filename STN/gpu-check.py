# checking if my gpu works in this venv pytorch versions

import torch

print("torch version:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

device = "cuda" if torch.cuda.is_available() else "cpu"
print("using device:", device)

x = torch.randn(2, 3).to(device)
print("tensor device:", x.device)

if device == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))