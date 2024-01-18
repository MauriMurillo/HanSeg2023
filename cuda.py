import torch

import numpy as np

di = torch.rand(20, 20, 1)
print(di.dtype)


if torch.cuda.is_available():
    device = torch.device("cuda")
    all_device = True
else:
    print("cuda not available")
    all_device = False
    device = torch.device("cpu")

di.to(torch.device("cuda"))
