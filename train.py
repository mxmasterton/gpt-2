import sys
import time
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from model import GPTConfig, GPT

from dataloader import DataLoader

# -------------------------------------------------------------

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print("using device: {}".format(device))

torch.manual_seed(2005)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2005)
    
torch.set_float32_matmul_precision("high")

# -------------------------------------------------------------

train_loader = DataLoader(B=16, T=1024)

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)

# optimise!
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    
    optimiser.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time difference in milectons
    print("{}/ {}/ {}".format(i, loss.item(), dt))
    
sys.exit(0)