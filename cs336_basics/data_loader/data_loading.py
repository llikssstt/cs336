import torch
import numpy as np
import torch.nn as nn

def data_loading(x, batch_size, context_length, device='cuda:0') -> tuple[torch.Tensor, torch.Tensor]:
    N = int(x.shape[0])

    max_start = N - context_length - 1  # -1 because targets use idx+1
    starts = torch.randint(
        low=0,
        high=max_start,
        size=(batch_size,),
        device=device,
        dtype=torch.long
    )
    offsets = torch.arange(context_length, device=device, dtype=torch.long)  # (T,)
    idx = starts[:, None] + offsets[None, :]  # (B, T)
    x = np.asarray(x).copy()         # 或 x = x.copy()
    x_t = torch.as_tensor(x, dtype=torch.long, device=device)
    inputs = x_t[idx]         # x[s : s+T]
    targets = x_t[idx + 1]    # x[s+1 : s+T+1]

    return inputs, targets