import torch
import torch.nn as nn
import math
def learning_rate_schedule(t, a_max, a_min, T_w, T_c):
    if t < T_w:
        a_t = t * a_max / T_w
    elif t >= T_w and t <= T_c:
        a_t = a_min + 1/2 * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi))*(a_max, a_min)
    elif t > T_c:
        a_t = a_min
    return a_t 