
import torch
import numpy as np
import time

def transformation(t):
    # t: n, d
    t_pad = -torch.sum(t, dim=0, keepdim=True)
    t = torch.cat([t, t_pad], dim=0)
    m, d = t.shape
    # solve the eq
    t_ifft = torch.fft.ifft(t, n=m, dim=-2, norm="ortho")
    b = torch.cat([torch.zeros(1, d).to(t.device), t_ifft[1:m] / np.sqrt(m)], dim=0)

    theta = -2j * torch.pi * torch.arange(1, m).reshape(1, -1, 1) / m
    theta = theta.to(t.device)
    
    return theta, b

def verify(t, b):
    # t: n, d
    t_pad = -torch.sum(t, dim=0, keepdim=True)
    t = torch.cat([t, t_pad], dim=0)
    m, d = t.shape
    t_res = np.sqrt(m) * (torch.fft.fft(b, n=m, dim=-2, norm="ortho"))

    print(torch.norm(t - t_res))

n = 512
d = 1
t = torch.rand(n, d) * 10
theta, b = transformation(t)
verify(t, b)