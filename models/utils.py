#!/usr/bin/env python3
# Author: hiro
# Date: 2022-09-22 19:08:25
# LastEditors: hiro
# Copyright (c) 2022 by hiro jinguang.tong@anu.edu.au, All Rights Reserved. 

import torch
import torch.nn as nn

import numpy as np

class Sine(nn.Module):
    def __init__(self, omega):
        super().__init__()
        
        self.omega = omega
        
    def forward(self, input):
        return torch.sin(self.omega * input)
    

def sine_init(m, is_first_layer=False, omega=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            if is_first_layer:
                bound = 1 / num_input
            else:
                bound = np.sqrt(6 / num_input) / omega
            m.weight.uniform_(-bound, bound)