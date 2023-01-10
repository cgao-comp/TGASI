import numpy as np
import torch.nn as nn
import torch
import args
import random as rand
import heapq

# NN layers and models
import math
import torch
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GAT(nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        self.a =torch.tensor([1,2,3], dtype=float, requires_grad=True)
        self.weight = Parameter(torch.FloatTensor(3, 5))



if __name__ == '__main__':
    gat_model=GAT()
    train_params4 = list(filter(lambda p: p.requires_grad,gat_model.parameters()))
    print(train_params4)