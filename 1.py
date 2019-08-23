import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

a = torch.randn(3, 4)
b = a.unsqueeze(1)
print(a.size(),b.size())