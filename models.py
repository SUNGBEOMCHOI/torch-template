import torch 
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model = nn.Sequential()

    def forward(self, x):
        x = self.model(x)
        return x
