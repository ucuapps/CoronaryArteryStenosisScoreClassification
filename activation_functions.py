import torch.nn as nn
import torch
import torch.functional as F

class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()
        self.relu = nn.ReLU()
    def forward(self, x):
        return torch.cat((self.relu(x),self.relu(-x)))


class CustomSwish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
    
class Swish(nn.Module):
    def forward(self, input_tensor):
        return CustomSwish.apply(input_tensor)