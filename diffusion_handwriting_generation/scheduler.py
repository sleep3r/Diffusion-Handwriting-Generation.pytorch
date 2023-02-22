from torch.optim.lr_scheduler import LambdaLR
import torch


class InvSqrtSchedule(LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        arg1 = torch.rsqrt(torch.tensor(step, dtype=torch.float32))
        arg2 = step * (self.warmup_steps ** -1.5)
        return torch.rsqrt(self.d_model) * torch.min(arg1, torch.tensor(arg2, dtype=torch.float32))
