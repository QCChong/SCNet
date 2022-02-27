import torch
import random
import numpy as np

class AverageValueMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#initialize the weighs of the network for Convolutional layers and batchnorm layers
def weights_init(m):
    import math
    import torch.nn.init as init
    if isinstance(m, torch.nn.Linear):
        init.kaiming_normal_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, (torch.nn.Conv1d, torch.nn.Conv2d)):
        init.kaiming_normal_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(m.bias, -bound, bound)

    elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
        init.ones_(m.weight)
        init.zeros_(m.bias)