import random

import torch


def set_seed(manualSeed: int):
    # Fix the seed value
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manualSeed)
        # if True, causes cuDNN to benchmark multiple convolution algorithms and select the fastest.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True  # if True, causes cuDNN to only use deterministic convolution algorithms.
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(manualSeed)


class WorkerInitializer:
    def __init__(self, manualSeed: int):
        self.manualSeed = manualSeed

    def worker_init_fn(self, worker_id: int):
        random.seed(self.manualSeed + worker_id)
