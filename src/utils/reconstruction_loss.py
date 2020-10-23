import torch


class ReconstructionLoss:
    def __init__(self):
        super()

    def compute_loss(self, x, y):
        return torch.dist(x, y, 2)