import numpy as np
import torch


def __ema(p1, p2, factor):
    return factor * p1 + (1 - factor) * p2


def ema_update(student, teacher, cur_step, L=400):
    if cur_step < L:
        alpha = 0.99
    else:
        alpha = 0.999

    for stud_p, teach_p in zip(student.parameters(), teacher.parameters()):
        teach_p.data = __ema(teach_p.data, stud_p.data, alpha)
    return student, teacher


def update_adaptiveRate(cur_step, L):
    if cur_step > L:
        return 1.0
    return np.exp(-5 * (1 - cur_step / L) ** 2)


def load_model(path, device):
    model = torch.load(path, map_location=device)
    return model


def load_state_dict(model, path):
    model.load_state_dict(torch.load(path))
    return model


def save_state_dict(model, path):
    torch.save(model.state_dict(), path)
