import numpy as np
import torch


def __ema(p1, p2, factor):
    return factor * p1 + (1 - factor) * p2


def copy_params(src_model, dest_model):
    dest_model.load_state_dict(src_model.state_dict())


@torch.no_grad()
def ema_update(student, teacher, cur_step, epoch, alpha=0.999, max_step=600, max_epoch=5):
    # if cur_step < L:
    #     alpha = 0.99
    # else:
    #     alpha = 0.999
    if cur_step < max_step and epoch < max_epoch:
        alpha = min(np.exp(-5 * (1 - cur_step / max_step) ** 2), alpha)
    for stud_p, teach_p in zip(student.parameters(), teacher.parameters()):
        teach_p.data = __ema(teach_p.data, stud_p.data, alpha)

    for teach_p, stud_p in zip(teacher.buffers(), student.buffers()):
        teach_p.data = __ema(teach_p.data, stud_p.data, alpha)


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
    with open(path, 'wb') as f:
        torch.save(model.state_dict(), f)

@torch.no_grad()
def temp_rampDown(x_s, y_s, x_e, y_e, cur_x):

    if cur_x >= x_e :
        return y_e
    r = (y_e-y_s)/(x_e-x_s)
    cur_y = r * cur_x + y_s
    return cur_y


