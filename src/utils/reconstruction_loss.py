import torch
def mse_power( x, y,power=3):
    temp = (x ** power - y ** power) **2
    return temp.mean()


def regularization(segments,alpha=0.013):
    n_segments = segments.shape[1]
    n_subjects = segments.shape[0]
    cos_sim = torch.nn.CosineSimilarity()
    err = 0
    for i in range(n_segments):
        in1 = segments[:,i,:,:].reshape(1,5*212*256)
        for j in range(n_segments):
            if i == j:
                continue
            in2 = segments[:, j, :, :].reshape(1, 5 * 212 * 256)
            sim = cos_sim(in1,in2)
            err += sim
    return alpha * err.mean()
