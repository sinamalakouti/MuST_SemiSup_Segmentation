from utils.Constants import *
from utils.dataloader import *


# class Utils:
#     def __init__(self, dataset):

def get_dataset(dataset) -> torch.utils.data.DataLoader:
    mem_pin = False
    if Constants.USE_CUDA:
        mem_pin = True
    if dataset is Datasets.PittLocalFull:
        batch_sz = 5
        train = torch.utils.data.DataLoader(
            PittLocalFull(
                1,
                None,
                None,
                None,
                [f'paths/fold-0/data_paths_ws.txt', f'paths/fold-1/data_paths_ws.txt',
                 f'paths/fold-2/data_paths_ws.txt', f'paths/fold-3/data_paths_ws.txt',
                 f'paths/fold-4/data_paths_ws.txt'],
                [f'paths/fold-0/label_paths.txt', f'paths/fold-1/label_paths.txt', f'paths/fold-2/label_paths.txt',
                 f'paths/fold-3/label_paths.txt', f'paths/fold-4/label_paths.txt'],
                [f'paths/fold-0/mask_paths.txt', f'paths/fold-1/mask_paths.txt', f'paths/fold-2/mask_paths.txt',
                 f'paths/fold-3/mask_paths.txt', f'paths/fold-4/mask_paths.txt'],
                augment=False),
            batch_size=batch_sz,
            drop_last=True,
            num_workers=0,
            shuffle=True,
            pin_memory = mem_pin
        )

        return train

