from re import S
import torch
import os
from src.dataset import MOSEI_Datasets, avMNIST_Datasets

def get_data(args, split='train'):
    dataset = str.lower(args.dataset.strip())
    if dataset == 'mosei_senti':
        data_path = os.path.join(args.data_path, dataset) + f'_{split}.dt'
        if not os.path.exists(data_path):
            print(f"  - Creating new {split} data")
            data = MOSEI_Datasets(args.data_path, split)
            torch.save(data, data_path)
        else:
            print(f"  - Found cached {split} data")
            data = torch.load(data_path)
        return data
    elif dataset == 'avmnist':
        data = avMNIST_Datasets(args.data_path, split)
        return data
      

