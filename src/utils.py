from re import S
import torch
import os
from src.dataset import MOSEI_Datasets, avMNIST_Datasets, GentlePush_Datasets, Enrico_Datasets, EEG2a_Datasets
from prettytable import PrettyTable

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_data(args, split='train', train_ratio = None, file_num_range_train = None, file_num_range_test = None):
    dataset = str.lower(args.dataset.strip())
    if dataset == 'mosei_senti':
        data = MOSEI_Datasets(args.data_path, split)
        return data
    elif dataset == 'avmnist':
        data = avMNIST_Datasets(args.data_path, split)
        return data
    elif dataset == 'mojupush':
        data = GentlePush_Datasets(args.data_path, split)
        return data
    elif dataset == 'enrico':
        data = Enrico_Datasets(args.data_path, split)
        return data
    elif dataset == 'eeg2a':
        data = EEG2a_Datasets(args.data_path, train_ratio = train_ratio, 
                              file_num_range_train = file_num_range_train, 
                              file_num_range_test = file_num_range_test, split_type=split)
        return data
    else:
        print(dataset + " does not exist!")
        raise NotImplementedError
