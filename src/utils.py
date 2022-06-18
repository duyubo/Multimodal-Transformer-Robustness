from re import S
import torch
from torch import nn
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

class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

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
