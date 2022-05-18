from pickle import FALSE
import sys
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train
from src.data_utils import compute_weights
import random

parser = argparse.ArgumentParser(description='MULT Multimodality Learning')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--dataset', type=str, default=None,
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default=None,
                    help='path for storing the models')

# Dropouts
parser.add_argument('--attn_dropout', nargs="*", type=float, default=[0.1, 0, 0],
                    help='attention dropout')
parser.add_argument('--relu_dropout', type=float, default=0.1,
                    help='relu dropout')
parser.add_argument('--embed_dropout', type=float, default=0.3,
                    help='embedding dropout')
parser.add_argument('--res_dropout', type=float, default=0.3,
                    help='residual block dropout')
parser.add_argument('--out_dropout', type=float, default=0.1,
                    help='output layer dropout')

# Architecture
parser.add_argument('--dimension', type=int, default=30,
                    help='number of hiddenlayers in the network (default: 30)')
parser.add_argument('--layers_cross_attn', type=int, default=3,
                    help='number of layers in the cross attention(default: 3)')     
parser.add_argument('--layers_single_attn', type=int, default=3,
                    help='number of layers in the single attention(default: 3)')
parser.add_argument('--layers_self_attn', type=int, default=3,
                    help='number of layers in the self attention(default: 3)')               
parser.add_argument('--num_heads', type=int, default=5,
                    help='number of heads for the transformer network (default: 5)')
parser.add_argument('--head_dim', type=int, default=6,
                    help='hidden dimensions for each head (default: 6)')
parser.add_argument('--attn_mask', action='store_false',
                    help='use attention mask for Transformer (default: true)')
parser.add_argument('--modality_pool', type=int, nargs='+', action='append', default=None,
                    help='possible modality combinations [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]')
parser.add_argument('--modality_set', type=str, nargs="*", default=['t', 'a', 'v'],
                    help=' a list of modality names [\'t\', \'a\', \'v\']')
parser.add_argument('--all_steps', action = 'store_true', help = 'keep all intermidiate results')     

# Tuning
parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                    help='batch size (default: 24)')
parser.add_argument('--clip', type=float, default=1.0,
                    help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 1e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs (default: 40)')
parser.add_argument('--when', type=int, default=10,
                    help='when to decay learning rate (default: 20)')
parser.add_argument('--batch_chunk', type=int, default=1,
                    help='number of chunks per batch (default: 1)')

# Logistics
parser.add_argument('--log_interval', type=int, default=360,
                    help='frequency of result logging (default: 30)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')

# Stages
parser.add_argument('--pretrain', type=str, default = None,
                    help='Whether load pretrain model')
parser.add_argument('--experiment_type', type=str, default = 'random_sample', 
                    help='Which experiment?')

args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
output_dim_dict = {
    'mosei_senti': 1,
    'avmnist': 10,
    'mojupush': 2,
    'enrico': 20,
    'eeg2a': 4,
}

criterion_dict = {
    'mosei_senti':  'L1Loss',
    'avmnist':   'CrossEntropyLoss', 
    'mojupush':  'MSELoss',
    'enrico': 'CrossEntropyLoss',
    'eeg2a': 'CrossEntropyLoss',
    'kinects': None, 
}

batch_sizes = {
    'mosei_senti':  128 * 4,
    'avmnist':   128 * 4, 
    'mojupush': 128 * 4,
    'enrico': 128 * 4,
    'eeg2a': 64,
    'kinects': None, 
}
if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)   
        args.use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################



####################################################################
#
# Hyperparameters
#
####################################################################



if __name__ == '__main__':
    print("Start loading the data....")

   
    for i in range(1, 10):
        """
        # cross test
        file_num_range_train = ['A0' + str(j) + 'E.mat' for j in range(1, 10) if j != i]
        file_num_range_train.extend(['A0' + str(j) + 'T.mat' for j in range(1, 10) if j != i])
        file_num_range_test = ['A0' + str(i) + 'E.mat', 'A0' + str(i) + 'T.mat']"""

        """#single test
        file_num_range_train = ['A0' + str(i) + 'T.mat']
        file_num_range_test = ['A0' + str(i) + 'E.mat']"""

        #combine test
        file_num_range_train = ['A0' + str(j) + 'T.mat' for j in range(1, 10)]
        file_num_range_test = ['A0' + str(j) + 'E.mat' for j in range(1, 10)]

        print(file_num_range_train)
        print(file_num_range_test)
        train_data = get_data(args, split = 'train', train_ratio = 0.8, file_num_range_train = file_num_range_train, file_num_range_test = file_num_range_test)
        valid_data = get_data(args, split = 'valid', train_ratio = 0.8, file_num_range_train = file_num_range_train, file_num_range_test = file_num_range_test)
        test_data = get_data(args, split = 'test', train_ratio =0.8, file_num_range_train = file_num_range_train, file_num_range_test = file_num_range_test)

        train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True)
        valid_loader = DataLoader(valid_data, batch_size = batch_sizes[args.dataset], shuffle = False)
        test_loader = DataLoader(test_data, batch_size = batch_sizes[args.dataset], shuffle = False)

        print('Finish loading the data....')
        hyp_params = args
        hyp_params.orig_d = train_data.get_dim()
        hyp_params.l = train_data.get_seq_len()
        hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
        hyp_params.output_dim = output_dim_dict[hyp_params.dataset]
        hyp_params.criterion = criterion_dict[hyp_params.dataset]
        hyp_params.model_path = '/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/MULT-eeg2a-cross' + str(i) + '.pt'
        print('orig_d:', hyp_params.orig_d)
        print('attn_dropout:', hyp_params.attn_dropout)
        print('modality_set:', hyp_params.modality_set)
        print('modality_pool:', hyp_params.modality_pool)
        print('criterion: ', hyp_params.criterion)
        print('batch size: ', hyp_params.batch_size)
        print('num of train: ', hyp_params.n_train)
        print('sequence length: ', hyp_params.l)

        test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)

