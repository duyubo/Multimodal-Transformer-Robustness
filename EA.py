import copy
from pickle import FALSE
import random
import numpy as np
import torch
from src.eval_metrics import *

__all__ = ["EvolutionFinder"]

"""This version only considers'modality combinations
and only under one modality conditions"""

def gen_subnet(parent_set:list, p):
    result = []
    probs = torch.rand((len(parent_set),))
    for i in range(len(probs)):
      if probs[i] < p:
        result.append(parent_set[i])
    return result

class EvolutionSearch:
    def __init__(self, parent_model, hyper_params, valid_loader, test_loader):
        """evolution hyper-parameters"""
        self.mutate_prob = hyper_params.mutate_prob
        self.population_size = hyper_params.population_size
        self.max_time_budget = hyper_params.max_time_budget
        self.parent_ratio = hyper_params.parent_ratio
        self.mutation_ratio = hyper_params.mutation_ratio
        self.subnet_prob = hyper_params.subnet_prob

        """"input info"""
        self.active_modality = hyper_params.active_modality
        self.hyper_params = hyper_params
        self.valid_loader = valid_loader
        self.test_loader =  test_loader
        self.criterion = hyper_params.criterion
        self.modality_list = hyper_params.modality_list
        
        """constraints """
        self.model = parent_model
        self.latency_constraint = 100

    """mutate"""
    def mutate(self, sample):
        """
        sample = [active_cross, active_cross_output]
        """
        while True:
            new_sample = copy.deepcopy(sample)
            probs = torch.rand(len(sample[1]),)
            for i in range(len(probs)):
                if probs[i] < self.mutate_prob:
                    subset_temp = self.model.gen_active_cross(active_modality = self.active_modality)
                    new_sample[0][i] = copy.deepcopy(subset_temp[0][i])
                    new_sample[1][i] = copy.deepcopy(subset_temp[1][i])
            """to be modified"""
            efficiency = 0
            if efficiency <= self.latency_constraint:
                return new_sample, efficiency

    """crossover"""
    def crossover(self, sample1, sample2):
        while True:
            new_sample = copy.deepcopy(sample1)
            for i in range(len(new_sample[0])):
                random_i = random.choice([0, 1])
                if random_i == 0:
                  new_sample[0][i] = copy.deepcopy(sample2[0][i])
                  new_sample[1][i] = copy.deepcopy(sample2[1][i])
            """to be modified"""
            efficiency = 0
            if efficiency <= self.latency_constraint:
                return new_sample, efficiency
    def get_acc(self, sample):
        self.model.set_active_modalities(
                  active_modality = self.active_modality,
                  active_cross = copy.deepcopy(sample[0]), 
                  active_cross_output = copy.deepcopy(sample[1]))    
        acc = self.eval_model()
        return acc
    """search"""
    def search(self):
        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-10]
        population = []  # (validation, sample, latency) tuples
        best_info = None
        
        print("Generate random population...")
        for _ in range(self.population_size):
            active_cross, active_cross_output = self.model.gen_active_cross(active_modality = self.active_modality)
            sample = [active_cross, active_cross_output]
            acc = self.get_acc(sample) 
            population.append([acc, sample])
            print(acc, sample[1])

        print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]  
                acc = parents[0][0]
                print("Iter: {} Acc: {}".format(i, acc), parents[0])
                if acc > best_valids[-1]:
                    best_valids.append(acc)
                    best_info = copy.deepcopy(parents[0])
                else:
                    best_valids.append(best_valids[-1])
                #print('best info in ', i, ':', best_info)    
                if i >= self.max_time_budget - 1:
                    self.model.set_active_modalities(
                      active_modality = self.active_modality,
                      active_cross = best_info[1][0], 
                      active_cross_output = best_info[1][1]) 
                    self.eval_model(test = True)
                    return best_valids, best_info

                population = copy.deepcopy(parents)

                """ Mutate """
                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    new_sample, efficiency = self.mutate(par_sample)
                    acc = self.get_acc(new_sample) 
                    population.append(
                        [acc, new_sample]
                    ) 
                    
                """ Crossover """
                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    new_sample, efficiency = self.crossover(par_sample1, par_sample2)
                    acc = self.get_acc(new_sample) 
                    population.append([acc, new_sample])
    
    """test with given modality combinations"""
    def test_modality(self, active_code):
        self.model.set_active_modalities(
                    active_modality = self.active_modality,
                    active_cross = active_code[0], 
                    active_cross_output = active_code[1])    
        acc = self.eval_model()
        print(acc)
        acc = self.eval_model(test = True)
        return acc
        
    def eval_model(self, test = False):
        hyp_params = self.hyper_params 
        self.model.eval()
        loader = self.valid_loader if not test else self.test_loader
        total_loss = 0.0
        results = []
        truths = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(loader):
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1)
                if hyp_params.use_cuda:
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()   
                preds = self.model([text, audio, vision]) #
                results.append(preds.cpu().detach())
                truths.append(eval_attr.cpu().detach())
        results = torch.cat(results)
        truths = torch.cat(truths)
        if test:
          eval_mosei_senti(results, truths,  exclude_zero = True)
        return binary_acc(results, truths, exclude_zero = True) + mosei_multiclass_acc(results, truths)
    
import sys
import torch
import argparse
from src.utils import *
from torch.utils.data import DataLoader
from src import train

parser = argparse.ArgumentParser(description='MOSEI Sentiment Analysis')
parser.add_argument('-f', default='', type=str)
parser.add_argument('--dataset', type=str, default='mosei_senti',
                    help='dataset to use (default: mosei_senti)')
parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/Colab_Notebooks/MultiBench-main/data',
                    help='path for storing the dataset')
parser.add_argument('--model_path', type=str, default='/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/MULT-single-test.pt',
                    help='path for storing the models')

# Tuning
parser.add_argument('--batch_size', type=int, default=584*2*2, metavar='N',
                    help='batch size')
# Logistics
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--no_cuda', action='store_true',
                    help='do not use cuda')

parser.add_argument('--mutate_prob', type=float, default=0.5,
                    help='mutate_prob')
parser.add_argument('--parent_ratio', type=float, default=0.8,
                    help='parent_ratio')
parser.add_argument('--mutation_ratio', type=float, default=0.8,
                    help='mutation_ratio')     
parser.add_argument('--subnet_prob', type=float, default=0.5,
                    help='subnet_prob')
parser.add_argument('--population_size', type=int, default=100,
                    help='population_size')
parser.add_argument('--max_time_budget', type=int, default=200,
                    help='max_time_budget')  
parser.add_argument('--active_modality', type=list, default=[0, 1, 2],
                    help='active_modality')        
parser.add_argument('--modality_list', type=list, default=['t', 'a', 'v'],
                    help='modality_list') 

args = parser.parse_args()

torch.manual_seed(args.seed)
dataset = str.lower(args.dataset.strip())
use_cuda = False

output_dim_dict = {
    'mosei_senti': 1,
}

criterion_dict = {

}

if torch.cuda.is_available():
    if args.no_cuda:
        print("WARNING: You have a CUDA device, so you should probably not run with --no_cuda")
    else:
        torch.cuda.manual_seed(args.seed)   
        use_cuda = True

####################################################################
#
# Load the dataset (aligned or non-aligned)
#
####################################################################

print("Start loading the data....")
valid_data = get_data(args, 'valid')
test_data = get_data(args, 'test')

valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False)
print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = valid_data.get_dim()
hyp_params.l = valid_data.get_seq_len()
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = torch.nn.L1Loss()
hyp_params.n_valid, hyp_params.n_test = len(valid_data), len(test_data)
print(hyp_params.n_valid, hyp_params.n_test)

if __name__ == '__main__':
    parent_model = torch.load(hyp_params.model_path)
    e = EvolutionSearch(parent_model, hyper_params = hyp_params, valid_loader = valid_loader, test_loader = test_loader)
    best_selection = e.search()
    #e.test_modality([[[], ['at', 'atv'], ['vt', 'va', 'vta']], [['t'], ['at'], ['v', 'vt', 'va']]])
