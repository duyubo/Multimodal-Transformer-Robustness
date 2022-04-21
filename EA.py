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
            new_sample = copy.copy(sample)
            probs = torch.rand(len(sample[1]),)
            for i in range(len(probs)):
                if probs[i] < self.mutate_prob:
                    subset_temp = self.model.gen_active_cross(active_modality = self.active_modality)
                    new_sample[0][i] = copy.copy(subset_temp[0][i])
                    new_sample[1][i] = copy.copy(subset_temp[1][i])
            """to be modified"""
            efficiency = 0
            if efficiency <= self.latency_constraint:
                return new_sample, efficiency

    """crossover"""
    def crossover(self, sample1, sample2):
        while True:
            new_sample = copy.copy(sample1)
            for i in range(len(new_sample[0])):
                random_i = random.choice([0, 1])
                if random_i == 0:
                  new_sample[0][i] = copy.copy(sample2[0][i])
                  new_sample[1][i] = copy.copy(sample2[1][i])

            """to be modified"""
            efficiency = 0
            if efficiency <= self.latency_constraint:
                return new_sample, efficiency
    
    """search"""
    def search(self):
        mutation_numbers = int(round(self.mutation_ratio * self.population_size))
        parents_size = int(round(self.parent_ratio * self.population_size))

        best_valids = [-10]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        best_info = None
        
        print("Generate random population...")
        child_pool.append([[['ta', 'tv'], ['at', 'av'], ['vt', 'va']],[['ta', 'tv'], ['at', 'av'], ['vt', 'va']]])
        child_pool.append([[['ta'], [], ['va', 'vat']], [['t', 'ta'], ['a'], ['va']]])
        child_pool.append([[['tv'], ['at', 'atv'], ['va', 'vat']], [['t', 'tv'], ['a', 'atv'], ['v', 'va', 'vat']]])
        child_pool.append([[['ta'], ['av', 'avt'], []], [['ta'], ['a', 'avt'], ['v']]])
        child_pool.append([[[], ['at'], ['vt', 'va', 'vta']], [['t'], ['a', 'at'], ['v', 'va', 'vta']]])
        for _ in range(self.population_size - 5):
            active_cross, active_cross_output = self.model.gen_active_cross(active_modality = self.active_modality)
            sample = [active_cross, active_cross_output]
            child_pool.append(sample)
       
        print("Get population accuracy")
        for i in range(self.population_size):
            self.model.set_active_modalities(
                  active_modality = self.active_modality,
                  active_cross = child_pool[i][0], 
                  active_cross_output = child_pool[i][1])    
            err = self.eval_model()
            print(err, child_pool[i][1])
            population.append((err, child_pool[i]))

        print("Start Evolution...")
        # After the population is seeded, proceed with evolving the population.
        for i in range(self.max_time_budget):
                parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
                err = parents[0][0]
                print("Iter: {} Err: {}".format(i, err))
                print(parents)
                if err > best_valids[-1]:
                    best_valids.append(err)
                    best_info = parents[0]
                else:
                    best_valids.append(best_valids[-1])
                print('best info in ', i, ':', best_info)    
                if i >= self.max_time_budget - 1:
                    self.model.set_active_modalities(
                      active_modality = self.active_modality,
                      active_cross = best_info[1][0], 
                      active_cross_output = best_info[1][1]) 
                    self.eval_model(test = True)
                    return best_valids, best_info

                population = parents
                child_pool = []

                '''Mutate'''
                for j in range(mutation_numbers):
                    par_sample = population[np.random.randint(parents_size)][1]
                    new_sample, efficiency = self.mutate(par_sample)
                    child_pool.append(new_sample)
                
                """Crossover"""
                for j in range(self.population_size - mutation_numbers):
                    par_sample1 = population[np.random.randint(parents_size)][1]
                    par_sample2 = population[np.random.randint(parents_size)][1]
                    new_sample, efficiency = self.crossover(
                        par_sample1, par_sample2
                    )
                    child_pool.append(new_sample)

                """Get accuracys for newly added population"""
                for j in range(self.population_size):
                    self.model.set_active_modalities(
                      active_modality = self.active_modality,
                      active_cross = child_pool[j][0], 
                      active_cross_output = child_pool[j][1])   
                    err = self.eval_model()
                    population.append(
                        (err, child_pool[j])
                    )

        
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
        test_preds = results.view(-1).numpy()#.cpu().detach()
        test_truth = truths.view(-1).numpy()

        non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])
        binary_truth = (test_truth[non_zeros] > 0)
        binary_preds = (test_preds[non_zeros] > 0)
        if test:
          eval_mosei_senti(results, truths, True)
        return accuracy_score(binary_truth, binary_preds)
    
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

parser.add_argument('--mutate_prob', type=float, default=0.8,
                    help='mutate_prob')
parser.add_argument('--parent_ratio', type=float, default=0.8,
                    help='parent_ratio')
parser.add_argument('--mutation_ratio', type=float, default=0.8,
                    help='mutation_ratio')     
parser.add_argument('--subnet_prob', type=float, default=0.5,
                    help='subnet_prob')
parser.add_argument('--population_size', type=int, default=100,
                    help='population_size')
parser.add_argument('--max_time_budget', type=float, default=200,
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
valid_data = get_data(args, dataset, 'valid')
test_data = get_data(args, dataset, 'test')

valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = True)
test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = True)
print('Finish loading the data....')

####################################################################
#
# Hyperparameters
#
####################################################################

hyp_params = args
hyp_params.use_cuda = use_cuda
hyp_params.orig_d_l, hyp_params.orig_d_a, hyp_params.orig_d_v = valid_data.get_dim()
hyp_params.l_len, hyp_params.a_len, hyp_params.v_len = valid_data.get_seq_len()
hyp_params.output_dim = output_dim_dict.get(dataset, 1)
hyp_params.criterion = torch.nn.L1Loss()
hyp_params.n_valid, hyp_params.n_test = len(valid_data), len(test_data)
print(hyp_params.n_valid, hyp_params.n_test)

if __name__ == '__main__':
    parent_model = torch.load(hyp_params.model_path)
    e = EvolutionSearch(parent_model, hyper_params = hyp_params, valid_loader = valid_loader, test_loader = test_loader)
    e.search()

