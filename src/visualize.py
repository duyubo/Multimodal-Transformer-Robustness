import torch
from torch import nn
import sys
from src import models
from src.utils import *
import torch.optim as optim
import numpy as np
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import pickle
from torchsummary import summary
import copy

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *
from src.models2 import *
import itertools

from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count_table

"""!!!!!!!!!!!!! Choose which MULT model version !!!!!!!!!!!!!"""
from src.dynamic_models2 import DynamicMULTModel
    
def initiate(hyp_params, train_loader, valid_loader, test_loader):
    if hyp_params.pretrain is not None:
        print("Load from pretrain model!!!!!!!!")
        model = torch.load(hyp_params.pretrain)
    else:
        model = DynamicMULTModel(
            origin_dimensions = hyp_params.orig_d, dimension = hyp_params.dimension, 
            num_heads = hyp_params.num_heads, head_dim = hyp_params.head_dim, 
            layers_single_attn = hyp_params.layers_single_attn, layers_hybrid_attn = hyp_params.layers_cross_attn, 
            layers_self_attn = hyp_params.layers_self_attn, attn_dropout = hyp_params.attn_dropout, 
            relu_dropout = hyp_params.relu_dropout, res_dropout = hyp_params.res_dropout, 
            out_dropout = hyp_params.out_dropout, embed_dropout = hyp_params.embed_dropout, 
            attn_mask = True, output_dim = hyp_params.output_dim, modality_set = hyp_params.modality_set,
            all_steps =  hyp_params.all_steps,
            stride = 0, # To be modified!!!!
            padding = 0, 
            kernel_size = 0, 
            experiment_type = hyp_params.experiment_type
        ) 
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    """if hyp_params.dataset == 'enrico':
        criterion = hyp_params.criterion
        if hyp_params.use_cuda:
            criterion = criterion.cuda()
    else:
    """
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    cmd_loss = CMD()
    settings = {
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler,
                'cmd': cmd_loss
              }
    
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    #criterion is used to train model, evaluation may need different criterion   
    scheduler = settings['scheduler']
    cmd_loss = settings['cmd']
    activation = nn.Sigmoid()
    
    def test_missing_modality(model, hyp_params):
        """!!!!! Test performance under modality drop !!!!!"""
        modalities = hyp_params.modality_set
        if hyp_params.experiment_type == 'baseline_ib':
            # no single modality in baseline_ib, generate automatically
            modality_choices = []
            for i in range(2, len(modalities) + 1):
              modality_choices.extend(itertools.combinations(list(range(len(modalities))), i))
        elif hyp_params.experiment_type == 'random_sample' or hyp_params.experiment_type == 'baseline_ic':
            # all combinations, generate automatically
            modality_choices = []
            for i in range(1, len(modalities) + 1):
              modality_choices.extend(itertools.combinations(list(range(len(modalities))), i))
        else:
            print('No ' + hyp_params.experiment_type)
            raise NotImplementedError

        loss_conditions = []
        for active_modality in modality_choices:
          print([modalities[m] for m in active_modality], ": { ")
          modality_list = [modalities[j] for j in active_modality]
          m = ModalityStr(modality_list)
          active_cross = [[]] * len(modalities)
          active_cross_output = [[]] * len(modalities)
          for j in active_modality:
              r = m.gen_modality_str(modalities[j])
              active_cross[j] = r.copy()
              active_cross_output[j] = r.copy() if r else [modalities[j]]
          max_acc = -100
          lay_single = itertools.combinations_with_replacement([ii for ii in range(hyp_params.layers_single_attn + 1)], len(modalities))
          best_layer_num = None
          best_active_output = None
          possible_active_cross = []
          if len(active_modality) == 2 and hyp_params.experiment_type == 'random_sample':
              """generate all possible modality combinations when there are only two active modalities"""
              a_c_o = [[]] * len(modalities)
              #1 
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0]]
              a[active_modality[1]] = [modality_list[1]]
              possible_active_cross.append(a)
              #2
              """a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0], modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)"""
              #3
              """a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1], modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)"""
              #4
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              a[active_modality[1]] = [modality_list[1]]
              possible_active_cross.append(a)
              #5
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0]]
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #6
              """a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)"""
              #7
              """a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)"""
              #8
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #9
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0], modality_list[0] + modality_list[1]]
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #10
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              a[active_modality[1]] = [modality_list[1], modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #11
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0], modality_list[0] + modality_list[1]]
              a[active_modality[1]] = [modality_list[1], modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #12
              """a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)"""
              #13
              """a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)"""
          else:
              possible_active_cross.append(active_cross_output)
          print('Possible Active Cross: ', possible_active_cross)
          """generate all possible active_cross_output, if there is only two modalities"""
          for lay_num in lay_single:
              if hyp_params.experiment_type == 'baseline_ic':
                  l = [hyp_params.layers_single_attn] * len(hyp_params.modality_set)
              elif hyp_params.experiment_type == 'baseline_ia' or hyp_params.experiment_type == 'baseline_ib':
                  l = [0] * len(hyp_params.modality_set)
              elif hyp_params.experiment_type == 'random_sample':
                  l = lay_num
              for a in possible_active_cross:
                  model.set_active(active_single_attn_layer_num = l, 
                                    active_self_attn_layer_num = hyp_params.layers_self_attn, 
                                    active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                                    active_dimension = hyp_params.dimension, 
                                    active_head_num = hyp_params.num_heads, 
                                    active_head_dim = hyp_params.head_dim, 
                                    active_modality = active_modality,
                                    active_cross = active_cross, 
                                    active_cross_output = a
                                  )
                  acc, results, truths = evaluate(model, criterion,  activate_modality = list(range(len(hyp_params.modality_set))), test=False)
                  if acc > max_acc:
                    best_results = results
                    best_active_output = a.copy()
                    best_layer_num = l
                    max_acc = acc
          print('best self atten layer number: ', best_layer_num, best_active_output, 'best validation accuracy: ', max_acc)
          model.set_active(active_single_attn_layer_num = best_layer_num, 
                              active_self_attn_layer_num = hyp_params.layers_self_attn, 
                              active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                              active_dimension = hyp_params.dimension, 
                              active_head_num = hyp_params.num_heads, 
                              active_head_dim = hyp_params.head_dim, 
                              active_modality = active_modality,
                              active_cross = active_cross, 
                              active_cross_output = best_active_output
                          )
          """Test parameter number!"""
          """net = model.get_active_subnet(active_single_attn_layer_num = best_layer_num, 
                              active_self_attn_layer_num = hyp_params.layers_self_attn, 
                              active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                              active_dimension = hyp_params.dimension, 
                              active_head_num = hyp_params.num_heads, 
                              active_head_dim = hyp_params.head_dim, 
                              active_modality = active_modality,
                              active_cross = active_cross, 
                              active_cross_output = best_active_output)
          
          rand_input = [torch.rand(1, 50, 300), torch.rand(1, 50, 74), torch.rand(1, 50, 35)]
          flops = FlopCountAnalysis(net, inputs = ([rand_input[i].cuda() for i in active_modality], ))
          print(flops.total())
          macs, params = profile(net, inputs = ([rand_input[i].cuda() for i in active_modality], ))
          print(macs, params)"""
          acc, best_results, truths = evaluate(model, criterion,  activate_modality = list(range(len(hyp_params.modality_set))), test=True)
          if hyp_params.dataset == 'avmnist':
              print('acc: ', acc)
          elif hyp_params.dataset == 'mosei_senti':
              eval_mosei_senti(best_results, truths, True)
          elif hyp_params.dataset == 'mojupush':
              print('MSE: ', -acc)
          elif hyp_params.dataset == 'enrico':
              print('acc: ', acc)
          elif hyp_params.dataset == 'eeg2a':
              print('acc: ', acc)
          elif hyp_params.dataset == 'kinects':
              raise NotImplementedError
          else:
              print(hyp_params.dataset + ' does not exist')
              raise NotImplementedError
          print("},")
        print("}")

    def masking_inputs(model, hyp_params):
        modalities = hyp_params.modality_set
        modality_choices = [[]]
        for i in range(1, len(modalities) + 1):
            modality_choices.extend(itertools.combinations(list(range(len(modalities))), i))
        
        for i in modality_choices:
          print([modalities[m] for m in i], ": { ")
          min_loss = 100
          loss, results, truths = evaluate(model, criterion,  activate_modality = i, test=True)
          best_results = results
          if hyp_params.dataset == 'avmnist':
              print('acc: ', multiclass_acc(best_results.argmax(dim=-1).numpy(), truths.numpy()))
          elif hyp_params.dataset == 'mosei_senti':
              eval_mosei_senti(best_results, truths, True)
          elif hyp_params.dataset == 'mojupush':
              print('MSE: ', criterion(best_results, truths))
          elif hyp_params.dataset == 'enrico':
              print('acc: ', multiclass_acc(best_results.argmax(dim=-1).numpy(), truths.numpy()))
          elif hyp_params.dataset == 'eeg2a':
              print('acc: ', multiclass_acc(results.argmax(dim=-1).numpy(), truths.numpy()))
          elif hyp_params.dataset == 'kinects':
              raise NotImplementedError
          else:
              print(hyp_params.dataset + ' does not exist')
              raise NotImplementedError
          print("},")
        print("}")

    best_valid = -1e8
    time_total_start = time.time()
    time_total_end = time.time()
    print(time_total_end - time_total_start)
    model = torch.load(hyp_params.model_path)
    if hyp_params.experiment_type == 'baseline_ia':
        masking_inputs(model, hyp_params)
    else:
        test_missing_modality(model, hyp_params)

  

