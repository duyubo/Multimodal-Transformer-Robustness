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
from torch.nn.utils.rnn import pad_sequence

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
    translation_loss = nn.L1Loss() 

    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        i_batch = 0
        for batch_X, batch_Y in train_loader:
            inputs = batch_X[1:]
            eval_attr = batch_Y
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    inputs = [i.cuda() for i in inputs]
                    eval_attr = eval_attr.cuda()

            batch_size = inputs[0].size(0)  
            preds, translation_pair = model(inputs)
            #print(preds.shape, eval_attr.shape)
            raw_loss = criterion(preds, eval_attr)  
            """ set up active part """
            if hyp_params.experiment_type == 'random_sample':
                active_modality = hyp_params.modality_pool[torch.randint(low=0, high = len(hyp_params.modality_pool), size = (1, ))[0].item()]
                active_cross, active_cross_output = model.gen_active_cross(active_modality)
                active_single_attn_layer_num = torch.randint(low=0, high = hyp_params.layers_single_attn + 1, size = (len(hyp_params.modality_set),)).tolist()
                model.set_active(active_single_attn_layer_num = active_single_attn_layer_num,
                              active_self_attn_layer_num = hyp_params.layers_self_attn, 
                              active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                              active_dimension = hyp_params.dimension, 
                              active_head_num = hyp_params.num_heads,
                              active_head_dim = hyp_params.head_dim, 
                              active_modality = active_modality,
                              active_cross = active_cross, 
                              active_cross_output = active_cross_output)   
                #print(active_cross_output)           
            elif hyp_params.experiment_type == 'baseline_ic':
                if hyp_params.all_module: 
                    all_module_ic = model.m.gen_modality_str_all(hyp_params.modality_set)
                    active_cross_output = []
                    for i in hyp_params.modality_set:
                        active_cross_output_i = []
                        for module_ic in all_module_ic:
                            if i == module_ic[0]:
                                active_cross_output_i.append(module_ic)
                        active_cross_output.append(active_cross_output_i)
                    active_cross = copy.deepcopy(active_cross_output)
                else:
                    active_cross_output = [[i] + model.m.gen_modality_str(i) for i in hyp_params.modality_set]
                    active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set]
                
                model.set_active(
                                active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                                active_self_attn_layer_num = hyp_params.layers_self_attn, 
                                active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                                active_dimension = hyp_params.dimension, 
                                active_head_num = hyp_params.num_heads, 
                                active_head_dim = hyp_params.head_dim, 
                                active_modality = list(range(len(hyp_params.modality_set))),
                                active_cross = active_cross, 
                                active_cross_output = active_cross_output
                                )           
            elif hyp_params.experiment_type == 'baseline_ia' or hyp_params.experiment_type == 'baseline_ib':
                model.set_active(
                                active_single_attn_layer_num = [0] * len(hyp_params.modality_set),
                                active_self_attn_layer_num = hyp_params.layers_self_attn, 
                                active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                                active_dimension = hyp_params.dimension, 
                                active_head_num = hyp_params.num_heads, 
                                active_head_dim = hyp_params.head_dim, 
                                active_modality = list(range(len(hyp_params.modality_set))),
                                active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set], 
                                active_cross_output = [model.m.gen_modality_str(i) for i in hyp_params.modality_set]
                                )
            elif hyp_params.experiment_type == 'test_single':
                active_modality_set = [hyp_params.modality_set[i] for i in hyp_params.modality_pool[0]]
                test_single_m = ModalityStr(active_modality_set)
                active_cross = [[]] * len(hyp_params.modality_set)
                active_cross_output = [[]] * len(hyp_params.modality_set)
                if hyp_params.specific is not None:
                    active_cross = hyp_params.specific[0] 
                    active_cross_output = hyp_params.specific[1]
                elif len(active_modality_set) > 1:
                    for i in range(len(active_modality_set)):
                        active_cross[hyp_params.modality_pool[0][i]] = test_single_m.gen_modality_str(active_modality_set[i]) 
                        active_cross_output[hyp_params.modality_pool[0][i]] = test_single_m.gen_modality_str(active_modality_set[i]) 
                else:
                    active_cross_output[hyp_params.modality_pool[0][0]] = active_modality_set
                #print(active_cross, active_cross_output)
                model.set_active(
                                active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                                active_self_attn_layer_num = hyp_params.layers_self_attn, 
                                active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                                active_dimension = hyp_params.dimension, 
                                active_head_num = hyp_params.num_heads, 
                                active_head_dim = hyp_params.head_dim, 
                                active_modality = hyp_params.modality_pool[0],
                                active_cross = active_cross, 
                                active_cross_output = active_cross_output
                                )
                
            else:
              print("No such experiment") 
              raise(NotImplementedError)
            """ end set up active part """                
            raw_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += raw_loss.item() * batch_size
            raw_loss.detach()

            del inputs
            torch.cuda.empty_cache()

            i_batch += 1
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
             
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, activate_modality, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        results = []
        truths = []
        with torch.no_grad():
            for i_batch, (batch_X, batch_Y) in enumerate(loader):
                sample_ind = batch_X[0]
                inputs = batch_X[1:]
                eval_attr = batch_Y
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        inputs = [i.cuda() for i in inputs]    
                        eval_attr = eval_attr.cuda()   
                batch_size = inputs[0].size(0)
                preds, _ = model([inputs[i] if ( i in activate_modality) else (torch.zeros(inputs[i].size()).cuda()) for i in range(len(inputs))])
                results.append(preds.cpu().detach())
                truths.append(eval_attr.cpu().detach())
                del inputs
                torch.cuda.empty_cache()
                

        results = torch.cat(results)
        truths = torch.cat(truths)
        if hyp_params.dataset == 'avmnist':
            r = multiclass_acc(results.argmax(dim=-1).numpy(), truths.numpy())
        elif hyp_params.dataset == 'mosei_senti':
            if test:
                r = binary_acc(results, truths, True) 
            else:
                #criterion_test = nn.L1Loss()
                #r = 0-criterion_test(results, truths)
                r = binary_acc(results, truths, True)
                #r += multiclass_acc_eval(results, truths)
        elif hyp_params.dataset == 'mojupush':
            r = 0-criterion(results, truths)
        elif hyp_params.dataset == 'enrico':
            r = multiclass_acc(results.argmax(dim=-1).numpy(), truths.numpy())
        elif hyp_params.dataset == 'eeg2a':
            r = multiclass_acc(results.argmax(dim=-1).numpy(), truths.numpy())
        elif hyp_params.dataset == 'kinects':
            raise NotImplementedError
        else:
            print(hyp_params.dataset + ' does not exist')
            raise NotImplementedError        
        return r, results, truths 

    def test_missing_modality(model, hyp_params):
        """!!!!! Test performance under modality drop !!!!!"""
        modalities = hyp_params.modality_set
        if hyp_params.experiment_type == 'baseline_ib':
            # no single modality in baseline_ib, generate automatically
            modality_choices = []
            for i in range(2, len(modalities) + 1):
              modality_choices.extend(itertools.combinations(list(range(len(modalities))), i))
        elif hyp_params.experiment_type == 'random_sample' or hyp_params.experiment_type == 'baseline_ic' or hyp_params.experiment_type == 'test_single':
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
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0], modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)
              #3
              a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1], modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
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
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)
              #7
              a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
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
              a = a_c_o.copy()
              a[active_modality[1]] = [modality_list[1] + modality_list[0]]
              possible_active_cross.append(a)
              #13
              a = a_c_o.copy()
              a[active_modality[0]] = [modality_list[0] + modality_list[1]]
              possible_active_cross.append(a)
          elif len(active_modality) > 1 and hyp_params.experiment_type == 'test_single':
              if hyp_params.specific is not None:
                possible_active_cross.append(hyp_params.specific[1])
          else:
              possible_active_cross.append(active_cross_output)
          print('Possible Active Cross: ', possible_active_cross)
          """generate all possible active_cross_output, if there is only two modalities"""
          for lay_num in lay_single:
              if hyp_params.experiment_type == 'baseline_ic' or hyp_params.experiment_type == 'test_single' :
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
    training_curve = []
    time_total_start = time.time()
    for epoch in range(1, hyp_params.num_epochs + 1):# 1, hyp_params.num_epochs + 1
        start = time.time()
        train(model, optimizer, criterion)

        """calculate the missing modality performance"""
        if hyp_params.experiment_type == 'random_sample':
            val_acc = 0
            for i in range(len(hyp_params.modality_set)):
                active_cross_output = [[]] * len(hyp_params.modality_set)
                active_cross_output[i] = [hyp_params.modality_set[i]]
                model.set_active(
                            active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                            active_self_attn_layer_num = hyp_params.layers_self_attn, 
                            active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                            active_dimension = hyp_params.dimension, 
                            active_head_num = hyp_params.num_heads, 
                            active_head_dim = hyp_params.head_dim, 
                            active_modality = list(range(len(hyp_params.modality_set))),
                            active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set] if len(hyp_params.modality_set) > 1 else [[]], 
                            active_cross_output = [[i] + model.m.gen_modality_str(i) for i in hyp_params.modality_set ] #if i == a else []
                )
                val_acc += evaluate(model, criterion, activate_modality = list(range(len(hyp_params.modality_set))), test=False)[0]
            
        """ set back to the full modality during eval and test, Begin"""
        if  hyp_params.experiment_type == 'baseline_ic' or hyp_params.experiment_type == 'random_sample':
            model.set_active(
                            active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                            active_self_attn_layer_num = hyp_params.layers_self_attn, 
                            active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                            active_dimension = hyp_params.dimension, 
                            active_head_num = hyp_params.num_heads, 
                            active_head_dim = hyp_params.head_dim, 
                            active_modality = list(range(len(hyp_params.modality_set))),
                            active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set] if len(hyp_params.modality_set) > 1 else [[]], 
                            active_cross_output = [[i] + model.m.gen_modality_str(i) for i in hyp_params.modality_set] if len(hyp_params.modality_set) > 1 else hyp_params.modality_set
            )
        elif hyp_params.experiment_type == 'test_single' and len(hyp_params.modality_pool) > 1:
            active_cross_output = [[]] * len(hyp_params.modality_set)
            active_cross_output[hyp_params.modality_pool[1][0]] = [hyp_params.modality_set[hyp_params.modality_pool[1][0]]]
            print('validation on ', active_cross_output)
            model.set_active(
                            active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                            active_self_attn_layer_num = hyp_params.layers_self_attn, 
                            active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                            active_dimension = hyp_params.dimension, 
                            active_head_num = hyp_params.num_heads, 
                            active_head_dim = hyp_params.head_dim, 
                            active_modality = list(range(len(hyp_params.modality_set))),
                            active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set] if len(hyp_params.modality_set) > 1 else [[]], 
                            active_cross_output =  active_cross_output
            )
        
        end = time.time()
        """ set back to the full modality during eval and test, End"""
        val_acc1, _, _ = evaluate(model, criterion, activate_modality = list(range(len(hyp_params.modality_set))), test=False)
        if hyp_params.experiment_type == 'random_sample':
            val_acc += val_acc1
        else:
            val_acc = val_acc1
        test_acc, _, _ = evaluate(model, criterion, activate_modality = list(range(len(hyp_params.modality_set))), test=True)
        
        training_curve.append([val_acc, test_acc])
        duration = end - start
        scheduler.step(1-val_acc)# Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Acc {:5.4f} | Test Acc {:5.4f}'.format(epoch, duration, abs(val_acc), abs(test_acc)))
        print("-"*50)
        
        if val_acc > best_valid:
            print("Saved model at ", hyp_params.model_path)
            torch.save(model, hyp_params.model_path)
            best_valid = val_acc
        if optimizer.param_groups[0]['lr'] <= 1e-16:
          break

    time_total_end = time.time()
    print(time_total_end - time_total_start)
    print(training_curve)
    #model = torch.load(hyp_params.model_path)
    """test effects of replacing translation header"""
    """new_model = torch.load(hyp_params.model_path)
    for param_tensor in model.state_dict():
          model.state_dict()[param_tensor].copy_(new_model.state_dict()[param_tensor])
    if hyp_params.experiment_type == 'baseline_ia':
        masking_inputs(model, hyp_params)
    else:
        test_missing_modality(model, hyp_params)
    """
    

