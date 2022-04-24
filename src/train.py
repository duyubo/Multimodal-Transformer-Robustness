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

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
from src.eval_metrics import *
from src.models2 import *
import itertools

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
            attn_mask = True, output_dim = hyp_params.output_dim, modality_set = hyp_params.modality_set
        ) 
    if hyp_params.use_cuda:
        model = model.cuda()
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {
                'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler
              }
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']    
    scheduler = settings['scheduler']
    
    def train(model, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        start_time = time.time()
        for i_batch, (batch_X, batch_Y) in enumerate(train_loader): 
            sample_ind = batch_X[0]
            inputs = batch_X[1:]
            eval_attr = batch_Y
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    inputs = [i.cuda() for i in inputs]
                    eval_attr = eval_attr.cuda()
                   
            batch_size = inputs[0].size(0)
            raw_loss = 0   
            preds = model(inputs)
            raw_loss += criterion(preds, eval_attr)

            """ set up active part """
            if hyp_params.random_sample:
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
                """ end set up active part """
            else:
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
                              

            combined_loss = raw_loss
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            proc_loss += raw_loss.item() * batch_size
            proc_size += batch_size
            epoch_loss += combined_loss.item() * batch_size
            if i_batch % hyp_params.log_interval == 0 and i_batch > 0:
                avg_loss = proc_loss / proc_size
                elapsed_time = time.time() - start_time
                print('Epoch {:2d} | Batch {:3d}/{:3d} | Time/Batch(ms) {:5.2f} | Train Loss {:5.4f}'.
                      format(epoch, i_batch, num_batches, elapsed_time * 1000 / hyp_params.log_interval, avg_loss))
                proc_loss, proc_size = 0, 0
                start_time = time.time()
            i_batch = i_batch + 1  
        return epoch_loss / hyp_params.n_train

    def evaluate(model, criterion, test=False, active_modality = [0, 1, 2]):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0
    
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
                preds = model(inputs)
                total_loss += criterion(preds, eval_attr).item() * batch_size
                # Collect the results into dictionary
                results.append(preds.cpu().detach())
                truths.append(eval_attr.cpu().detach())
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)
        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(1, hyp_params.num_epochs + 1):# 1, hyp_params.num_epochs + 1
        start = time.time()
        train(model, optimizer, criterion)
        """ set back to the full modality during eval and test"""
        if hyp_params.random_sample:
            model.set_active(
                            active_single_attn_layer_num = [hyp_params.layers_single_attn] * len(hyp_params.modality_set),
                            active_self_attn_layer_num = hyp_params.layers_self_attn, 
                            active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                            active_dimension = hyp_params.dimension, 
                            active_head_num = hyp_params.num_heads, 
                            active_head_dim = hyp_params.head_dim, 
                            active_modality = list(range(len(hyp_params.modality_set))),
                            active_cross = [model.m.gen_modality_str(i) for i in hyp_params.modality_set], 
                            active_cross_output = [model.m.gen_modality_str(i) for i in hyp_params.modality_set]
            )
        
        """ set back to the full modality during eval and test"""
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end - start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print("Saved model at ", hyp_params.model_path)
            torch.save(model, hyp_params.model_path)
            best_valid = val_loss
        if optimizer.param_groups[0]['lr'] <= 1e-6:
          break

    model = torch.load(hyp_params.model_path)
    """Test performance under modality drop"""
    modalities = hyp_params.modality_set
    modality_choices = hyp_params.modality_pool
    loss_conditions = []
    l = 0
    for i in modality_choices:
      print([modalities[m] for m in i], ": { ")
      active_modality = i
      modality_list = [modalities[j] for j in active_modality]
      m = ModalityStr(modality_list)
      active_cross = [[]] * len(modalities)
      active_cross_output = [[]] * len(modalities)
      for j in i:
          r = m.gen_modality_str(modalities[j])
          active_cross[j] = r.copy()
          active_cross_output[j] = r.copy() if r else [modalities[j]]
      min_loss = 100
      for lay_num in itertools.combinations_with_replacement([ii for ii in range(hyp_params.layers_single_attn + 1)], len(modalities)):
          model.set_active(active_single_attn_layer_num = lay_num, 
                            active_self_attn_layer_num = hyp_params.layers_self_attn, 
                            active_hybrid_attn_layer_num = hyp_params.layers_cross_attn, 
                            active_dimension = hyp_params.dimension, 
                            active_head_num = hyp_params.num_heads, 
                            active_head_dim = hyp_params.head_dim, 
                            active_modality = active_modality,
                            active_cross = active_cross, 
                            active_cross_output = active_cross_output
                            )
          loss, results, truths = evaluate(model, criterion, test=True)
          if loss < min_loss:
            min_loss = loss
            best_results = results
            best_layer_num = lay_num
      print('best self atten layer number: ', best_layer_num)
      #print(best_results[:10].argmax(dim=-1), truths[:10])
      print('acc: ', multiclass_acc(best_results.argmax(dim=-1).numpy(), truths.numpy()))
      #eval_mosei_senti(best_results, truths, True)
      print("},")
      l += 1
    print("}")

    """model.set_active(active_single_attn_layer_num = [hyp_params.layers_single_attn] * 3, 
                          active_self_attn_layer_num = layers_self_attn, 
                          active_hybrid_attn_layer_num = layers_hybrid_attn, 
                          active_dimension = dimension, 
                          active_head_num = num_heads, 
                          active_head_dim = head_dim, 
                          active_modality = active_modality,
                          active_cross = active_cross, 
                          active_cross_output = active_cross_output
                          )
    _, results, truths = evaluate(model, criterion, test=True)
    eval_mosei_senti(results, truths, True)"""

