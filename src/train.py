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

"""!!!!!!!!!!!!! Choose which MULT model !!!!!!!!!!!!!"""
from src.dynamic_models2 import DynamicMULTModel

####################################################################
#
# Construct the model and the CTC module (which may not be needed)
#
####################################################################
dimension = 40
num_heads = 1
head_dim = 20
layers_hybrid_attn = 4
layers_self_attn = 3 
modality_pool = [[0], [1], [2], [0, 1], [0, 2], [1, 2], [0, 1, 2]]

def initiate(hyp_params, train_loader, valid_loader, test_loader):
    
    model = DynamicMULTModel(origin_dimensions = [300, 74, 35], dimension = dimension, 
        num_heads = num_heads, head_dim = head_dim, layers_hybrid_attn = 4, layers_self_attn = 3, attn_dropout = [0.1, 0, 0, 0], 
        relu_dropout = 0, res_dropout = 0, out_dropout = 0, embed_dropout = 0, attn_mask = True, output_dim = 1, 
        modality_set = ['t', 'a', 'v']) 
    """Load from exist model"""
    model = torch.load('/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/MULT-single-test.pt')

    if hyp_params.use_cuda:
        model = model.cuda()

    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr)
    criterion = getattr(nn, hyp_params.criterion)()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}
    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

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
            sample_ind, text, audio, vision = batch_X
            eval_attr = batch_Y.squeeze(-1)   # if num of labels is 1
            
            model.zero_grad()
            if hyp_params.use_cuda:
                with torch.cuda.device(0):
                    text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                   
            batch_size = text.size(0)
            batch_chunk = hyp_params.batch_chunk
            raw_loss = 0   
            preds = model([text, audio, vision])
            raw_loss += criterion(preds, eval_attr)
            """ set up active part"""
            active_modality = modality_pool[torch.randint(low=0, high = len(modality_pool), size = (1, ))[0].item()]
            active_cross, active_cross_output = model.gen_active_cross(active_modality)
            model.set_active(active_self_attn_layer_num = layers_self_attn, 
                          active_hybrid_attn_layer_num = layers_hybrid_attn, 
                          active_dimension = dimension, 
                          active_head_num = num_heads,
                          active_head_dim = head_dim, 
                          active_modality = active_modality,
                          active_cross = active_cross, 
                          active_cross_output = active_cross_output)

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
                sample_ind, text, audio, vision = batch_X
                eval_attr = batch_Y.squeeze(dim=-1) # if num of labels is 1
            
                if hyp_params.use_cuda:
                    with torch.cuda.device(0):
                        text, audio, vision, eval_attr = text.cuda(), audio.cuda(), vision.cuda(), eval_attr.cuda()
                        
                batch_size = text.size(0)
                
                input_modalities = [text, audio, vision]
                preds = model(input_modalities) #
                
                total_loss += criterion(preds, eval_attr).item() * batch_size

                # Collect the results into dictionary
                results.append(preds.cpu().detach())
                truths.append(eval_attr.cpu().detach())
                
        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return avg_loss, results, truths

    best_valid = 1e8
    for epoch in range(0):# 1, hyp_params.num_epochs+1
        start = time.time()
        train(model, optimizer, criterion)

        """ set back to the full modality during eval and test"""
        model.set_active(active_self_attn_layer_num = layers_self_attn, 
                            active_hybrid_attn_layer_num = layers_hybrid_attn, 
                            active_dimension = dimension, 
                            active_head_num = num_heads, 
                            active_head_dim = head_dim, 
                            active_modality = [0, 1, 2],
                            active_cross = [model.m.gen_modality_str(i) for i in model.modality_list], 
                            active_cross_output = [model.m.gen_modality_str(i) for i in model.modality_list]
                            )
        """ set back to the full modality during eval and test"""
        val_loss, _, _ = evaluate(model, criterion, test=False)
        test_loss, _, _ = evaluate(model, criterion, test=True)
        
        end = time.time()
        duration = end-start
        scheduler.step(val_loss)    # Decay learning rate by validation loss

        print("-"*50)
        print('Epoch {:2d} | Time {:5.4f} sec | Valid Loss {:5.4f} | Test Loss {:5.4f}'.format(epoch, duration, val_loss, test_loss))
        print("-"*50)
        
        if val_loss < best_valid:
            print("Saved model at /content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness")
            torch.save(model, '/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/MULT-single-test.pt')
            best_valid = val_loss

    model = torch.load('/content/drive/MyDrive/Colab_Notebooks/Multimodal-Transformer-Robustness/MULT-single-test.pt')

    """Test performance under modality drop"""
    modalities = ['t', 'a', 'v']
    modality_choices = [[0],[1],[2],[0, 1],[0, 2],[1, 2],[0, 1, 2]]
    loss_conditions = ['\"no_audio_vision\"', '\"no_text_vision\"', '\"no_text_audio\"', '\"no_vision\"', '\"no_audio\"', '\"no_text\"', '\"full\"']
    l = 0
    print("result = {")
    for i in modality_choices:
      print(loss_conditions[l], ": { ")
      active_modality = i
      modality_list = [modalities[j] for j in active_modality]
      m = ModalityStr(modality_list)
      active_cross = [[]] * len(modalities)
      active_cross_output = [[]] * len(modalities)
      for j in i:
        r = m.gen_modality_str(modalities[j])
        active_cross[j] = r.copy()
        active_cross_output[j] = r.copy() if r else [modalities[j]]
      model.set_active(
                            active_self_attn_layer_num = layers_self_attn, 
                            active_hybrid_attn_layer_num = layers_hybrid_attn, 
                            active_dimension = dimension, 
                            active_head_num = num_heads, 
                            active_head_dim = head_dim, 
                            active_modality = active_modality,
                            active_cross = active_cross, 
                            active_cross_output = active_cross_output
                            )

      _, results, truths = evaluate(model, criterion, test=True, active_modality = i)
      eval_mosei_senti(results, truths, True)
      print("},")
      l += 1
    print("}")
    _, results, truths = evaluate(model, criterion, test=True)
    eval_mosei_senti(results, truths, True)

