import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

############################################################################################
# This file provides basic processing script for the multimodal datasets we use. For other
# datasets, small modifications may be needed (depending on the type of the data, etc.)
############################################################################################

class Multimodal_Datasets(Dataset):
    def __init__(self, dataset_path, data='mosei_senti', split_type='train', if_align=False):
        super(Multimodal_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, 'mosei_senti_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]['vision']).float()
        self.text = torch.tensor(dataset[split_type]['text']).float()
        self.audio = dataset[split_type]['audio']
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).float()
        self.labels = torch.tensor(dataset[split_type]['labels']).float()
        self.data = data
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.text[index], self.audio[index], self.vision[index])
        Y = self.labels[index]
        return X, Y

