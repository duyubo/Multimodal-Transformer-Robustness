import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

"""Sentiment Analysis Dataset: MOSEI Dataset"""
class MOSEI_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(MOSEI_Datasets, self).__init__()
        dataset_path = os.path.join(dataset_path, 'mosei_senti_data.pkl')
        dataset = pickle.load(open(dataset_path, 'rb'))
        self.vision = torch.tensor(dataset[split_type]['vision']).float()
        self.text = torch.tensor(dataset[split_type]['text']).float()
        self.audio = dataset[split_type]['audio']
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).float()
        self.labels = torch.tensor(dataset[split_type]['labels']).float()
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return [self.text.shape[1], self.audio.shape[1], self.vision.shape[1]]
    def get_dim(self):
        return [self.text.shape[2], self.audio.shape[2], self.vision.shape[2]]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = [index, self.text[index], self.audio[index], self.vision[index]]
        Y = self.labels[index].squeeze(-1)
        return X, Y

class avMNIST_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(avMNIST_Datasets, self).__init__()
        if split_type == 'test':
            self.image = torch.tensor(np.load(dataset_path + "/image/" + split_type + "_data.npy" )).float()
            self.audio = torch.tensor(np.load(dataset_path + "/audio/" + split_type + "_data.npy" )).float()
            self.labels = torch.tensor(np.load(dataset_path + "/" + split_type + "_labels.npy" )).long()
        else:
            image = torch.tensor(np.load(dataset_path + "/image/train_data.npy" )).float()
            audio = torch.tensor(np.load(dataset_path + "/audio/train_data.npy" )).float()
            labels = torch.tensor(np.load(dataset_path + "/train_labels.npy" )).long()

            if split_type == 'valid':
                self.image = image[55000:60000]
                self.audio = audio[55000:60000]
                self.labels = labels[55000:60000]
            else:
                self.image = image[:55000]
                self.audio = audio[:55000]
                self.labels = labels[:55000]

        self.image /= 255.0
        self.audio /= 255.0

        self.image = self.image.reshape(self.image.shape[0], 28, 28)
        self.image = self.image.unsqueeze(0)
        self.image = torch.nn.functional.interpolate(self.image,
                        size=(112, 112), 
                        mode='bilinear')
        self.image = self.image.squeeze(0)
        print(self.image.shape)
        """print(self.image.shape) 
        print(self.audio.shape)"""  
        
        self.n_modalities = 2 # vision/ audio

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return [self.image.shape[1], self.audio.shape[1]]
    def get_dim(self):
        return [self.image.shape[2], self.audio.shape[2]]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        X = [index, self.image[index], self.audio[index]]
        Y = self.labels[index]
        return X, Y 

class GentlePush_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(GentlePush_Datasets, self).__init__()

        