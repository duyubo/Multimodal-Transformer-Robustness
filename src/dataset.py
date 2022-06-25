from re import T
import numpy as np
from numpy.lib.index_tricks import index_exp
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch
import numpy as np
import sys
from src.data_utils import *
import random
import csv
import json
from collections import Counter
import torch
from torchvision import transforms
from PIL import Image
import scipy
import scipy.io as sio
import h5py
from torch.nn.utils.rnn import pad_sequence
from transformers import *
bert_tokenizer = BertTokenizer.from_pretrained('/home/yubo/Multimodal-Transformer-Robustness/bert_en', do_lower_case=True)

def collate_fn_mosei(batch):
    #sample 0: index, 1: text, 2: audio, 3: vision
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things 

    labels = torch.cat([sample[1] for sample in batch], dim = 0) 
    visual = pad_sequence([torch.FloatTensor(sample[0][3].cpu()) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2].cpu()) for sample in batch])
    #print(visual.shape, batch[0][0][3].shape, batch[1][0][3].shape)
    SENT_LEN = 0
    for sample in batch:
        if len(sample[0][1]) > SENT_LEN:
             SENT_LEN = len(sample[0][1])
    # Create bert indices using tokenizer
    bert_details = []
    for sample in batch:
        text = " ".join(sample[0][1])
        encoded_bert_sent = bert_tokenizer.encode_plus(
            text, add_special_tokens=True, max_length=SENT_LEN+2,  pad_to_max_length=True) # , max_length=SENT_LEN+2
        
        bert_details.append(encoded_bert_sent)

    # Bert things are batch_first
    
    bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
    bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
    bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])
    
    text = torch.stack([bert_sentences, bert_sentence_types, bert_sentence_att_mask])
    #text = bert_sentences
    return [0, text, acoustic.permute(1, 0, 2), visual.permute(1, 0, 2)], labels.unsqueeze(1)
  
"""New generated MOSEI dataset: """
"""Under Test"""
class MOSEI_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(MOSEI_Datasets, self).__init__()
        if split_type == 'test':
            dataset = []
            for i in range(1, 48):
                dataset_p = os.path.join(dataset_path, f"processed_data_test{i*100}.pt")
                dataset.extend(torch.load(dataset_p, map_location='cpu'))
        elif split_type == 'train':
            dataset = []
            for i in range(1, 165):
                dataset_p = os.path.join(dataset_path, f"processed_data_train{i*100}.pt")
                dataset.extend(torch.load(dataset_p, map_location='cpu'))
        else:
            dataset = []
            for i in range(1, 20):
                dataset_p = os.path.join(dataset_path, f"processed_data_valid{i*100}.pt")
                dataset.extend(torch.load(dataset_p, map_location='cpu'))
        self.vision = []
        print("!!!!!!!!!!!!!!!!!!!!!!!", split_type)
        empty_list = []
        count = 0
        for i in range(len(dataset)):
            if type(dataset[i][2]) == list:
                if dataset[i][2] == []:
                    print(dataset[i][0])
                    count += 1
                    empty_list.append(i)
                    vision = torch.zeros(1, 1, 512)
                else:
                    vision = torch.stack(dataset[i][2])
            else:
                vision = dataset[i][2]
            #print(vision.shape)
            #reshape_size = vision.shape[0]//4 if vision.shape[0]//4 > 50 else 50
            #vision = torch.nn.functional.interpolate(vision.reshape(1, vision.shape[0], 512).permute(0, 2, 1), size = (reshape_size)).permute(0, 2, 1).reshape(reshape_size, 512)
            #print(vision.shape)
            self.vision.append(vision)
        print(count)
        #self.vision = [dataset[i][2] for i in range(len(dataset))]

        self.text = [dataset[i][-2] for i in range(len(dataset)) if not i in empty_list] 

        self.audio = [dataset[i][-1] for i in range(len(dataset)) if not i in empty_list] 
        self.name = [dataset[i][0] for i in range(len(dataset)) if not i in empty_list]
        self.labels = torch.tensor([dataset[i][1] for i in range(len(dataset)) if not i in empty_list])

        self.n_modalities = 3 # vision/ text/ audio
        non_zeros = np.array([self.labels[i] for i in range(len(self.labels)) if self.labels[i] != 0])

        print(split_type, len([i for i in non_zeros if i > 0])/len(non_zeros))
        self.len = 50
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.len
    def get_dim(self):
        return [768, 768, 512]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.name)
    def __getitem__(self, index):
        X = [index, 
            self.text[index], 
            self.audio[index].squeeze(), 
            self.vision[index].squeeze(1)]
        
        Y = self.labels[index].unsqueeze(-1)
        return X, Y


"""Sentiment Analysis Dataset: MOSEI Dataset without BERT"""
class CMOSEI_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(MOSEI_Datasets, self).__init__()
        dataset_p = os.path.join(dataset_path, 'mosei_raw.pkl')
        dataset = pickle.load(open(dataset_p, 'rb'))
        self.vision = torch.tensor(dataset[split_type]['vision']).float()
        self.text = torch.tensor(dataset[split_type]['text']).float()
        self.audio = dataset[split_type]['audio']
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).float()
        self.labels = torch.tensor(dataset[split_type]['labels'][:, :, 0]).float().unsqueeze(-1)
        print(dataset[split_type].keys())
        self.n_modalities = 3 # vision/ text/ audio
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1]
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

"""Sentiment Analysis Dataset: MOSEI Dataset with BERT"""
class CMOSEI_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train'):
        super(MOSEI_Datasets, self).__init__()
        # same preprocessing as MSAF
        dataset_folder = os.path.join(dataset_path, split_type)
        vision = np.load(os.path.join(dataset_folder, "visual50.npy")).astype(np.float32)
        audio = np.load(os.path.join(dataset_folder, "audio50.npy")).astype(np.float32)
        audio[audio == -np.inf] = 0
        EPS = 1e-6
        self.vision = torch.tensor(vision).float().cpu().detach()#
        self.audio = torch.tensor(audio).float().cpu().detach()#
        #self.vision = torch.tensor(np.array([np.nan_to_num((v - np.mean(v, axis = 0, keepdims=True)) / (EPS + np.std(v, axis=0, keepdims=True))) for v in vision])).float().cpu().detach()
        #self.audio = torch.tensor(np.array([np.nan_to_num((a - np.mean(a, axis = 0, keepdims=True)) / (EPS + np.std(a, axis=0, keepdims=True))) for a in audio])).float().cpu().detach()
        #self.vision = torch.tensor(1/(1 + np.exp(-vision))).float().cpu().detach()
        #self.audio = torch.tensor(1/(1 + np.exp(-audio))).float().cpu().detach()
        
        self.text = torch.tensor(np.load(os.path.join(dataset_folder, "bert50.npy"))).float()
        self.labels = torch.tensor(np.load(os.path.join(dataset_folder, "label50.npy"))[:, :, 0]).float()
      
        self.n_modalities = 3 # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.text.shape[1]
    def get_dim(self):
        return [self.text.shape[2], self.audio.shape[2], self.vision.shape[2]]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = [index, self.text[index], self.audio[index], self.vision[index]]
         
        Y = self.labels[index]
        return X, Y

class avMNIST_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train', n_patches = 4):
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

        self.audio /= 255.0
        l = self.image.shape[0]
        d = int(self.image.shape[1] ** 0.5)
        da = int(self.audio.shape[1])
        self.image = self.image.reshape(l, d, d, 1).permute(0, 3, 1, 2)
        self.audio = self.audio.reshape(l, da, da, 1).permute(0, 3, 1, 2)
        #self.image = self.image.reshape(l, n_patches, d//n_patches, n_patches, d//n_patches).permute(0, 1, 3, 2, 4).reshape(l, n_patches **2, -1)
        #self.audio = self.audio.reshape(l, n_patches, da//n_patches, n_patches, da//n_patches).permute(0, 1, 3, 2, 4).reshape(l, n_patches **2, -1)  
        #self.audio = self.audio.reshape(l, n_patches ** 2, da//(n_patches ** 2), da).reshape(l, n_patches ** 2 , -1)  
        #self.audio = self.audio.reshape(l, n_patches ** 2, da//(n_patches ** 2), da).reshape(l, n_patches ** 2 , -1)
        self.n_modalities = 2 # vision/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.image.shape[1]

    def get_dim(self):
        return [self.image.shape[2], self.audio.shape[2]]#image

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
    def __init__(self, dataset_path, split_type='train', vision_interval: int = 10,
                    sequential_image_rate: int = 1,
                    start_timestep: int = 0, visual_noise: int = 0,
                    prop_noise: int = 0, haptics_noise: int = 0,
                    controls_noise: int = 0, multimodal_noise: int = 0, 
                    subsequence_length: int = 16):
        super(GentlePush_Datasets, self).__init__()
        self.dataset_path = dataset_path
        if split_type  == 'train':
          self.data_name = "gentle_push_1000.hdf5"
        elif split_type == 'valid':
          self.data_name = "gentle_push_10.hdf5"
        elif split_type == 'test':
          self.data_name = "gentle_push_300.hdf5"
        self.n_modalities = 4

        """parameters for data augmentation"""
        self.vision_interval = vision_interval
        self.start_timestep = start_timestep
        self.visual_noise = visual_noise
        self.prop_noise = prop_noise
        self.haptics_noise = haptics_noise
        self.controls_noise = controls_noise
        self.subsequence_length = subsequence_length
        self.multimodal_noise = multimodal_noise
        self.noise_range = 0
        
        self.trajectories = self._load_trajectories([os.path.join(self.dataset_path,self.data_name)])
        
        if split_type == 'test':
            print("Adding visual noise!!")
            for i in range(self.noise_range):
                self.visual_noise = i/10
                self.trajectories.extend(self._load_trajectories(
                            [os.path.join(self.dataset_path,self.data_name)]))
            print("Adding prop noise!!")
            for i in range(self.noise_range):
                self.prop_noise = i/10
                self.trajectories.extend(
                    self._load_trajectories([os.path.join(self.dataset_path,self.data_name)]))

            print("Adding haptics noise!!")
            for i in range(self.noise_range):
                self.haptics_noise = i/10
                self.trajectories.extend(
                    self._load_trajectories([os.path.join(self.dataset_path,self.data_name)]))
            
            print("Adding controls noise!!")
            for i in range(self.noise_range):
                self.controls_noise=i/10
                self.trajectories.extend(
                    self._load_trajectories([os.path.join(self.dataset_path,self.data_name)]))
            
            print("Adding multimodal noise!!")
            for i in range(self.noise_range):
                self.multimodal_noise=i/10
                self.trajectories.extend(
                    self._load_trajectories([os.path.join(self.dataset_path,self.data_name)]))                         
        
        self.subsequences = self.split_trajectories()
        
    
    def _load_trajectories(self, path):
        trajectories = []
        for name in path:
            max_trajectory_count = sys.maxsize
            if type(name) == tuple:
                name, max_trajectory_count = name
            assert type(max_trajectory_count) == int

            # Load trajectories file into memory, all at once
            with fannypack.data.TrajectoriesFile(name) as f:
                raw_trajectories = list(f)
                
            # Iterate over each trajectory
            for raw_trajectory_index, raw_trajectory in enumerate(raw_trajectories):
                if raw_trajectory_index >= max_trajectory_count:
                    break
                timesteps = len(raw_trajectory["object-state"])
                # State is just (x, y)
                state_dim = 2
                states = np.full((timesteps, state_dim), np.nan, dtype=np.float32)
                states = raw_trajectory["Cylinder0_pos"][:, :2]  # x, y

                # Pull out observations
                # This is currently consisted of:
                # > gripper_pos: end effector position
                # > gripper_sensors: F/T, contact sensors
                # > image: camera image
                observations = {}
                observations["gripper_pos"] = raw_trajectory["eef_pos"]
                if self.prop_noise != 0:
                    """Adding prop_noise!!"""
                    observations["gripper_pos"] = add_timeseries_noise(
                        [observations["gripper_pos"]], noise_level= self.prop_noise, struct_drop=False)[0]
                assert observations["gripper_pos"].shape == (timesteps, 3)

                observations["gripper_sensors"] = np.concatenate(
                    (
                        raw_trajectory["force"],
                        raw_trajectory["contact"][:, np.newaxis]
                    ),
                    axis=1,
                )
                if self.haptics_noise != 0:
                    """Adding haptics noise!!"""
                    observations["gripper_sensors"] = add_timeseries_noise(
                        [observations["gripper_sensors"]], noise_level= self.haptics_noise, struct_drop=False)[0]
                assert observations["gripper_sensors"].shape[1] == 7

                # Get image
                observations["image"] = raw_trajectory["image"].copy()
                if self.visual_noise != 0:
                    """Adding visual noise!!"""
                    observations["image"] = np.array(add_visual_noise(
                        observations["image"], noise_level=self.visual_noise))
                observations["image"] = observations["image"].reshape(timesteps, -1)
                assert observations["image"].shape == (timesteps, 32 * 32)

                # Pull out controls
                # This is currently consisted of:
                # > previous end effector position
                # > end effector position delta
                # > binary contact reading
                eef_positions = raw_trajectory["eef_pos"]
                eef_positions_shifted = np.roll(eef_positions, shift=1, axis=0)
                eef_positions_shifted[0] = eef_positions[0]

                # Force controls to be of type float32
                # NOTE: In numpy 1.20+, this can be done
                # through dtype kwarg to concatenate
                controls = np.empty((timesteps, 7), dtype=np.float32)
                np.concatenate(
                    [
                        eef_positions_shifted,
                        eef_positions - eef_positions_shifted,
                        raw_trajectory["contact"][
                            :, np.newaxis
                        ]
                    ],
                    axis=1,
                    out=controls
                )
                if self.controls_noise != 0:
                    """Adding time series noise!!"""
                    controls = add_timeseries_noise(
                        [controls], noise_level= self.controls_noise, struct_drop=False)[0]
                
                if self.multimodal_noise != 0:
                    tmp = add_timeseries_noise([observations["image"], observations["gripper_pos"],
                                            observations["gripper_sensors"], controls], noise_level= self.multimodal_noise, rand_drop=False)
                    observations["image"] = tmp[0]
                    observations["gripper_pos"] = tmp[1]
                    observations["gripper_sensors"] = tmp[2]
                    controls = tmp[3]
                # Normalize data
                if True:
                    """Normalizing data"""
                    observations["gripper_pos"] -= np.array(
                        [[0.46806443, -0.0017836, 0.88028437]],
                        dtype=np.float32,
                    )
                    observations["gripper_pos"] /= np.array(
                        [[0.02410769, 0.02341035, 0.04018243]],
                        dtype=np.float32,
                    )
                    observations["gripper_sensors"] -= np.array(
                        [
                            [
                                4.9182904e-01,
                                4.5039989e-02,
                                -3.2791464e00,
                                -3.3874984e-03,
                                1.1552566e-02,
                                -8.4817986e-04,
                                2.1303751e-01,
                            ]
                        ],
                        dtype=np.float32,
                    )
                    observations["gripper_sensors"] /= np.array(
                        [
                            [
                                1.6152629,
                                1.666905,
                                1.9186896,
                                0.14219016,
                                0.14232528,
                                0.01675198,
                                0.40950698,
                            ]
                        ],
                        dtype=np.float32,
                    )
                    states -= np.array(
                        [[0.4970164, -0.00916641]],
                        dtype=np.float32,
                    )
                    states /= np.array(
                        [[0.0572766, 0.06118315]],
                        dtype=np.float32,
                    )
                    controls -= np.array(
                        [
                            [
                                4.6594709e-01,
                                -2.5247163e-03,
                                8.8094306e-01,
                                1.2939950e-04,
                                -5.4364675e-05,
                                -6.1112235e-04,
                                2.2041667e-01,
                            ]
                        ],
                        dtype=np.float32,
                    )
                    controls /= np.array(
                        [
                            [
                                0.02239027,
                                0.02356066,
                                0.0405312,
                                0.00054858,
                                0.0005754,
                                0.00046352,
                                0.41451886,
                            ]
                        ],
                        dtype=np.float32,
                    )

                trajectories.append(
                     TrajectoryNumpy(
                        states[self.start_timestep:],
                        fannypack.utils.SliceWrapper(observations)[self.start_timestep:],
                        controls[self.start_timestep:],
                      )
                )
                # Reduce memory usage
                raw_trajectories[raw_trajectory_index] = None
                del raw_trajectory
        return trajectories
    
    def split_trajectories(self):
        """
        Helper for splitting a list of trajectories into a list of overlapping
        subsequences. For each trajectory, assuming a subsequence length of 10, this function
        includes in its output overlapping subsequences corresponding to
        timesteps...
        ```
            [0:10], [10:20], [20:30], ...
        ```
        as well as...
        ```
            [5:15], [15:25], [25:30], ...
        ```
        Args:
            trajectories (List[torchfilter.base.TrajectoryNumpy]): List of trajectories.
            subsequence_length (int): # of timesteps per subsequence.
        Returns:
            List[torchfilter.base.TrajectoryNumpy]: List of subsequences.
        """

        subsequences = []

        for traj in self.trajectories:
            # Chop up each trajectory into overlapping subsequences
            trajectory_length = len(traj.states)
            assert len(fp.utils.SliceWrapper(
                traj.observations)) == trajectory_length
            assert len(fp.utils.SliceWrapper(traj.controls)) == trajectory_length
            # We iterate over two offsets to generate overlapping subsequences
            for offset in (0, self.subsequence_length // 2):
                def split_fn(x: np.ndarray):
                    """Helper: splits arrays of shape `(T, ...)` into `(sections,
                    subsequence_length, ...)`, where `sections = orig_length //
                    subsequence_length`."""
                    # Offset our starting point
                    x = x[offset:]
                    # Make sure our array is evenly divisible
                    sections = len(x) // self.subsequence_length
                    new_length = sections * self.subsequence_length
                    x = x[:new_length]
                    # Split & return
                    return np.split(x, sections)

                for s, o, c in zip(
                    # States are always raw arrays
                    split_fn(traj.states),
                    # Observations and controls can be dictionaries, so we have to jump
                    # through some hoops
                    fp.utils.SliceWrapper(
                        fp.utils.SliceWrapper(traj.observations).map(split_fn)
                    ),
                    fp.utils.SliceWrapper(
                        fp.utils.SliceWrapper(traj.controls).map(split_fn)
                    ),
                ):
                    # Add to subsequences
                    subsequences.append(
                        [
                            torch.tensor(np.array(o['gripper_pos'])),
                            torch.tensor(np.array(o['gripper_sensors'])),
                            torch.tensor(np.array(o['image'])),
                            torch.tensor(np.array(c)),
                            torch.tensor(np.array(s))
                        ]
                    )       
        return subsequences
        
    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.subsequences[0][0].shape[0]

    def get_dim(self):
        return [self.subsequences[0][0].shape[-1], self.subsequences[0][1].shape[-1],
                self.subsequences[0][2].shape[-1], self.subsequences[0][3].shape[-1] ]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return  self.subsequences.shape[0], self.subsequences[0][-1].shape[-1]

    def __len__(self):
        return len(self.subsequences)

    def __getitem__(self, index):
        X = [
              index, self.subsequences[index][0], 
              self.subsequences[index][1], self.subsequences[index][2],
              self.subsequences[index][3]
            ]
        Y = torch.tensor(self.subsequences[index][-1])
        return X, Y 

class Enrico_Datasets(Dataset): 
    def __init__(self, dataset_path, split_type="train", 
        noise_level=0, img_noise=False, wireframe_noise=False, 
        img_dim_x = 256, img_dim_y = 128, random_seed=42, train_split=0.8, 
        val_split=0.15, test_split=0.2, normalize_image=False):
        super(Enrico_Datasets, self).__init__()#128, 256
        self.noise_level = noise_level
        self.img_noise = img_noise
        self.wireframe_noise = wireframe_noise
        self.img_dim_x = img_dim_x
        self.img_dim_y = img_dim_y
        self.split_type = split_type
        csv_file = os.path.join(dataset_path, "design_topics.csv")
        self.img_dir = os.path.join(dataset_path, "screenshots")
        self.wireframe_dir = os.path.join(dataset_path, "wireframes")
        self.hierarchy_dir = os.path.join(dataset_path, "hierarchies")
        self.n_modalities = 2
        self.patch_x = 16 
        self.patch_y = 8  
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            example_list = list(reader)
        # the wireframe files are corrupted for these files
        IGNORES = set(["50105", "50109"])
        example_list = [
            e for e in example_list if e['screen_id'] not in IGNORES]
        self.example_list = example_list
        keys = list(range(len(example_list)))
        # shuffle and create splits
        random.Random(random_seed).shuffle(keys)
        if self.split_type == "train":
            # train split is at the front
            start_index = 0
            stop_index = int(len(example_list) * train_split)
        elif self.split_type == "valid":
            # val split is in the middle
            start_index = int(len(example_list) * train_split)
            stop_index = int(len(example_list) * (train_split + val_split))
        elif self.split_type == "test":
            # test split is at the end
            start_index = int(len(example_list) * (train_split + val_split))
            stop_index = len(example_list)

        # only keep examples in the current split
        keys = keys[start_index:stop_index]
        self.keys = keys

        img_transforms = [
            transforms.Resize((img_dim_y, img_dim_x)),
            transforms.ToTensor()
        ]
        # pytorch image transforms
        self.img_transforms = transforms.Compose(img_transforms)

        # make maps
        topics = set()
        for e in example_list:
            topics.add(e['topic'])
        topics = sorted(list(topics))

        idx2Topic = {}
        topic2Idx = {}

        for i in range(len(topics)):
            idx2Topic[i] = topics[i]
            topic2Idx[topics[i]] = i

        self.idx2Topic = idx2Topic
        self.topic2Idx = topic2Idx

        UI_TYPES = ["Text", "Text Button", "Icon", "Card", "Drawer", "Web View", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab", "List Item", "Toolbar", "Bottom Navigation", "Multi-Tab",
                    "Background Image", "Image", "Video", "Input", "Number Stepper", "Checkbox", "Radio Button", "Pager Indicator", "On/Off Switch", "Modal", "Slider", "Advertisement", "Date Picker", "Map View"]

        idx2Label = {}
        label2Idx = {}

        for i in range(len(UI_TYPES)):
            idx2Label[i] = UI_TYPES[i]
            label2Idx[UI_TYPES[i]] = i

        self.idx2Label = idx2Label
        self.label2Idx = label2Idx
        self.ui_types = UI_TYPES

        self.screenImg = []
        self.screenWireframeImg = []
        for idx in range(len(self.keys)):
            if idx % 100 == 0:
              print("Finish " + str(idx) + " loadings")
            example = self.example_list[self.keys[idx]]
            screenId = example['screen_id']
            # image modality
            screenImg = Image.open(os.path.join(self.img_dir, screenId + ".jpg")).convert("RGB")
            screenImg = self.img_transforms(screenImg)
            self.screenImg.append(screenImg)
            # wireframe image modality
            screenWireframeImg = Image.open(os.path.join(self.wireframe_dir, screenId + ".jpg")).convert("RGB")
            screenWireframeImg = self.img_transforms(screenWireframeImg)
            self.screenWireframeImg.append(screenWireframeImg)

    def __len__(self):
        """Get number of samples in dataset."""
        return len(self.keys)

    def featurizeElement(self, element):
        """Convert element into tuple of (bounds, one-hot-label)."""
        bounds, label = element
        labelOneHot = [0 for _ in range(len(self.ui_types))]
        labelOneHot[self.label2Idx[label]] = 1
        return bounds, labelOneHot

    def __getitem__(self, index):
        idx = index
        screenImg = self.screenImg[idx]
        screenWireframeImg  = self.screenWireframeImg[idx]
        example = self.example_list[self.keys[idx]]
        screenLabel = self.topic2Idx[example['topic']]
        X = [index, 
            screenImg.reshape(3, self.patch_x, self.img_dim_x//self.patch_x, self.patch_y, self.img_dim_y//self.patch_y).permute(1, 3, 0, 2, 4).reshape(self.patch_x * self.patch_y, -1), 
            screenWireframeImg.reshape(3, self.patch_x, self.img_dim_x//self.patch_x, self.patch_y, self.img_dim_y//self.patch_y).permute(1, 3, 0, 2, 4).reshape(self.patch_x * self.patch_y, -1)]
        Y = screenLabel
        return X, Y 

    def get_n_modalities(self):
        return self.n_modalities
    
    def get_seq_len(self):
        return self.patch_x * self.patch_y
    
    def get_dim(self):
        return [self.img_dim_x * self.img_dim_y // (self.patch_x * self.patch_y) * 3, 
            self.img_dim_x * self.img_dim_y // (self.patch_x * self.patch_y) * 3]
    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return len(self.keys), 1

"""To be modified"""
class NTU_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train', n_patches = 7):
        super(avMNIST_Datasets, self).__init__()
        """self.transformer = transforms.Compose([
            avmnist_data.ToTensor(),
            avmnist_data.Normalize((0.1307,), (0.3081,))
        ])"""
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

        self.audio /= 255.0
        l = self.image.shape[0]
        d = int(self.image.shape[1] ** 0.5)
        da = int(self.audio.shape[1])
        self.image = self.image.reshape(l, n_patches, d//n_patches, n_patches, d//n_patches).permute(0, 1, 3, 2, 4).reshape(l, n_patches **2, -1)
        self.audio = self.audio.reshape(l, n_patches, da//n_patches, n_patches, da//n_patches).permute(0, 1, 3, 2, 4).reshape(l, n_patches **2, -1)  
        self.n_modalities = 2 # vision/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.image.shape[1]

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

class EEG2a_Datasets(Dataset):
    def __init__(self, dataset_path, split_type='train' , train_ratio = None, file_num_range_train = None, file_num_range_test = None):
        super(EEG2a_Datasets, self).__init__()
        eeg_signal = []
        labels = []
        data_class = ['data1', 'data2', 'data3', 'data4']
        file_num_range_train = file_num_range_train
        file_num_range_test = file_num_range_test
        if split_type == 'test':
            for set_num in file_num_range_test:
                file_name = set_num
                mat = scipy.io.loadmat(os.path.join(dataset_path, file_name))
                for i in range(4): # each file has 4 classes
                    data_num = mat[data_class[i]].shape[2]
                    for j in range(data_num): # each class has a specific number of data
                        eeg_signal.append(mat[data_class[i]][:,:,j])
                        labels.append(i)
            self.labels = torch.tensor(np.array(labels)).long()
            self.eeg_signal = torch.tensor(np.array(eeg_signal)).float()
        else:
            for set_num in file_num_range_train:
                file_name = set_num
                mat = scipy.io.loadmat(os.path.join(dataset_path, file_name))
                for i in range(4): # each file has 4 classes
                    data_num = mat[data_class[i]].shape[2]
                    for j in range(data_num): # each class has a specific number of data
                        eeg_signal.append(mat[data_class[i]][:,:,j])
                        labels.append(i)
            total_num = len(labels)
            train_num = int(total_num * train_ratio)
            """c = list(zip(eeg_signal, labels))
            random.shuffle(c)
            eeg_signal, labels = zip(*c)"""
            labels = torch.tensor(np.array(labels)).long()
            eeg_signal  = torch.tensor(np.array(eeg_signal)).float()
            g = torch.Generator()
            g.manual_seed(0)
            indexes = torch.randperm(labels.shape[0], generator = g)
            if split_type == 'train':
                self.labels = labels[indexes[:train_num]]
                self.eeg_signal = eeg_signal[indexes[:train_num]]
            else:
                self.labels = labels[indexes[train_num:]]
                self.eeg_signal = eeg_signal[indexes[train_num:]]

        self.n_modalities = 1
    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.eeg_signal.shape[1]

    def get_dim(self):
        return [self.eeg_signal.shape[2]]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)
  
    def __getitem__(self, index):
        X = [index, self.eeg_signal[index]]
        Y = self.labels[index]
        return X, Y 
