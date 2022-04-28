import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import colorsys
import sys
from typing import Any, Dict, List, NamedTuple, Tuple
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import random
import csv
import os
from collections import Counter
import torch


class TrajectoryNumpy(NamedTuple):
    states: Any
    observations: Any
    controls: Any

##############################################################################
# Time-Series
def add_timeseries_noise(tests, noise_level=0.3, gaussian_noise=True, rand_drop=True, struct_drop=True):
    """
    Add various types of noise to timeseries data.
    
    :param noise_level: Standard deviation of gaussian noise, and drop probability in random drop and structural drop
    :param gauss_noise:  Add Gaussian noise to the time series ( default: True )
    :param rand_drop: Add randomized dropout to the time series ( default: True )
    :param struct_drop: Add randomized structural dropout to the time series ( default: True )
    """
    # robust_tests = np.array(tests)
    robust_tests = tests
    if gaussian_noise:
        robust_tests = white_noise(robust_tests, noise_level)
    if rand_drop:
        robust_tests = random_drop(robust_tests, noise_level)
    if struct_drop:
        robust_tests = structured_drop(robust_tests, noise_level)
    return robust_tests

def white_noise(data, p):
    """Add noise sampled from zero-mean Gaussian with standard deviation p at every time step.
    
    :param data: Data to process.
    :param p: Standard deviation of added Gaussian noise.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            data[i][time] += np.random.normal(0, p)
    return data

def random_drop(data, p):
    """Drop each time series entry independently with probability p.
    
    :param data: Data to process.
    :param p: Probability to drop feature.
    """
    for i in range(len(data)):
        data[i] = _random_drop_helper(data[i], p, len(np.array(data).shape))
    return data

def _random_drop_helper(data, p, level):
    """
    Helper function that implements random drop for 2-/higher-dimentional timeseris data.
    :param data: Data to process.
    :param p: Probability to drop feature.
    :param level: Dimensionality.
    """
    if level == 2:
        for i in range(len(data)):
            if np.random.random_sample() < p:
                data[i] = 0
        return data
    else:
        for i in range(len(data)):
            data[i] = _random_drop_helper(data[i], p, level - 1)
        return data

def structured_drop(data, p):
    """Drop each time series entry independently with probability p, but drop all modalities if you drop an element.
    
    :param data: Data to process.
    :param p: Probability to drop entire element of time series.
    """
    for i in range(len(data)):
        for time in range(len(data[i])):
            if np.random.random_sample() < p:
                data[i][time] = np.zeros(data[i][time].shape)
    return data

def add_visual_noise(tests, noise_level=0.3, gray=True, contrast=True, inv=True, temp=True, color=True, s_and_p=True, gaus=True, rot=True, flip=True, crop=True):
    """
    Add various types of noise to visual data.
    :param noise_level: Probability of randomly applying noise to each audio signal, and standard deviation for gaussian noise, and structured dropout probability.
    :param gray: Boolean flag denoting if grayscale should be applied as a noise type.
    :param contrast: Boolean flag denoting if lowering the contrast should be applied as a noise type. 
    :param inv: Boolean flag denoting if inverting the image should be applied as a noise type. 
    :param temp: Boolean flag denoting if randomly changing the image's color balance should be applied as a noise type.  
    :param color: Boolean flag denoting if randomly tinting the image should be applied as a noise type. 
    :param s_and_p: Boolean flag denoting if applying salt and pepper noise should be applied as a noise type. 
    :param gaus: Boolean flag denoting if applying Gaussian noise should be applied as a noise type. 
    :param rot: Boolean flag denoting if randomly rotating the image should be applied as a noise type. 
    :param flip: Boolean flag denoting if randomly flipping the image should be applied as a noise type. 
    :param crop: Boolean flag denoting if randomly cropping the image should be applied as a noise type. 
    """
    noises = []
    if gray:
        noises.append(grayscale)
    if contrast:
        noises.append(low_contrast)
    if inv:
        noises.append(inversion)
    if temp:
        noises.append(WB)
    if color:
        noises.append(colorize)
    if s_and_p:
        noises.append(salt_and_pepper)
    if gaus:
        noises.append(gaussian)
    if rot:
        noises.append(rotate)
    if flip:
        noises.append(horizontal_flip)
    if crop:
        noises.append(random_crop)
    robustness_tests = []
    for i in range(len(tests)):
        img = Image.fromarray(tests[i])
        mode = img.mode
        img = img.convert('RGB')
        for noise in noises:
            img = noise(img, noise_level)
        img = img.convert(mode)
        robustness_tests.append(np.array(img))
    return robustness_tests

def grayscale(img, p):
    """Randomly make an image grayscale.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    
    """
    if np.random.sample() <= p:
        return ImageOps.grayscale(img)
    else:
        return img

def low_contrast(img, p):
    """Randomly reduce the contract of an image.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p:
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(0.5)
    else:
        return img

def inversion(img, p):
    """Randomly invert an image.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p:
        return ImageOps.invert(img)
    else:
        return img

def WB(img, p):
    """Randomly change the white-black balance / temperature of an image.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p and img.mode == 'RGB':
        kelvin_table = {1000: (255, 56, 0), 1500: (255, 109, 0), 2000: (255, 137, 18), 2500: (255, 161, 72), 3000: (255, 180, 107), 3500: (255, 196, 137), 4000: (255, 209, 163), 4500: (255, 219, 186), 5000: (255, 228, 206), 5500: (
            255, 236, 224), 6000: (255, 243, 239), 6500: (255, 249, 253), 7000: (245, 243, 255), 7500: (235, 238, 255), 8000: (227, 233, 255), 8500: (220, 229, 255), 9000: (214, 225, 255), 9500: (208, 222, 255), 10000: (204, 219, 255)}
        temps = list(kelvin_table.keys())
        temp = temps[np.random.choice(len(temps))]
        r, g, b = kelvin_table[temp]
        matrix = (r / 255.0, 0.0, 0.0, 0.0,
                  0.0, g / 255.0, 0.0, 0.0,
                  0.0, 0.0, b / 255.0, 0.0)
        return img.convert('RGB', matrix)
    else:
        return img

def colorize(img, p):
    """Randomly tint the color of an image using an existing RGB channel.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p and img.mode == 'RGB':
        color = np.random.choice(['red', 'blue', 'green'])
        layer = Image.new('RGB', img.size, color)
        return Image.blend(img, layer, 0.3)
    else:
        return img

def salt_and_pepper(img, p):
    """Randomly add salt-and-pepper noise to the image.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p:
        img = ImageOps.grayscale(img)
        output = np.copy(np.array(img))
        nb_salt = np.ceil(p*output.size*0.5)
        coords = [np.random.randint(0, i-1, int(nb_salt))
                  for i in output.shape]
        for i in range(int(nb_salt)):
            output[coords[0][i]][coords[1][i]] = 1
        nb_pepper = np.ceil(p*output.size*0.5)
        coords = [np.random.randint(0, i-1, int(nb_pepper))
                  for i in output.shape]
        for i in range(int(nb_pepper)):
            output[coords[0][i]][coords[1][i]] = 0
        return Image.fromarray(output)
    else:
        return img

def gaussian(img, p):
    """Randomly add salt-and-pepper noise to the image.
    
    :param img: Input to add noise to.
    :param p: Probability of applying transformation.
    """
    if np.random.sample() <= p:
        dim = np.array(img).shape
        gauss = np.random.normal(0, p, (dim[0], dim[1]))
        return Image.fromarray((np.array(ImageOps.grayscale(img))+gauss).astype('uint8'))
    else:
        return img

def rotate(img, p):
    """Randomly rotate the image by a random angle in [20, 40]."""
    if np.random.sample() <= p:
        angle = np.random.random_sample()*40-20
        return img.rotate(angle, Image.BILINEAR)
    else:
        return img

def horizontal_flip(img, p):
    """Randomly flip the image horizontally."""
    if np.random.sample() <= p:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img

def random_crop(img, p):
    """Randomly apply cropping changes."""
    if np.random.sample() <= p:
        dim = np.array(img).shape
        height = dim[0]
        width = dim[1]
        cropped_height = height / 5
        cropped_width = width / 5
        init_height = np.random.random_sample() * cropped_height
        init_width = np.random.random_sample() * cropped_width
        end_height = height - cropped_height + init_height
        end_width = width - cropped_width + init_width
        return img.crop((init_width, init_height, end_width, end_height)).resize((height, width))
    else:
        return img

def periodic(img, periodic_noise_filename="periodic_noise"):
    """Randomly expose the image to periodic pattern/noise."""
    height = img.height
    width = img.width
    output = []
    for i in range(6):
        noise = Image.open("{}_{}.png".format(
            periodic_noise_filename, i+1)).convert("RGBA")
        noise = random_crop(rotate(noise.resize(
            (width*2, height*2)), np.random.random_sample()*360, 'white'), height, width)
        output.append(Image.blend(img.convert("RGBA"), noise, 0.3))
    return output

def compute_weights(train_data):
    targets = []
    class_counter = Counter()
    for i in range(len(train_data)):
        example_topic = train_data.example_list[train_data.keys[i]]['topic']
        targets.append(example_topic)
        class_counter[example_topic] += 1

    weights = []
    for t in targets:
        weights.append(1 / class_counter[t])

    weights = torch.tensor(weights, dtype=torch.float)
    sampler = WeightedRandomSampler(weights, len(targets))
    if True:
        class_counter = Counter()
        for i in range(len(train_data)):
            example_topic = train_data.example_list[train_data.keys[i]]['topic']
            class_counter[train_data.topic2Idx[example_topic]] += 1
        weights = []
        for i in range(len(train_data.topic2Idx)):
            weights.append(class_counter[i])

        weights2 = []
        for w in weights:
            weights2.append(1 / (w / sum(weights)))
        weights3 = []
        for w in weights2:
            weights3.append(w / sum(weights2))
        weights = weights3

    return weights, sampler