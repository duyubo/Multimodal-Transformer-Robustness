import h5py
import json
import numpy as np
import soundfile
import h5py
import json
import numpy as np
import os
from moviepy.config import get_setting
from moviepy.tools import subprocess_call
from std_folder import standard_train_fold, standard_valid_fold, standard_test_fold
import torch
from os.path import exists
import pandas as pd
import cv2

def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)
    fps = 30
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
           "-ss", "%0.2f"%t1,
           "-i", filename,
           "-t", "%0.2f"%(t2-t1),
           "-r", "%d"%fps, 
           "-map", "0", "-vcodec", "copy", "-acodec", "copy", targetname]
    
    subprocess_call(cmd)

def process(dataset, video_dir, video_dir_new, resample_dir, resample_dir_new, name:str):
        features = dataset[f"All Labels/data/{name}/features"]
        intervals = np.array(dataset[f"All Labels/data/{name}/intervals"])
        required_video_file = video_dir+'/'+(name + ".mp4")
        file_exists = exists(required_video_file) and exists(resample_dir+'/'+(name + ".wav"))
        if not file_exists:
            print(required_video_file + " does not exist")
            return []
        else:
            print("processing: ", name)
        with open(resample_dir+'/'+(name + ".wav"), "rb") as fid:
            audio, _ = soundfile.read(fid)
        labels = []  
        for i in range(len(intervals)):
            newname = f"{name}_{i:02d}"   
            """split audio file"""
            interval_i = 16000 * intervals[i]
            interval_i = np.array([interval_i[0] if interval_i[0] >= 0 else 0, interval_i[1] if interval_i[1] >= 0 else 0]).astype(int)
            segment = audio[slice(*interval_i)]
            """save audio file"""
            with open(resample_dir_new + '/' + (newname + ".wav"), "wb") as fid:
                soundfile.write(fid, segment, 16000)
            labels.append([newname, features[i][0]])

        # split videos
        cap = cv2.VideoCapture(required_video_file)
        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')    
        video_clips  = [cv2.VideoWriter(filename= f"{video_dir_new}/{name}_{i:02d}.mp4", fourcc = fourcc, fps = 30, frameSize=(w, h), isColor = True) for i in range(len(intervals))]
        intervals = intervals * 30
        f = 0
        while ret:
            flag = 0
            for i in range(len(intervals)):
                if intervals[i][0] - 1  <= f <= intervals[i][1] + 1:
                    video_clips[i].write(frame)
                elif f > intervals[i][1]:
                    flag += 1
            if flag == len(intervals):
                break
            ret, frame = cap.read()
            f += 1

        for writer in video_clips:
            writer.release()
        cap.release()
        return labels

if __name__ == "__main__": 
    dataset = h5py.File("./CMU_MOSEI_Labels.csd", "r")
    print(dataset.keys())
    video_dir_new = "/data/dataset/MOSEI/processed/video"
    resample_dir_new = "/data/dataset/MOSEI/processed/audio"
    video_dir = "/data/dataset/MOSEI/Videos/Full/Combined"
    resample_dir = "/data/dataset/MOSEI/Audio/Full/WAV_16000"

    dim_names = json.loads(dataset["All Labels/metadata/dimension names"][0])
    names = list(dataset["All Labels/data"].keys())

    labels_train = []
    labels_valid = []
    labels_test = []
    
    for train_name in standard_train_fold:
        if train_name in names:
            labels_train.extend(process(dataset, video_dir, video_dir_new, resample_dir, resample_dir_new, train_name))
        else:
            print(f"skip {train_name}! not included!")
    df = pd.DataFrame(labels_train)
    writer = pd.ExcelWriter('train.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='train', index=False)
    writer.save()

    for valid_name in standard_valid_fold:
        if valid_name in names:
            labels_valid.extend(process(dataset, video_dir, video_dir_new, resample_dir, resample_dir_new, valid_name))
        else:
            print(f"skip {valid_name}! not included!")
    df = pd.DataFrame(labels_valid)
    writer = pd.ExcelWriter('valid.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='valid', index=False)
    writer.save()
    
    for test_name in standard_test_fold:
        if test_name in names:
            labels_test.extend(process(dataset, video_dir, video_dir_new, resample_dir, resample_dir_new, test_name))
        else:
            print(f"skip {test_name}! not included!")
    df = pd.DataFrame(labels_test)
    writer = pd.ExcelWriter('test.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name='test', index=False)
    writer.save()
       
