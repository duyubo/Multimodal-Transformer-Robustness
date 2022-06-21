import torchaudio
from transformers import BertTokenizer
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import pandas as pd

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank
    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

def extract_audio_features(audio_dir, name, model, decoder, bert_tz):
    waveform, sample_rate = torchaudio.load(audio_dir + '/' + name + '.wav')
    #print(waveform.shape, sample_rate)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    waveform = waveform.cuda()
    #print(waveform.shape)
    with torch.inference_mode():
        features, _ = model.extract_features(waveform)
    with torch.inference_mode():
      emission, _ = model(waveform)
    transcript = decoder(emission[0]).lower().split("|")
    for i in range(12):
        features[i].detach().cpu()
    emission[0].detach().cpu()
    return features, transcript

def extract_vision_features(video_dir, name, mtcnn, resnet):
    cap = cv2.VideoCapture(video_dir + '/' + name + '.mp4')
    success, img = cap.read()
    frames = []
    while success:
        img_cropped = mtcnn(img)
        if img_cropped is not None:
            with torch.inference_mode():
                img_embedding = resnet(img_cropped.unsqueeze(0).cuda())
            frames.append(img_embedding.detach().cpu())
        success, img = cap.read()
    return frames

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().cuda()
print(bundle)
decoder = GreedyCTCDecoder(labels=bundle.get_labels()).cuda()
bert_tz = BertTokenizer.from_pretrained("bert-base-cased")
mtcnn = MTCNN(device = torch.device('cuda:0'))
resnet = InceptionResnetV1(pretrained='vggface2').cuda().eval()
audio_dir = '/data/dataset/MOSEI/processed/audio'
video_dir = '/data/dataset/MOSEI/processed/video'

split_type = 'train'
labels = pd.read_excel(f"/home/yubo/data_prep/raw_data/{split_type}.xlsx")
labels = labels.rename(columns={0: 'name', 1: 'sentiment'})
print(labels) 
processed_data = []


for i in range(0, 2002):
    name = labels.iloc[i]['name']
    print(i, name)
    features, transcript = extract_audio_features(audio_dir, name, model, decoder, bert_tz)
    v_feature = extract_vision_features(video_dir, name, mtcnn, resnet)
    #print(v_feature)
    processed_data.append([name, labels.iloc[i]['sentiment'], v_feature if v_feature != [] else [], transcript, features[-1]])
    
    if i%100 == 0:
        torch.save(processed_data,f"/data/dataset/MOSEI/processed/all/processed_data_{split_type}{i}.pt")
        processed_data = []    

torch.save(processed_data,f"/data/dataset/MOSEI/processed/all/processed_data_{split_type}{i}.pt")


