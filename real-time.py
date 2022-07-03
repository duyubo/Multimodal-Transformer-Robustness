import torch
import numpy as np
import cv2
from transformers import BertTokenizer
from facenet_pytorch import MTCNN, InceptionResnetV1
import time
from torch.profiler import profile, record_function, ProfilerActivity
import torch
from src.dynamic_models2 import *
torch.manual_seed(0)
from torchsummary import summary
import torchaudio


def face_detection(img, face_detection_model):
    img_cropped = face_detection_model(img)
    return img_cropped
def face_feature_extraction(face_img, feature_extraction_model):
    face_embedding = feature_extraction_model(face_img.unsqueeze(0))
    return face_embedding

def face_pipeline(face_detection_model, feature_extraction_model, video_path):
    cap = cv2.VideoCapture(video_path)
    success, img = cap.read()
    face_features = []
    counter = 0
    timer1 = 0
    timer2 = 0
    while success:
        if counter%10 == 0:
            #img = cv2.resize(img, (480, 854))
            start = time.time()
            face_img = face_detection(img = img, face_detection_model = face_detection_model)
            end = time.time()
            timer1 += end - start
            if face_img is not None:
                start = time.time()
                face_embedding = face_feature_extraction(face_img = face_img.cuda(), feature_extraction_model = feature_extraction_model)
                end = time.time()
                timer2 += end - start
                face_features.append(face_embedding)
        success, img = cap.read()  
        counter += 1
    print('mtcnn time: ', timer1, 'feature extraction time: ', timer2)
    if face_features != []:
        face_features = torch.stack(face_features, dim = 0).permute(1, 0, 2)
    return face_features

def audio_pipeline(bundle, model, decoder, audio_path, sample_rate): 
    waveform, sample_rate = torchaudio.load(audio_path)
    start = time.time()
    print(waveform.shape)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate).cuda()
    with torch.inference_mode():
        x, lengths = model.feature_extractor(waveform, length = None)
        features = model.encoder.extract_features(x, lengths, 12)
        end = time.time()
        print('wavw2vec time: ', end - start)
        start = time.time()
        emission = model.aux(features[-1]) 
    transcript = decoder(emission[0]).lower().split("|")
    end = time.time()
    print('ctcdecoder time: ', end - start)
    return features[-1], transcript

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

class hyp_params():
        def __init__(self, ):
            self.orig_d = [768, 768, 512]
            self.dimension = 200
            self.num_heads = 8
            self.head_dim = 25
            self.layers_single_attn = 3
            self.layers_cross_attn = 4
            self.layers_self_attn = 2
            self.attn_dropout = [0.1, 0.1, 0, 0]
            self.relu_dropout = 0.1
            self.res_dropout = 0.3
            self.out_dropout = 0.1
            self.embed_dropout = 0.3
            self.output_dim = 1
            self.modality_set = ['t', 'a', 'v']
            self.all_steps = False
            self.experiment_type = 'test_single'

def Squential_Pipeline(video_path, audio_path, dynamic_model_path, hyp_params):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("!!!!!!!!!!!loading models!!!!!!!!!!!!!!!!")
    # audio feature extraction
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval().cuda()
    #audio feature to text
    decoder = GreedyCTCDecoder(labels=bundle.get_labels()).eval().cuda()
    #face detection
    mtcnn = MTCNN(select_largest=True, device = 'cuda').eval() 
    #face feature extraction
    resnet = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    #Bert Tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('./bert_en')
    #dynamic multimodal 
    dynamic_model = DynamicMULTModel(
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
        ).cuda()
    

    # feature extractions
    print('Extracting face embeddings!!!')
    start = time.time()
    face_features = face_pipeline(face_detection_model = mtcnn, feature_extraction_model = resnet, video_path = video_path)
    end = time.time()
    print('time for face preprocessing:', end - start)
    print('Extracting audio features!!!!!')
    start = time.time()
    audio_features, transcript = audio_pipeline(bundle = bundle, model = model, sample_rate = 16000, decoder = decoder, audio_path = audio_path)
    end = time.time()
    print('time for audio processing:', end - start)
    print(transcript)

    # get word embedding
    start = time.time()
    text = " ".join(transcript)
    encoded_bert_sent = bert_tokenizer.encode_plus(
            text, add_special_tokens=True, 
            max_length = len(transcript) + 2, 
            pad_to_max_length = True)

    bert_sentences = torch.LongTensor([encoded_bert_sent["input_ids"]])
    bert_sentence_types = torch.LongTensor([encoded_bert_sent["token_type_ids"]])
    bert_sentence_att_mask = torch.LongTensor([encoded_bert_sent["attention_mask"]])
    text = torch.stack([bert_sentences, bert_sentence_types, bert_sentence_att_mask])
    end = time.time()
    print('Bert Tokenizer time: ', end - start)
    """
    # load from pretrained model
    Dynamic_Multimodal_trained = torch.load(dynamic_model_path, map_location='cpu')
    model.load_state_dict(Dynamic_Multimodal_trained.state_dict())
    """
    #predict sentiment
    start = time.time()
    face_features = torch.zeros(1, 10, 512).cuda()
    text = text.cuda()
    with torch.no_grad():
        Sentiment, _ = dynamic_model([text, audio_features, face_features])
    print(Sentiment)
    end = time.time()
    print('time for multimodality learning: ', end-start)

hyp_params1 = hyp_params()
Squential_Pipeline(video_path = './_0efYOjQYRc_00.mp4', 
                   audio_path = './cap.wav', 
                   dynamic_model_path = 'model.pt',
                   hyp_params = hyp_params1)

'''        
hyp_params1 = hyp_params()
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("Squential_Pipeline"):
        Squential_Pipeline(video_path = '/data/dataset/MOSEI/processed/video/_0efYOjQYRc_00.mp4', 
                   audio_path = '/data/dataset/MOSEI/processed/audio/_0efYOjQYRc_00.wav', 
                   dynamic_model_path = 'model.pt',
                   hyp_params = hyp_params1)

print(prof.key_averages().table())
'''


