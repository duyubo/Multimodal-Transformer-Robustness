import numpy as np
import cv2
import time
import threading
from threading import Thread
from queue import Queue
import time
import subprocess
import sounddevice as sd
import queue
import pyaudio
import wave
import os
import torch
import torchaudio
import numpy
from src.dynamic_models2 import *
torch.manual_seed(0)
         
waveform = np.array([])
transcript = ''
sentiment_value = 0
audio_thread_ = None
transcript_thread_ = None
video_thread_ = None

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

def audio_pipeline(bundle, model, decoder, waveform, sample_rate): 
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate).cuda()
    with torch.inference_mode():
        x, lengths = model.feature_extractor(waveform, length = None)
        features = model.encoder.extract_features(x, lengths, 12)
        emission = model.aux(features[-1]) 
    transcript = decoder(emission[0]).lower().split("|")
    return features[-1], transcript

class TranscriptTranslation():
    def __init__(self, hyp_params, rate = 8000):
        # audio feature extraction
        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = self.bundle.get_model().eval().cuda()
        #audio feature to text
        self.decoder = GreedyCTCDecoder(labels=self.bundle.get_labels()).eval().cuda()
        self.sentiment_model = DynamicMULTModel(
            origin_dimensions = hyp_params.orig_d, dimension = hyp_params.dimension, 
            num_heads = hyp_params.num_heads, head_dim = hyp_params.head_dim, 
            layers_single_attn = hyp_params.layers_single_attn, layers_hybrid_attn = hyp_params.layers_cross_attn, 
            layers_self_attn = hyp_params.layers_self_attn, attn_dropout = hyp_params.attn_dropout, 
            relu_dropout = hyp_params.relu_dropout, res_dropout = hyp_params.res_dropout, 
            out_dropout = hyp_params.out_dropout, embed_dropout = hyp_params.embed_dropout, 
            attn_mask = True, output_dim = hyp_params.output_dim, modality_set = hyp_params.modality_set,
            all_steps =  hyp_params.all_steps,
            stride = 0, 
            padding = 0, 
            kernel_size = 0, 
            experiment_type = hyp_params.experiment_type
        ).cuda()
        Dynamic_Multimodal_trained = torch.load('./model.pt', map_location='cpu')
        self.sentiment_model.load_state_dict(Dynamic_Multimodal_trained.state_dict())
        self.bert_tokenizer = BertTokenizer.from_pretrained('./bert_en')

        self.open = True
        self.rate = rate
    def get_transcript(self):
        global waveform
        global transcript
        global sentiment_value
        wave_length = 0
        translate_length = 0
        pipeline_interval = 10000# conduct inference for transcript every # samples
        translate_interval = 50000# set up the transcript length
        while self.open:
            if len(waveform) > 1:
                with torch.inference_mode():
                    waveform_temp = torch.tensor(waveform).reshape(1, -1).to(torch.float32)
                    temp_length = waveform_temp.shape[1]
                    if temp_length > wave_length + pipeline_interval:
                        wave_length += pipeline_interval
                        waveform_temp = waveform_temp[:, translate_length:]
                        if temp_length > translate_length + translate_interval:
                            translate_length += translate_interval
                        print(translate_length, waveform_temp.shape)
                        features, text = audio_pipeline(self.bundle, self.model, self.decoder, waveform_temp, self.rate)
                        transcript = ''
                        for t in text:
                            transcript += t + ' ' 
                        if len(text) > 1:
                            print(transcript)
                            text = " ".join(text)
                            encoded_bert_sent = self.bert_tokenizer.encode_plus(
                                text, add_special_tokens=True, 
                                max_length = len(transcript) + 2, 
                                pad_to_max_length = True)

                            bert_sentences = torch.LongTensor([encoded_bert_sent["input_ids"]])
                            bert_sentence_types = torch.LongTensor([encoded_bert_sent["token_type_ids"]])
                            bert_sentence_att_mask = torch.LongTensor([encoded_bert_sent["attention_mask"]])
                            text = torch.stack([bert_sentences, bert_sentence_types, bert_sentence_att_mask])
                            
                            sentiment_value, _ = self.sentiment_model([text.cuda(), features.cuda(), torch.zeros(1, 2, 512).cuda()])
                            sentiment_value = sentiment_value.item()
                            print('sentiment analysis result: ', sentiment_value)
                            self.sentiment_model.set_active_modalities(active_modality = [0], active_cross = [[], [], []], active_cross_output = [['t'], [], []])
    def start(self):
        global transcript_thread_
        transcript_thread_ = threading.Thread(target=self.get_transcript)
        transcript_thread_.start()
    def stop(self):
        self.open = False


class AudioRecorder():
    "Audio class based on pyAudio and Wave"
    def __init__(self,  filename="temp_audio.wav", rate=8000, fpb=1024, channels=1, input_device = 2):
        self.open = True
        self.rate = rate
        self.frames_per_buffer = fpb
        self.channels = channels
        self.format = pyaudio.paInt16
        self.audio_filename = filename
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True, 
                                      input_device_index = input_device,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
    def record(self):
        "Audio starts being recorded"
        global waveform
        self.stream.start_stream()
        while self.open:
            data = self.stream.read(self.frames_per_buffer, exception_on_overflow = False) 
            self.audio_frames.append(data)
            waveform = np.append(waveform, numpy.fromstring(data, dtype=numpy.int16))
            if not self.open:
                break
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        waveFile = wave.open(self.audio_filename, 'wb')
        waveFile.setnchannels(self.channels)
        waveFile.setsampwidth(self.audio.get_sample_size(self.format))
        waveFile.setframerate(self.rate)
        waveFile.writeframes(b''.join(self.audio_frames))
        waveFile.close()


    def stop(self):
        "Finishes the audio recording therefore the thread too"
        if self.open:
            self.open = False
            """self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)
            waveFile.writeframes(b''.join(self.audio_frames))
            waveFile.close()"""

    def start(self):
        "Launches the audio recording function using a thread"
        global audio_thread_
        audio_thread_ = threading.Thread(target=self.record)
        audio_thread_.start()

class VideoRecorder():  
    "Video class based on openCV"
    def __init__(self, pipeline, name="temp_video.mp4", fourcc='mp4v'):
        self.open = True
        self.fourcc = fourcc         
        self.video_filename = name
        self.video_cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        width  = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
        height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
        self.frameSize = (int(width), int(height))
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        self.video_writer = cv2.VideoWriter_fourcc(*self.fourcc)
        self.video_out = cv2.VideoWriter(self.video_filename, self.video_writer, self.fps, self.frameSize)
        self.start_time = time.time()
        self.frame_counts = 1

    def record(self):
        "Video starts being recorded"
        # counter = 1
        timer_start = time.time()
        timer_current = 0
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1.3
        text_color = (255, 255, 255)
        thickness = 3
        margin = 4
        pos = (20, 500)
        bg_color = (0, 0, 0)
        peace_img = cv2.imread("peace.jpeg")
        smile_img = cv2.imread("smile.jpg")
        emo_img = cv2.imread("down.jpeg")
        y_offset = 20
        x_offset = 680
        dim = (80, 80)
        global transcript
        global sentiment_value
        while self.open:
            ret, video_frame = self.video_cap.read()
            if ret:
                self.frame_counts += 1
                text = transcript
                txt_size = cv2.getTextSize(text, font_face, scale, thickness)
                end_x = pos[0] + txt_size[0][0] + margin
                end_y = pos[1] - txt_size[0][1] - margin
                cv2.rectangle(video_frame, pos, (end_x, end_y), bg_color, -1)
                cv2.putText(video_frame, text, pos, font_face, scale, text_color, 1, 2)
                
                if sentiment_value > 0.5:
                    face = smile_img
                elif sentiment_value < -0.5:
                    face = emo_img
                else:
                    face = peace_img
                face = cv2.resize(face, dim, interpolation = cv2.INTER_AREA)  
                video_frame[y_offset:y_offset + face.shape[0], x_offset:x_offset + face.shape[1]] = face
                cv2.imshow('video_frame', video_frame)
                self.video_out.write(video_frame)
                cv2.waitKey(1)
            else:
                break
        self.video_out.release()
        self.video_cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        "Finishes the video recording therefore the thread too"
        if self.open:
            self.open=False
            """self.video_out.release()
            self.video_cap.release()
            cv2.destroyAllWindows()"""

    def start(self):
        "Launches the video recording function using a thread"
        global video_thread_
        video_thread_ = threading.Thread(target=self.record)
        video_thread_.start()

def start_AVrecording(hyp_params, filename="test"):
    global video_thread
    global audio_thread
    global transcript_thread
    video_thread = VideoRecorder(pipeline = 'nvarguscamerasrc !  video/x-raw(memory:NVMM), width=960, height=540, format=NV12, framerate=5/1 ! nvvidconv flip-method=0 ! video/x-raw, width=960, height=540, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink')
    audio_thread = AudioRecorder()
    transcript_thread = TranscriptTranslation(hyp_params)
    audio_thread.start()
    video_thread.start()
    transcript_thread.start()
    return filename

def start_video_recording(filename="test"):
    global video_thread
    video_thread = VideoRecorder()
    video_thread.start()
    return filename

def start_audio_recording(filename="test"):
    global audio_thread
    audio_thread = AudioRecorder()
    audio_thread.start()
    return filename

def stop_AVrecording(filename="test"):
    print('stop audio thread')
    audio_thread.stop() 
    print('stop transcript thread')
    transcript_thread.stop()
    print('stop video thread')
    video_thread.stop() 
    
    """
    while threading.active_count() > 1:
        time.sleep(1)
    """
    global audio_thread_
    global transcript_thread_
    global video_thread_
    audio_thread_.join()
    transcript_thread_.join()
    video_thread_.join()

    frame_counts = video_thread.frame_counts
    elapsed_time = time.time() - video_thread.start_time
    recorded_fps = frame_counts / elapsed_time
    print("total frames " + str(frame_counts))
    print("elapsed time " + str(elapsed_time))
    print("recorded fps " + str(recorded_fps))
    # Makes sure the threads have finished
    
    """
    # Merging audio and video signal
    if abs(recorded_fps - 6) >= 0.01:    # If the fps rate was higher/lower than expected, re-encode it to the expected
        print("Re-encoding")
        cmd = "ffmpeg -r " + str(recorded_fps) + " -i temp_video.mp4 -pix_fmt yuv420p -r 6 temp_video2.avi"
        subprocess.call(cmd, shell=True)
        print("Muxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
    else:
        print("Normal recording\nMuxing")
        cmd = "ffmpeg -y -ac 2 -channel_layout stereo -i temp_audio.wav -i temp_video.mp4 -pix_fmt yuv420p " + filename + ".avi"
        subprocess.call(cmd, shell=True)
        print("..")"""

class hyp_params():
        def __init__(self, ):
            self.orig_d = [768, 768, 512]
            self.dimension =100
            self.num_heads = 4
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


if __name__ == '__main__':
    hyp_params1 = hyp_params()
    start_AVrecording(hyp_params1)
    time.sleep(50)
    stop_AVrecording()






