# Real-time Robustn Multimodality Learning
This is a real time application of the robust multimodality learning for sentimenst analysis on Jetson TX2 platform.
## Environment && Installation
**Environment Setup**
- pytorch and torch_vision for jetson tx2: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048. The version is decided by the jetpack version. This implementation uses jetpack 4.6.
- mtcnn: https://towardsdatascience.com/face-detection-using-mtcnn-a-guide-for-face-extraction-with-a-focus-on-speed-c6d59f82d49
- ninja: pip3 install ninja
- torch_audio: 
  - upgrade cmake: https://graspingtech.com/upgrade-cmake/ and https://github.com/jetsonhacks/buildLibrealsense2TX/issues/13
  - install: git clone -b ${TORCHAUDIO_VERSION} https://github.com/pytorch/audio torchaudio && cd torchaudio && python3 setup.py install && cd ../ && \

**Necessary File**
- folder bert-en: download pytorch_model.bin, config.json, and vocab.txt from https://huggingface.co/bert-base-uncased/tree/main into this folder 
- model.pt: this is a pretrained model got from the dynamic training process in branch main
- face_detection.pt: this is a pretrained model dowonladed from https://github.com/Furkan-Gulsen/face-classification

**Hardware Adjustment**
- Audio device configuration: https://blog.csdn.net/lee353086/article/details/121675173

## Files
- modules\: same as main branch
- src\: same as main branch
- analysis.py: realtime pipeline for sentiment analysis with image, audio and transcript (translated from audio)

## Running of Code
- Test Audio: arecord -D hw:tegrasndt186ref,0  -r 8000 -f S32_LE -d 30 -vv cap.wav
- Running Whole Pipeline: python3 analysis.py
 
