# Real-time Robustn Multimodality Learning
This is a real time application of the robust multimodality learning for sentimenst analysis on Jetson TX2 platform.
## Environment && Installation
**Environment Setup**

**Necessary File**
- folder bert-en: download pytorch_model.bin, config.json, and vocab.txt from https://huggingface.co/bert-base-uncased/tree/maininto this folder 
- model.pt: this is a pretrained model got from the dynamic training process in branch main
- face_detection.pt: this is a pretrained model dowonladed from 

## Files

## Running of Code
- Test Audio: arecord -D hw:tegrasndt186ref,0  -r 8000 -f S32_LE -d 30 -vv cap.wav
- Running Whole Pipeline: python3 analysis.py
 
