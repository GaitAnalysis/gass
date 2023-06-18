# GASS - Gait Analysis from Single camera Setup
## Introduction
This project uses a single camera to capture the gait of a person and then uses the captured video to extract the gait features. The gait features are then used to identify the person. The project is divided into two parts:
1. Gait Feature Extraction: For this part, OpenPose is used, you need to download the OpenPose weights from [here](https://www.dropbox.com/sh/7xbup2qsn7vvjxo/AABWFksdlgOMXR_r5v3RwKRYa?dl=0). The weights should be placed in the folder `model/`. You may not use OpenPose weights for commercial purposes, please refer to the [license](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
2. Classifier: For classifier, an LSTM is used, the model weights are under GPL2 license. You might download the file from [here](https://drive.google.com/file/d/1_tnbuixKJ6Caa2-LWPrj8Yc8OwYzi88t/view?usp=share_link) 

## Dependencies
  - Python 3.10 and above
  - ffmpeg-python
  - torch
  - torchvision

## Usage
### Gait Feature Extraction
To extract the gait features, check the following command:
```python3 vid2seq.py --help```
### Converting Gait Features
To convert the gait features, check the following command:
```python3 convert_infer_results.py --help```

### Train
Please check the jupyter notebook `train.ipynb` for training the model.

### Inference
main.py file has a web server code that can be used for inference. There are IP addresses referring to the web service backend, you might want to delete them. The web service is written in FastAPI and can be run using uvicorn.

## Thanks
Thanks to 
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) for providing the gait feature extraction model.
- Thanks to [openpose-pytorch](https://github.com/ruiminshen/openpose-pytorch) for providing the PyTorch implementation of OpenPose.
