import os
import argparse
from os.path import join
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import pandas as pd
import detect_from_video
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
metadata = pd.read_json('../dataset/train_sample_videos/metadata.json').T
model_path = '../classification/pretrainedmodels/faceforensics++_models_subset/full/xception/full_raw.p'
model = torch.load(model_path)
model = model.cuda()
def predict_model(video_fn, model, start_frame=0, end_frame=50, plot_every_x_frames = 7):
    fn = video_fn.split('.')[0]
    video_path = f'../dataset/test_videos/{video_fn}'
    output_path = './FuckInd'
    prob = detect_from_video.test_full_image_network(video_path, model, output_path, start_frame=0, end_frame=50, cuda=False)
    print(prob/50)
    vidcap = cv2.VideoCapture(f'{fn}.avi')
    success,image = vidcap.read()
    count = 0
    fig = plt.figure(figsize = (8, 8))
    i = 0
    j = 1
    while success:
        if count % plot_every_x_frames == 0:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            fig.add_subplot(2, 4, j)
            j += 1
            plt.imshow(image)
            i += 1
        success,image = vidcap.read()
        count += 1
    plt.show()
    fig.savefig(os.path.join(output_path, 'aktnlyqpah'))
predict_model('aktnlyqpah.mp4', model)

