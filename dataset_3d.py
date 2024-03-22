import torch
from torch.utils import data
from torchvision import transforms
import os
from os.path import dirname, join, basename, isfile

import sys
import time
import pickle
import glob
import csv
import pandas as pd
import numpy as np
import cv2
from augmentation import *
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.io import wavfile
import python_speech_features
import torchaudio

class deepfake_data(data.Dataset):
    def __init__(self,video_info,
                 mode='train',
                 transform=None):
        self.mode = mode
        self.transform = transform

        if (mode == 'test'):
            split = os.path.join(video_info,'test_split.csv')
            video_info = pd.read_csv(split, header=None)

        self.label_dict_encode = {}
        self.label_dict_decode = {}
        self.label_dict_encode['fake'] = 0
        self.label_dict_decode['0'] = 'fake'
        self.label_dict_encode['real'] = 1
        self.label_dict_decode['1'] = 'real'

        self.video_info = video_info

    def __getitem__(self, index):
        try:
            vpath, audiopath, label = self.video_info.iloc[index]
            seq = [pil_loader(os.path.join(vpath,img)) for img in sorted(os.listdir(vpath))]
            t_seq = self.transform(seq)
            t_seq = torch.stack(t_seq, 0)

            cct_batch =extract_mfcc(audiopath)
            cct = torch.tensor(cct_batch)
            vid = self.encode_label(label)
            batch = list(filter(lambda x: x is not None, t_seq))
            assert cct.size()[0] == 30
            return t_seq, cct, torch.LongTensor([vid]), audiopath
        except Exception as e:
            print("Error in extract_mfcc:", str(e))
            # return None
            random_idx = random.randint(0,self.__len__())
            return self.__getitem__(random_idx)

    def __len__(self):
        return len(self.video_info)

    def encode_label(self, label_name):
        return self.label_dict_encode[label_name]

    def decode_label(self, label_code):
        return self.label_dict_decode[label_code]

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            # return img.convert('L')
        

def extract_mfcc(audio_path, frame_length_ms=30, frame_stride_ms=1/30*1000, num_filters=30, nfft=2048):
    sample_rate, audio = wavfile.read(audio_path)
    frame_length = int(sample_rate * (frame_length_ms / 1000.0))
    frame_stride = int(sample_rate * (frame_stride_ms / 1000.0))

    mfcc_features = python_speech_features.mfcc(audio, sample_rate, 
                                                winlen=frame_length_ms / 1000, 
                                                winstep=frame_stride_ms / 1000, 
                                                numcep=num_filters, 
                                                nfilt=num_filters, 
                                                nfft=nfft, 
                                                appendEnergy=False)
    return mfcc_features[:30, :]

def get_image_list(data_root, split):
	filelist = []

	with open('filelists/{}.txt'.format(split)) as f:
		for line in f:
			line = line.strip()
			if ' ' in line: line = line.split()[0]
			filelist.append(os.path.join(data_root, line))

	return filelist
