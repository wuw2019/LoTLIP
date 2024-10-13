import json
import cv2
from PIL import Image
import re 

import torch
import torch.utils.data as data
import os
import numpy as np
import random

# import sys
# sys.path.append('./src')

class iiw_retrieval_dataset(data.Dataset):
    def __init__(self, iiw_data_root, transform, tokenizer):
        self.data_names = ['DOCCI_Test', 'IIW-400', 'DCI_Test']
        self.data_subroot = {
            'DOCCI_Test':'docci',
            'IIW-400':'docci_aar',
            'DCI_Test': 'dci'
        }
        self.anno_root = 'dataloaders/imageinwords'
        self.image_root = iiw_data_root
        
        self.image_list, self.text_list = self.get_data()

        self.preprocess = transform
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.image_list)

    def get_data(self):
        image_list = []
        text_list = []
        for data_name in self.data_names:
            anno_file = os.path.join(self.anno_root, data_name, 'data.jsonl')
            with open(anno_file, 'r') as json_file:
                anno = list(json_file)
            json_file.close()
            for data in anno:
                data = json.loads(data)
                if 'image' in data:
                    image_name = data['image']
                elif 'image/key' in data:
                    image_name = data['image/key']
                if '.jpg' not in image_name: image_name += '.jpg'
                image_list.append(os.path.join(self.data_subroot[data_name],image_name))
                text_list.append(data['IIW'])
        return image_list, text_list

    def split_caption(self, text):
        texts = re.split(r'\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap

    def __getitem__(self, index):

        caption = self.text_list[index]
        caption = '. '.join(self.split_caption(caption))
        caption = self.tokenizer([caption])[0]

        image_name = self.image_list[index]
        image_name = os.path.join(self.image_root , image_name)
        image = Image.open(image_name)
        image_tensor = self.preprocess(image)

        return image_tensor, caption
