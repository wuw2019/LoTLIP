import json
import cv2
from PIL import Image
import re 

import torch
import torch.utils.data as data
import os
import numpy as np
import collections


class urban1k_retrieval_dataset(data.Dataset):
    def __init__(self, data_root, transform, tokenizer):
        self.image_root = os.path.join(data_root, 'image')
        self.text_root = os.path.join(data_root, 'caption')

    
        self.total_image = os.listdir(self.image_root)
        self.total_caption = os.listdir(self.text_root)
        
        self.preprocess = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.total_image)
    
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
        caption_name = self.total_caption[index]
        f=open(os.path.join(self.text_root, caption_name))
        caption = f.readlines()[0]
        f.close()
        caption = '. '.join(self.split_caption(caption))
        caption = self.tokenizer([caption])[0]

        image_name = caption_name[:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image, caption