import json
import cv2
from PIL import Image
import re 

import torch
import torch.utils.data as data
import os
import numpy as np
import random

class dci_retrieval_dataset(data.Dataset):
    def __init__(self, dci_root, transform, tokenizer):
        anno_file = os.path.join(dci_root, 'densely_captioned_images', 'splits.json')
        with open(anno_file, 'r',encoding='utf8')as fp:
            split = json.load(fp)
        data=[]
        for k, v in split.items():
            data = data+v
        fp.close()
        
        self.image_root = os.path.join(dci_root, 'densely_captioned_images', 'photos')
        self.anno_root = os.path.join(dci_root, 'densely_captioned_images', 'annotations')

        self.image_list = [] 
        self.text_list = [] 
        for data_file in data:
            with open(os.path.join(self.anno_root, data_file), 'r',encoding='utf8')as fp:
                anno = json.load(fp)
                self.image_list.append(anno['image'])
                self.text_list.append(f"{anno['short_caption']}\n{anno['extra_caption']}")
            fp.close()

        self.preprocess = transform
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.image_list)

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
