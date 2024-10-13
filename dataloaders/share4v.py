import json
import cv2
from PIL import Image
import re 

import torch
import torch.utils.data as data
import os
import numpy as np
import random
import collections

class share4v_val_dataset(data.Dataset):
    def __init__(self, data4v_root, anno, total_len, transform, tokenizer):
        self.data4v_root = data4v_root
        self.json_name = anno
        self.image_root = data4v_root
        self.total_len = total_len
        with open(anno, 'r',encoding='utf8')as fp:
            self.json_data = json.loads(fp.read(), object_pairs_hook=collections.OrderedDict)[:self.total_len]
        fp.close()

        self.preprocess = transform
        self.tokenizer = tokenizer
        
    def __len__(self):
        return self.total_len

    def split_caption(self, text):
        texts = re.split(r'\n|</s>|[.]',text)
        subcap = []
        for text_prompt in texts:
            text_prompt = text_prompt.strip()
            if len(text_prompt) != 0:
                subcap.append(text_prompt)
        del texts
        return subcap

    def fetch_file(self, root_dir, filename):
        """Shortcut to reader's `fetch_file()`."""
        image_name = os.path.join(root_dir , filename)
        image = Image.open(image_name)
        return image

    def __getitem__(self, index):
        caption = self.json_data[index]['conversations'][1]['value']
        caption = '. '.join(self.split_caption(caption))
        caption = self.tokenizer([caption])[0]
        
        image_name = self.json_data[index]['image']

        image = self.fetch_file(self.image_root , image_name)
        image_tensor = self.preprocess(image)

        return image_tensor, caption
