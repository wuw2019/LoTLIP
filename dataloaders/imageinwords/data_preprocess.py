import os
import json
import argparse
import shutil 
from tqdm import tqdm

def get_iiw_dataset(image_path, target_subimage_folder, anno_files):
    for data_name in target_subimage_folder.keys():
        target_image_folder = target_subimage_folder[data_name]
        os.makedirs(target_image_folder, exist_ok=True)

        anno_file = anno_files[data_name]
        with open(anno_file, 'r') as json_file:
            anno = list(json_file)

        for data in tqdm(anno):
            data = json.loads(data)
            if 'image' in data:
                image_name = data['image']
            elif 'image/key' in data:
                image_name = data['image/key']
            if '.jpg' not in image_name: image_name += '.jpg'
            if not os.path.exists(os.path.join(image_path[data_name], image_name)):
                shutil.copyfile(os.path.join(image_path[data_name], image_name.replace('test','train')), os.path.join(target_image_folder, image_name))
                continue
            shutil.copyfile(os.path.join(image_path[data_name], image_name), os.path.join(target_image_folder, image_name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess IIW dataset")
    parser.add_argument("--dci-root", type=str, help="path to DCI dataset")
    parser.add_argument("--docci-root", type=str, help="path to DOCCI dataset")
    parser.add_argument("--iiw-root", type=str, help="path to save IIW dataset")

    args = parser.parse_args()
    
    image_path = {
        'DCI_Test': os.path.join(args.dci_root, 'densely_captioned_images/photos'),
        'DOCCI_Test': os.path.join(args.docci_root, 'images'),
        'IIW-400': os.path.join(args.docci_root, 'images_aar'),
    }

    target_subimage_folder = {
        'DCI_Test': os.path.join(args.iiw_root,'dci'),
        'DOCCI_Test': os.path.join(args.iiw_root,'docci'),
        'IIW-400': os.path.join(args.iiw_root,'docci_aar'),
    }

    anno_files = {
        'DCI_Test': 'DCI_Test/data.jsonl',
        'DOCCI_Test': 'DOCCI_Test/data.jsonl',
        'IIW-400': 'IIW-400/data.jsonl',
    }

    get_iiw_dataset(image_path, target_subimage_folder, anno_files)