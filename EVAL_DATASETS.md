# Data Preparation for Long Text-Image Retrieval


### Datasets list:
- [ShareGPT4v](#share4v)
- [DCI](#DCI)
- [IIW](#IIW)

### <span id ='share4v'> ShareGPT4v dataset

```
$path_to_SA-1B_dataset/
|–– sa_000000/
|–––– images/
|–––––– sa_1.jpg
|–––––– sa_2.jpg
|–––––– ...
|–– sa_000001/
|–– ...
```

Step 1. Download tar files from [SA-1B](https://huggingface.co/datasets/sailvideo/SA-1B) to path_to_SA-1B_dataset/

Step 2. Unzip all tar files

For the annotations, we have resaved the top 10k samples from [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main) in dataloaders/share4v/share4v_sam_10k.json


### <span id ='dci'> DCI dataset

```
$path_to_dci_dataset/
|–– densely_captioned_images/
|–––– annotations/
|–––– complete/
|–––– photos/
|–––– splits.json
```

**Download data following [DCI](https://github.com/facebookresearch/DCI)**:

Step 1. Download [dci.tar.gz](https://dl.fbaipublicfiles.com/densely_captioned_images/dci.tar.gz) and unzip the file in path_to_dci_dataset/densely_captioned_images 

Step 2. Download the archive sa_000138.tar and extract the images in the photos folder.



### <span id ='iiw'> IIW dataset

```
$path_to_iiw_dataset/
|–– dci/
|–– docci/
|–– docci_aar/
```

**Download human annotated data following [IIW](https://github.com/google/imageinwords/tree/main/datasets), including IIW-400, DCI-Test, DOCCI-Test**:

Step 1: Download [DCI](https://github.com/facebookresearch/DCI) to path_to_dci_dataset

Step 2: Download DOCCI images and AAR images from [DOCCI](https://google.github.io/docci/#downloads) dataset. Unzip the files to path_to_docci_dataset/images and path_to_docci_dataset/images_aar, respectively.

Step 3: 

```
cd src/dataloaders/imageinwords

python data_preprocess.py --dci-root path_to_dci_dataset --docci-root path_to_docci_dataset --iiw-root path_to_iiw_dataset
```
