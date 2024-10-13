python -m training.main \
    --share4v-retrieval path_to_sam \
    --share4v-anno dataloaders/share4v/share4v_sam_10k.json \
    --share4v_val_num 1000,10000 \
    --dci-retrieval path_to_dci_dataset \
    --iiw-retrieval path_to_iiw_dataset \
    --model ViT-B-16 \
    --pretrained 'openai'