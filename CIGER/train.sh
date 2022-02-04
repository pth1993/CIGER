#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=0 python train.py --drug_file "data/drug_smiles.csv" \
--drug_id_file "data/drug_id.csv" --gene_file "data/gene_feature.csv" \
--data_file "data/chemical_signature.csv" --fp_type 'neural' --label_type 'real' --loss_type 'list_wise_rankcosine' \
--batch_size 64 --max_epoch 100 --lr 0.0001 --fold 0 --model_name 'ciger' --warm_start 'False' \
--inference 'False' > output/output.txt
