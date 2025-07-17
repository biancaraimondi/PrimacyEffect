#!/usr/bin/env bash

modelvariants=('meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Meta-Llama-3-8B-Instruct' 'meta-llama/Meta-Llama-3-8B' 'mistralai/Mistral-7B-Instruct-v0.3' 'mistralai/Mistral-7B-v0.3')
datasets=('BANKING77' 'CLINC150' 'HWU64')

for model in "${modelvariants[@]}"
do
    for dataset in "${datasets[@]}"
    do
        python src/pretrained_finetuned.py --model_name ${model} --dataset ${dataset}
    done
done