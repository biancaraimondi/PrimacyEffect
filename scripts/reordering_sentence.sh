#!/usr/bin/env bash

modelvariants=('meta-llama/Llama-2-7b-chat-hf' 'meta-llama/Llama-2-7b-hf' 'meta-llama/Llama-2-13b-chat-hf' 'meta-llama/Llama-2-13b-hf' 'meta-llama/Meta-Llama-3-8B-Instruct' 'meta-llama/Meta-Llama-3-8B' 'mistralai/Mistral-7B-Instruct-v0.3' 'mistralai/Mistral-7B-v0.3')
reorder_labels=('True' 'False')
descending_order=('1' '0' '2' '3')
labels_query_order=('1' '0')
similarities=('sentence_transformer')
datasets=('BANKING77' 'CLINC150_subset' 'HWU64')

for model in "${modelvariants[@]}"
do
    for reorder in "${reorder_labels[@]}"
    do
        for descending in "${descending_order[@]}"
        do
        if ! ( [ "$descending" == '1' ] && [ "$reorder" == 'False' ] ); then
            for l_q_order in "${labels_query_order[@]}"
            do
                for similarity in "${similarities[@]}"
                do
                    for dataset in "${datasets[@]}"
                    do
                        python src/reordering_sentence.py --model_name ${model} --tokenizer_name ${model} --reorder_labels ${reorder} --descending_order ${descending} --labels_query_order ${l_q_order} --similarity ${similarity} --dataset ${dataset}
                    done
                done
            done
        fi
        done
    done
done