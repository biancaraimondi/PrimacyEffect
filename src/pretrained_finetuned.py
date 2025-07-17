from typing import Optional
import fire
import pandas as pd
import random
import csv
import torch
import json
import os
import warnings
import transformers

from utils import extractNum

warnings.filterwarnings("ignore")

def main(
    model_name: Optional[str]="meta-llama/Llama-2-7b-hf",
    tokenizer_name: Optional[str]="meta-llama/Llama-2-7b-hf",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = 3,
    labels_query_order: Optional[bool] = True,
    dataset: Optional[str] = "BANKING77"
):
    """
    The arguments are:
    - model_name: the name of the model to use
    - tokenizer_name: the name of the tokenizer to use
    - temperature: the temperature to use for sampling
    - top_p: the top_p to use for sampling
    - max_new_tokens: the maximum number of tokens to generate
    - labels_query_order: if True the query is asked after the labels, otherwise the question is asked before the labels
    - dataset: the dataset to use, one of 'BANKING77', 'CLINC150', 'HWU64'
    """
    print('\nModel:', model_name, 'Labels then query:', labels_query_order, 'Dataset:', dataset, end='\n')
    
    ''' Load the dataset '''
    label_ls_initial = None
    # read labels from json file
    with open('Dataset/' + dataset + '_labels.json', 'r') as f:
        label_ls_initial = json.load(f)
    # read samples from csv file
    df = pd.read_csv('Dataset/' + dataset + '.csv')
    sample_ls = df['question'].tolist()
    target_ls = df['label_position'].tolist()

    # set fixed seed
    random.seed(0)
    transformers.set_seed(0)
    
    # Load the model and tokenizer
    pipeline = transformers.pipeline(
        "text-generation", model=model_name, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
    )



    length = len(sample_ls) # Number of samples
    # Loop over the samples
    for id in range(0, length):
        # Get the text of the query of sample id
        question = str(sample_ls[id])

        label_ls = label_ls_initial.copy()

        
        shuffled_correct_ids = [0 for i in range(len(label_ls_initial))] # List of 0s, one for each position, indicating if the model correctly predicts the label in that position
        matched = 0 # Number of times the model correctly predicts the label in the correct position
        shuffling_counter = len(label_ls_initial) # Number of labels, i.e. number of positions that can assume the correct label
        target_id = target_ls[id] # Target label id in the original list of labels
        target_label = label_ls_initial[target_id] # Target label in the original list of labels

        '''
        Core code of our work: shuffle the correct label to change its position in the list of labels.
        The idea is to understand if changing the order of the correct label still allows the model to correctly predict the answer.
        '''
        for i_ in range(shuffling_counter):
            # Find target label in initial list
            target_label_index = label_ls.index(target_label)
            # Remove target label from the list
            label_ls.pop(target_label_index)
            # Insert target label in position i_
            label_ls.insert(i_, target_label)

            ''' Create the prompt '''
            # Add some context to the prompt
            prompt = 'You are given a list of labels and a question. The task is to predict the label that best matches the intent of the question. The labels are:\n'
            # Query of the sample 
            target_text = 'Target Text: ' + question + '\n'
            # Text to append to the prompt to let models generate the label id as next tokens prediction
            end = 'The Label that matches the intent of the Target Text best is: '
            # For each label in the new list of labels, add the label to the prompt
            for label_id_ in range(len(label_ls)):
                prompt = prompt + 'Label ' + str(label_id_) + ': ' + label_ls[label_id_].replace('\n', ' ').replace(':', ' ').replace('_', ' ') + '\n'
            
            # Add labels and query in the prompt in the correct order based on labels_query_order. 'end' is always placed at the end of the prompt
            if labels_query_order:
                prompt = prompt + target_text + end
            else:
                prompt = target_text + prompt + end


            result = pipeline(prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, pad_token_id=pipeline.tokenizer.eos_token_id)
            result = result[0]['generated_text']

            # From result remove the prompt leaving only the new generated text
            result = result.replace(prompt, '')
            # Extract the label id from the generated text
            label_id_generated = extractNum(result, label_ls)
            # If the label id is not present in the generated text, continue with the next sample
            if label_id_generated is None or label_id_generated >= len(label_ls):
                continue
            label_generated = label_ls[label_id_generated]

            # if the generated label is the same of the target label
            positions_to_print = ""
            if label_generated == target_label:
                matched += 1
                shuffled_correct_ids[i_] = 1
                if positions_to_print == "":
                    positions_to_print += str(i_)
                positions_to_print += ", " + str(i_)


        mean = pd.Series(shuffled_correct_ids).mean()
        median = pd.Series(shuffled_correct_ids).median()


        line_to_append = [model_name, labels_query_order, mean, median, target_label, question, matched]
        for i in shuffled_correct_ids:
            line_to_append.append(i)
        
        # CSV file path
        directory = 'results/pretrained_finetuned/' + dataset + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        csv_file = directory + model_name.replace('/', '_') + '.csv'

        # Open file in append mode and write the new line
        with open(csv_file, mode='a', newline='\n') as file:
            # if the file is empty, write the header
            if os.stat(csv_file).st_size == 0:
                header = "model_name,labels_query_order,mean,median,target_label,question,matched"
                for i in range(shuffling_counter):
                    header += ',' + str(i)
                header = header.replace('"', '')
                file.write(header + '\n')
            writer = csv.writer(file)
            writer.writerow(line_to_append)

        print("Instance ", str(id), "/", str(length), ": matched instances ", matched, "/", shuffling_counter, end='\r')


if __name__ == "__main__":
    fire.Fire(main)