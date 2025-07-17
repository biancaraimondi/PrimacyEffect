from typing import Optional
import fire
import pandas as pd
import random
import csv
import os
import torch
import json
import transformers

from utils import extractNum

from sentence_transformers import SentenceTransformer

def main(
    model_name: Optional[str]="meta-llama/Llama-2-7b-hf",
    tokenizer_name: Optional[str]="meta-llama/Llama-2-7b-hf",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    max_new_tokens: Optional[int] = 3,
    reorder_labels: Optional[bool] = True,
    descending_order: Optional[bool] = True,
    labels_query_order: Optional[bool] = True,
    similarity: Optional[str] = "sentence_transformer",
    dataset: Optional[str] = "BANKING77"
):
    """
    The arguments are:
    - model_name: the name of the hf model to use
    - tokenizer_name: the name of the hf tokenizer to use
    - temperature: the temperature to use for sampling
    - top_p: the top_p to use for sampling
    - max_new_tokens: the maximum number of tokens to generate
    - reorder_labels: whether to reorder the labels based on the similarity with the question
    - descending_order:
        if 0 the labels are reordered in ascending order,
        if 1 the labels are reordered in descending order,
        if 2 the labels are duplicated and reordered in both ascending and descending order
    - labels_query_order: if True the query is asked after the labels, otherwise the question is asked before the labels
    - similarity: the similarity metric to use, one of 'sentence_transformer'
    - dataset: the dataset to use, one of 'BANKING77', 'CLINC150', 'HWU64'
    """
    print('\nModel:', model_name, 'Reordering:', reorder_labels, 'Descending Order:', descending_order, 'Labels then query:', labels_query_order, 'Similarity:', similarity, 'Dataset:', dataset, end='\n')

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

    if reorder_labels:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        label_embeds = [model.encode(label) for label in label_ls_initial]

    matched = 0 # Count the number of matched samples
    length = len(sample_ls) # Number of samples
    lines = [] # List of lines to append to the csv file
    # Loop over the samples
    for id in range(0, length):
        # Get the text of the query of sample id
        question = str(sample_ls[id])
        # Dictionary that contains, for each reordered label, the similarity with the question. If reorder_labels is False, this will contains labels in the original order
        sorted_similarities = None
        '''
        Core code of our work: reorder the labels based on the similarity with the question
        The idea is to calculate the cosine similarity between the embeddings of the question and the embeddings of the labels.
        Then, we sort the labels based on the mean similarity between the question and the label.
        The similarity between the question and the label is calculated as the mean similarity between each token in the question and each token in the label.
        '''
        if reorder_labels:
            question_embeds = model.encode(question)

            # Dictionary that contains, for each non-reordered label, the similarity with the question
            similarities = {}
            for i, l_embed in enumerate(label_embeds):
                # Compute the mean of the similarities of each label_token-question_token pair
                similarities[label_ls_initial[i]] = model.similarity(question_embeds, l_embed)
            # Sort similarities in descending order
            if descending_order == 0 or descending_order == 1:
                sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=descending_order))
                labels = list(sorted_similarities.keys())
            elif descending_order == 2:
                sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
                # Duplicate the labels in ascending order and append them to the list of labels in descending order
                labels = list(sorted_similarities.keys())
                labels2 = labels[::-1]
                labels.extend(labels2)
            elif descending_order == 3:
                # Order the labels based on the model's bias
                pretrained_finetuned_path = 'results/pretrained_finetuned/' + dataset + '/' + model_name.replace("/", "_") + '.csv'
                if os.path.exists(pretrained_finetuned_path):
                    df = pd.read_csv(pretrained_finetuned_path)
                    count = [0] * (len(df.columns)-7)
                    for i in range(0, len(df.columns)-7):
                        count[i] = df[str(i)].sum()/len(df)
                    sorted_similarities = dict(sorted(similarities.items(), key=lambda item: item[1], reverse=True))
                    labels = list(sorted_similarities.keys())
                    # given the count list, sort the labels based on the count values
                    labels = [x for _, x in sorted(zip(count, labels), reverse=True)]
                else:
                    raise ValueError("The sorting technique can't be used because pretrained_finetuned file doesn't exist in :" + pretrained_finetuned_path)
            else:
                raise ValueError("Descending order must be 0, 1 or 2, but got " + descending_order)
        else:
            labels = label_ls_initial

        ''' Create the prompt '''
        # Query of the sample 
        target_text = 'Target Text: ' + question + '\n'
        # Text to append to the prompt to let models generate the label id as next tokens prediction
        end = 'The Label that matches the intent of the Target Text best is: '
        prompt = ''
        z = 0
        for label in labels:
            prompt = prompt + 'Label ' + str(z) + ': ' + label.replace('\n', ' ').replace(':', ' ').replace('_', ' ') + '\n'
            z += 1

        # Add labels and query in the prompt in the correct order based on labels_query_order. 'end' is always placed at the end of the prompt
        if labels_query_order:
            input_prompt = prompt + target_text + end
        else:
            input_prompt = target_text + prompt + end
        
        
        target_label_id = None
        label_id_generated = None
        target_label = None
        label_generated = None


        result = pipeline(input_prompt, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, pad_token_id=pipeline.tokenizer.eos_token_id)
        result = result[0]['generated_text']

        # From result remove the prompt leaving only the new generated text
        result = result.replace(input_prompt, '')

        # Extract the label id from the generated text
        label_id_generated = extractNum(result, labels)
        # If the label id is not present in the generated text, continue with the next sample
        if label_id_generated is not None:
            # From sorted_similarities dictionary, get the label corresponding to the generated label id
            label_generated = labels[label_id_generated]
            
            # Get the target id of the label in the original list of labels
            target_id = target_ls[id]
            # Get the target label from the original list of labels
            target_label = label_ls_initial[target_id]

            # Check if the generated label is the target label, i.e. if the model correctly predicted the target label
            if label_generated == target_label:
                matched += 1

            # Get the id of the target label in the sorted list of labels
            for i, label in enumerate(labels):
                if label == target_label:
                    target_label_id = i
                    break

        print("Number of matches at instance ", id, ": ", matched, ". Target label inserted at position: ", target_label_id, " and generated label is at position: ", str(label_id_generated), end='\r')

        # For each sample, save the result in a line of the csv file
        line_to_append = [model_name, reorder_labels, descending_order, labels_query_order, question, target_label, label_generated, target_label_id, label_id_generated, label_generated == target_label if label_id_generated is not None else None]
        lines.append(line_to_append)

    # CSV file path
    directory = 'results/reordering/' + dataset + '/' + similarity + '/'
    csv_file = directory + model_name.replace('/', '_') + '.csv'
    # Create the directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Open file in append mode and write the lines
    with open(csv_file, mode='a', newline='\n') as file:
        writer = csv.writer(file)
        # if empty file add the header
        if os.stat(csv_file).st_size == 0:
            writer = csv.writer(file)
            writer.writerow(['model_name', 'reorder_labels', 'descending_order', 'labels_query_order', 'question', 'target_label', 'label_generated', 'target_label_id', 'label_id_generated', 'matched'])
        for line in lines:
            writer.writerow(line)

    accuracy = matched / length

    # Save the aggregated results of the model to a file
    with open(directory + 'results_reordering.csv', 'a', newline='') as file:
        # if empty file add the header
        if os.stat(directory + 'results_reordering.csv').st_size == 0:
            writer = csv.writer(file)
            writer.writerow(['model_name', 'reorder_labels', 'descending_order', 'labels_query_order', 'matched', 'accuracy'])
        writer = csv.writer(file)
        writer.writerow([model_name, reorder_labels, descending_order, labels_query_order, matched, accuracy])


if __name__ == "__main__":
    fire.Fire(main)