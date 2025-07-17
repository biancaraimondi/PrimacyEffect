import re
import torch

# Function to extract the id of the label from the generated text
def extractNum(string, labels):
    max_num = len(labels)
    match = None
    if max_num < 10:
        match = re.search(r'\b([0-9])\b', string)
    elif max_num < 100:
        match = re.search(r'\b([0-9]|[1-9][0-9])\b', string)
    elif max_num < 1000:
        match = re.search(r'\b([0-9]|[1-9][0-9]|[1-9][0-9][0-9])\b', string)
    else:
        raise ValueError("Please handle the case where the number of labels is greater than 1000 in file src/utils.py")
    
    if match is not None and int(match.group()) < max_num:
        num = int(match.group())
        return num
    else:
        """ for i, label in enumerate(labels):
            if label in string:
                return i """
        return None

def cosine_similarity(vec1, vec2):
    distance = torch.nn.functional.cosine_similarity(vec1, vec2, dim=0)
    return distance.item()

def euclidean_distance(vec1, vec2):
    distance = torch.norm(vec1 - vec2, p=2)
    return distance.item()

def manhattan_distance(vec1, vec2):
    distance = torch.norm(vec1 - vec2, p=1)
    return distance.item()
