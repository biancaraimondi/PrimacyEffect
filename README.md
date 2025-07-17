# Exploiting Primacy Effect to Improve Large Language Models
This repository is the code base for the paper: [Exploiting Primacy Effect to Improve Large Language Models](TODO add link to arXiv).

## Getting started
Clone the repository and install the requirements:
```bash
git clone https://github.com/biancaraimondi/PrimacyEffect.git
cd PrimacyEffect
pip install -r requirements.txt
```

## Generate the data
The preprocessed datasets can be found in the `Dataset` folder.

If you want to do your own preprocessing, use the `src/dataset_preprocessing.ipynb` file.

For the HWU dataset, you can find the relative base version [here](https://github.com/xliuhw/NLU-Evaluation-Data).
For the BANKING dataset, you can find the relative base version [here](https://github.com/PolyAI-LDN/task-specific-datasets).
For the CLINC dataset, you can find the relative base version [here](https://github.com/clinc/oos-eval).

## Pre-trained vs Fine-tuned
Use the `scripts/pretrained_finetuned.sh` script to test the models on the datasets comparing pre-trained and fine-tuned versions.
Insert the following informations in the script variables:
- `modelvariants`: the list of models you want to test.
- `datasets`: the list of datasets you want to use.

```bash
./scripts/pretrained_finetuned.sh
```

The results will be saved in the `results/pretrained_finetuned` folder.

## Reordering
Use the `scripts/reordering.sh` script to test our technique over the models on the same datasets used in the latter section to compare accuracy.
Insert the following informations in the script variables:
- `modelvariants`: the list of models you want to test.
- `reorder_labels`: indicate `True` if you want to reorder the labels using our technique, `False` otherwise.
- `descending_order`: indicate `1` if you want to reorder the labels in descending order, `0` if you want to reorder the labels in ascending order.
- `labels_query_order`: indicate `1` if you want first the labels and then the question in the prompt, `0` if you want first the question and then the labels in the prompt.
- `similarities`: choose between `cosine_similarity`, `euclidean_distance`, and `manhattan_distance` as the similarity metric you want to use for reordering the labels.
- `datasets`: the list of datasets you want to use.

```bash
./scripts/reordering.sh
```

The results will be saved in the `results/reordering` folder.

## Results analysis
Use the `src/plot_pretrained_finetuned.ipynb` file to create the plots for the `Pre-trained vs Fine-tuned` work.
Use the `src/plot_reordering.ipynb` file to create the plots for the `Reordering` work.

## License
This repository is released under the [Apache 2.0](https://github.com/biancaraimondi/PrimacyEffect/blob/main/LICENSE.txt) license.

## Citation
If you use this code in your research, please cite the following paper:
```
@misc{TODO,
      title={TODO}, 
      author={TODO},
      year={TODO}
}
```