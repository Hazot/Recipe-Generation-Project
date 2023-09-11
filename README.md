# Recipe Generation 

This is basically a fork from https://github.com/Glorf/recipenlg. Please refer to the original repo for more details and to use as reference.

The objective of this project is to generate high-quality, context-aware cooking recipes by exploring and using the latest advancements in natural language processing. The project is targeted towards users who are interested in cooking and want help in generating recipes based on a list of ingredients as an input.

## Environment installation

You need Miniconda or Virtualenv to run this project. We recommend using Miniconda. We are using Python 3.9.

### For Anaconda

First install PyTorch 2.0.1 for your device: https://pytorch.org/get-started/locally/. We recommend using CUDA 11.7.

Afterwards, you can install the rest of the librairies.
```
pip install -r requirements.txt
```

To run QLoRa, please install these:

```
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
pip install -U git+https://github.com/huggingface/accelerate.git
```

We also need to download the full dataset that you will find here: [Full dataset](/guides/content/editing-an-existing-page). The code will preprocess this data during the finetuning phase.


## Finetuning a model

The default config `/config/config_finetuning.yaml` will first create the necessary datasets (requires 10GB of space and 45 minutes) for the GPT2 (or specified) model. Afterwards, this will start the fine-tuning of the specified model in the config (gpt2, opt, llama).

```
python main_finetuning.py
```

## Generating recipes

The default config `/config/config_generation.yaml` will create recipes from the GPT2 (or specified) model. You can set the prompt and the number of times you want the generation to repeat for a specific prompt. This is also used to create a fine-tuned dataset of recipes which is used in the evaluation pipeline.

```
python run_generation.py
```

## Evaluating the models

The default config `/config/config_generation.yaml` has a `evaluate` flag which will evaluate the model on the test set. This will generate recipes and evaluate them using various metrics. The results will be saved in the `results` folder.

## References

If you use the RecipeNLG dataset and this code or the original code, use the following BibTeX entry since the work is closely related to the following paper:

```
@inproceedings{bien-etal-2020-recipenlg,
    title = "{R}ecipe{NLG}: A Cooking Recipes Dataset for Semi-Structured Text Generation",
    author = "Bie{\'n}, Micha{\l}  and
      Gilski, Micha{\l}  and
      Maciejewska, Martyna  and
      Taisner, Wojciech  and
      Wisniewski, Dawid  and
      Lawrynowicz, Agnieszka",
    booktitle = "Proceedings of the 13th International Conference on Natural Language Generation",
    month = dec,
    year = "2020",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.inlg-1.4",
    pages = "22--28",
}
```

## Authors

- [@Kevin Lessard](https://www.github.com/Hazot) ([kevin.lessard@umontreal.ca](kevin.lessard@umontreal.ca))
- [@Joel Sageau ](https://www.github.com/JOELSAGEAU) ([joel.sageau@umontreal.ca](joel.sageau@umontreal.ca))
- [@Mathieu Peloquin](https://www.github.com/mathieupelo) ([mathieu.peloquin.1@umontreal.ca](mathieu.peloquin.1@umontreal.ca))
