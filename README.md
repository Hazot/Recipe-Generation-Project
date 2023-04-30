# Recipe Generation 

This is basically a fork from https://github.com/Glorf/recipenlg.
The objective of this project is to generate high-quality, context-aware cooking recipes using the latest advancements in natural language processing. The project is targeted towards users who are interested in cooking and need help in generating recipes based on a list of ingredients.

## Environment installation

You need Anaconda or venv.

### For Anaconda

If you are running on CPU:
```
conda env create -f environment.yml
```


If you are running on GPU:
1. please download the right CUDA version for you GPU at https://pytorch.org/get-started/locally/.
2. comment the lines in environment.yml for `torch`, `torchvision` and `torchaudio`.

Afterwards, run:
```
conda env create -f environment.yml
```

or

```
pip install -r requirements.txt
```


## Finetuning a model

The default config `/config/config_finetuning.yaml` will first create the necessary datasets (requires 10GB of space and 45 minutes). Afterwards, this will start the fine-tuning of the model.

```
python main_finetuning.py
```

## Generating recipes

The default config `/config/config_generation.yaml` will create recipes from the GPT2 model. You can set the prompt and the number of times you want the generation to repeat for a specific prompt. This is also used to create a finetuned dataset of recipes which is used in the evaluation pipeline.

```
python run_generation.py
```

## Evaluating the models

After creating the files by using:

```
python run_generation.py main.evaluate=True
```

you can run the cells in `/eval/evalutation.ipynb` to get scores based on different metrics.


## References

If you use the RecipeNLG dataset and this code or the original code, use the following BibTeX entry:

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

- [@Mathieu Peloquin](https://www.github.com/mathieupelo)
- [@Joel Sageau ](https://www.github.com/JOELSAGEAU)
- [@Kevin Lessard](https://www.github.com/Hazot)