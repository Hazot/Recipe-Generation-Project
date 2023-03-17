# Recipe Generation 


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

## Authors

- [@Mathieu Peloquin](https://www.github.com/mathieupelo)
- [@Joel Sageau ](https://www.github.com/JOELSAGEAU)
- [@Kevin Lessard](https://www.github.com/Hazot)