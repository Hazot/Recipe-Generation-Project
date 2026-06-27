# Recipe Generation

Loosely based on [recipenlg](https://github.com/Glorf/recipenlg). The goal is to
generate high-quality, context-aware cooking recipes using fine-tuned language
models, taking a list of ingredients as input.

## TODOs

- Reorganize the codebase structure
- Include tokenized sequence length in the h5 file name
- Fine-tune Llama 3.1 8B with LoRA
- Implement RAG
- Clean up the evaluation pipeline and consolidate config into `config_evaluation.yaml`
- Add support for evaluating a model at a specific checkpoint

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.
Install it if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install all dependencies (creates `.venv` automatically):

```bash
uv sync
```

To run scripts, either prefix commands with `uv run` or activate the environment:

```bash
source .venv/bin/activate
```

> **CUDA / Linux training:** The default PyTorch build from PyPI is CPU-only. For
> GPU training on Linux, add the following to `pyproject.toml` and re-run `uv sync`:
>
> ```toml
> [tool.uv.sources]
> torch = [{ index = "pytorch-cu121", marker = "sys_platform == 'linux'" }]
> torchaudio = [{ index = "pytorch-cu121", marker = "sys_platform == 'linux'" }]
> torchvision = [{ index = "pytorch-cu121", marker = "sys_platform == 'linux'" }]
>
> [[tool.uv.index]]
> name = "pytorch-cu121"
> url = "https://download.pytorch.org/whl/cu121"
> ```

You will also need to download the full dataset and place it in the `data/` folder.
The finetuning pipeline preprocesses it automatically. Git LFS is not used.

## Usage

### Fine-tuning

Configured via `config/config_finetuning.yaml`. Preprocessing the dataset requires
~10 GB of disk space and takes roughly 45 minutes on an i7-7700K or 15 minutes on
a Ryzen 9 7950X3D.

```bash
uv run run_finetuning.py
```

### Generating recipes

Configured via `config/config_generation.yaml`. Set the prompt and number of
generations in the config file.

```bash
uv run run_generation.py
```

### Evaluation

Set the `evaluate` flag in `config/config_generation.yaml` to run the evaluation
pipeline. Results are saved to the `results/` folder.

```bash
uv run run_evaluation.py
```

## References

```bibtex
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

- [@Kevin Lessard](https://github.com/Hazot) — [kevin.lessard@umontreal.ca](mailto:kevin.lessard@umontreal.ca)
- [@Joel Sageau](https://github.com/JOELSAGEAU) — [joel.sageau@umontreal.ca](mailto:joel.sageau@umontreal.ca)
- [@Mathieu Peloquin](https://github.com/mathieupelo) — [mathieu.peloquin.1@umontreal.ca](mailto:mathieu.peloquin.1@umontreal.ca)
