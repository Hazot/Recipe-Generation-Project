import hydra
import json
from omegaconf import DictConfig
import logging

from eval.prepare_evaluation import generate_finetuned_recipes
from generation.generation import generate_recipes


@hydra.main(config_path="config", config_name="config_generation", version_base="1.3")
def main(params: DictConfig):
    # setup basic logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if params["main"]["model_type"] == "gpt2":
        params["main"].update(params["gpt2"])
    elif params["main"]["model_type"] == "opt-125m":
        params["main"].update(params["opt-125m"])
    elif params["main"]["model_type"] == "opt-350m":
        params["main"].update(params["opt-350m"])
    elif params["main"]["model_type"] == "qlora":
        params["main"].update(params["qlora"])
    else:
        raise Exception("Unknown model type")

    # Set the absolute path to the model, if the model has already been trained
    params["main"]["model_name_or_path"] = (
        hydra.utils.get_original_cwd() + params["main"]["model_name_or_path"]
    )

    if not params["main"]["evaluate"]:
        logger.info("Generating recipes to print to the cli.")
        logger.info(
            f"Generating recipes with the following model: {params['main']['model_type']}"
        )
        params_string = json.dumps(dict(params["main"]), indent=4)
        logger.info(
            f"The generation will be done with the following parameters: {params_string}"
        )
        generate_recipes(params=params, logger=logger)
    else:
        logger.info("No generation will be done, the evaluate flag is set to false.")
        generate_finetuned_recipes(params=params, logger=logger)
        logger.info(
            "Finetuned recipes for evaluation have been successfully generated!"
        )

    logger.info("Generation successfully finished!")


if __name__ == "__main__":
    main()
