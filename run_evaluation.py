import os
import hydra
from omegaconf import DictConfig
import logging

from eval.prepare_evaluation import generate_finetuned_recipes
from eval.evaluation import evaluate
from generation.generation import generate_recipes


@hydra.main(config_path="config", config_name="config_evaluation", version_base="1.3")
def main(params: DictConfig):
    # setup basic logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    # TODO: do this part of the pipeline (evaluation)
    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
    elif params['main']['model_type'] == 'opt':
        params['main'].update(params['opt'])
    elif params['main']['model_type'] == 'qlora':
        params['main'].update(params['qlora'])
    else:
        raise Exception("Unknown model type")

    # Set the absolute path to the model, if the model has already been trained
    params['main']['model_name_or_path'] = hydra.utils.get_original_cwd() + params['main']['model_name_or_path']

    logger.info("No generation will be done, the evaluate flag is set to false.")
    generate_finetuned_recipes(params=params, logger=logger)
    logger.info("Finetuned recipes for evaluation have been successfully generated!")

    results = evaluate(params=params, logger=logger)
    print(results)

    logger.info("Evaluation successfully finished!")


if __name__ == "__main__":
    main()
