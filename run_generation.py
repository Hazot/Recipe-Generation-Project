import hydra
from omegaconf import DictConfig
import logging

from generation.generation import generate_recipes

@hydra.main(config_path="config", config_name="config_generation")
def main(params: DictConfig):
    # setup basic logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
    elif params['main']['model_type'] == 'opt':
        params['main'].update(params['opt'])
    elif params['main']['model_type'] == 'llama':
        params['main'].update(params['llama'])
    else:
        raise Exception("Unknown model type")

    logger.info("Generating recipes with the following model:", str(params['main']['model_type']))
    logger.info("The generation will be done with the following parameters:", params['main'])
    generate_recipes(params=params, logger=logger)

    logger.info("Generation successfully finished!")


if __name__ == "__main__":
    main()
