import hydra
from omegaconf import DictConfig
import logging

from generation.gpt2_generation import generate_recipes_gpt2
from generation.opt_generation import generate_recipes_opt

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_generation")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    if params['main']['model_type'] == 'gpt2':

        results = generate_recipes_gpt2(params=params)

        # logger.info(results)
        logger.info("Generation successfully finished!")

    elif params['main']['model_type'] == 'opt':

        results = generate_recipes_opt(params=params)

        # logger.info(results)
        logger.info("Generation successfully finished!")

    elif params['main']['model_type'] == 'llama':

        results = generate_recipes_gpt2(params=params)

        # logger.info(results)
        logger.info("Generation successfully finished!")

    else:
        raise Exception("Unknown model type")


if __name__ == "__main__":
    main()
