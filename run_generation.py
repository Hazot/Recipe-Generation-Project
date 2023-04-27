import hydra
from omegaconf import DictConfig
import logging

from generation.generation import generate_recipes

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_generation")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
        print(params['main'])
    elif params['main']['model_type'] == 'opt':
        params['main'].update(params['opt'])
    elif params['main']['model_type'] == 'llama':
        params['main'].update(params['llama'])
    else:
        raise Exception("Unknown model type")

    generate_recipes(params=params)

    # logger.info(results)
    logger.info("Generation successfully finished!")


if __name__ == "__main__":
    main()
