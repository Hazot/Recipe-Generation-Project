import hydra
from omegaconf import DictConfig
import logging

from finetuning.finetuning import trainer_finetuning

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text

logger = logging.getLogger(__name__)


def create_dataset(params, logger):

    logger.info('Creating txt files')
    dataset2text(params=params, logger=logger)  # takes 5 minutes

    logger.info('Creating h5 file')
    tokenize(params=params, logger=logger)  # takes 45 minutes


@hydra.main(config_path="config", config_name="config_finetuning", version_base="1.3")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)
    print("==========================================================================================================")

    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
    elif params['main']['model_type'] == 'opt':
        params['main'].update(params['opt'])
    elif params['main']['model_type'] == 'llama':
        params['main'].update(params['llama'])
    elif params['main']['model_type'] == 'lora':
        params['main'].update(params['lora'])
    else:
        raise Exception("Unknown model type")

    # Check for existing dataset or creating the dataset
    create_dataset(params=params, logger=logger)

    # Train the model as fine-tuning
    trainer_finetuning(params=params, logger=logger)

    logger.info("Training successfully finished!")


if __name__ == "__main__":
    main()
