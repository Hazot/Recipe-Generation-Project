import hydra
from omegaconf import DictConfig
import os
import logging

from finetuning.finetuning import trainer_finetuning

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_finetuning", version_base=None)
def main(params: DictConfig) -> None:

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

    # Check for existing dataset and/or create the dataset
    logger.info('Creating txt files')
    dataset2text(params=params, logger=logger)  # takes 5 minutes

    logger.info('Creating h5 file')
    tokenize(params=params, logger=logger)  # takes 45 minutes

    # Train the model as fine-tuning
    trainer_finetuning(params=params, logger=logger)

    logger.info("Training successfully finished!")

if __name__ == "__main__":
    main()
