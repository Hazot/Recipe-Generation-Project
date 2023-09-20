import hydra
from omegaconf import DictConfig
import os
import logging

from finetuning.finetuning import trainer_finetuning
from finetuning.json_qlora_finetuning import trainer_lora

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text
from utils.txt_2_json import txt_2_json

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_finetuning", version_base="1.3")
def main(params: DictConfig) -> None:

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)
    print("==========================================================================================================")

    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
    elif params['main']['model_type'] == 'opt-125m':
        params['main'].update(params['opt-125m'])
    elif params['main']['model_type'] == 'opt-350m':
        params['main'].update(params['opt-350m'])
    elif params['main']['model_type'] == 'qlora':
        params['main'].update(params['qlora'])
    else:
        raise Exception("Unknown model type")

    # Check for existing dataset and/or create the dataset
    try:
        logger.info('Creating txt files...')
        dataset2text(params=params, logger=logger)  # takes 5 minutes

        if not params['main']['json']:
            # Check for existing h5 file and/or create the h5 file
            logger.info('Checking for existing h5 file...')
            tokenize(params=params, logger=logger)  # takes 45 minutes on (i7 7700k)
        else:
            # Create the json train and test files
            logger.info('Creating json file...')
            txt_2_json(params=params, logger=logger)  # takes 45 minutes on (i7 7700k)
    except Exception as e:
        logger.info("Dataset creation has halted because of an exception.")
        logger.exception(e)

    # Train the model as fine-tuning
    try:
        if not params['main']['json']:
            trainer_finetuning(params=params, logger=logger)
        else:
            trainer_lora(params=params)
        logger.info("Training successfully finished!")
    except Exception as e:
        print(e)
        logger.info("Training has halted because of an exception.")
        logger.exception(e)
    logger.info("End of file.")


if __name__ == "__main__":
    main()
