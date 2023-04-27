import hydra
from omegaconf import DictConfig
import logging

from finetuning.gpt2_finetuning import trainer_gpt2
from finetuning.llama_finetuning import trainer_llama
from finetuning.lora_finetuning import trainer_lora
from finetuning.opt_finetuning import trainer_opt

from finetuning.finetuning import trainer_finetuning

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text

logger = logging.getLogger(__name__)


def create_dataset(params):
    if params['main']['create_txt_files']:
        print('Creating txt files')
        dataset2text(params=params)  # takes 5 minutes
        print('Txt files successfully created')
    else:
        print('Txt files already created')

    if params['main']['create_h5_file']:
        print('Creating h5 file')
        tokenize(params=params)  # takes 45 minutes
        print('H5 file successfully created')
    else:
        print('H5 file already created')


@hydra.main(config_path="config", config_name="config_finetuning")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)
    print("==========================================================================================================")

    if params['main']['model_type'] == 'gpt2':
        params['main'].update(params['gpt2'])
        print(params['main'])
    elif params['main']['model_type'] == 'opt':
        params['main'].update(params['opt'])
    elif params['main']['model_type'] == 'llama':
        params['main'].update(params['llama'])
    else:
        raise Exception("Unknown model type")

    # Check for existing dataset or creating the dataset
    create_dataset(params=params)

    if params['main']['model_type'] == 'gpt2':

        params['main']['create_valid'] = params['gpt2']['create_valid']

        create_dataset(params=params)

        results = trainer_gpt2(params=params)

        logger.info("Training/evaluation results %s", results)  # takes 12 hours on A100 40GB batch_size=4
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'opt':

        params['main']['create_valid'] = params['opt']['create_valid']
        create_dataset(params=params)

        results = trainer_opt(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'llama':

        params['main']['create_valid'] = params['llama']['create_valid']
        create_dataset(params=params)

        results = trainer_llama(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'lora':

        params['main']['create_valid'] = params['lora']['create_valid']
        create_dataset(params=params)

        results = trainer_lora(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    else:
        raise Exception("Unknown model type")


if __name__ == "__main__":
    main()
