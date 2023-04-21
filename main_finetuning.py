import hydra
from omegaconf import DictConfig
import logging

from finetuning.gpt2_finetuning import trainer_gpt2
from finetuning.llama_finetuning import trainer_llama
from finetuning.lora_finetuning import trainer_lora
from finetuning.opt_finetuning import trainer_opt

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text

logger = logging.getLogger(__name__)


def create_dataset(params):
    if params['main']['create_txt_files']:
        dataset2text(params=params)  # takes 5 minutes

    if params['main']['create_h5_file']:
        tokenize(params=params)  # takes 45 minutes


@hydra.main(config_path="config", config_name="config_finetuning")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    if params['main']['model_type'] == 'gpt2':

        params['gpt2']['model_type'] = 'gpt2'
        params['gpt2']['tokenizer_name'] = 'gpt2'
        params['gpt2']['model_name_or_path'] = 'gpt2'

        params['main']['create_txt_files'] = params['gpt2']['create_txt_files']
        params['main']['create_h5_file'] = params['gpt2']['create_h5_file']
        params['main']['create_valid'] = params['gpt2']['create_valid']

        create_dataset(params=params)

        results = trainer_gpt2(params=params)

        logger.info("Training/evaluation results %s", results)  # takes 12 hours on A100 40GB batch_size=4
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'opt':

        params['opt']['tokenizer_name'] = 'facebook/opt-125m'
        params['opt']['model_name_or_path'] = 'facebook/opt-125m'

        params['main']['create_txt_files'] = params['opt']['create_txt_files']
        params['main']['create_h5_file'] = params['opt']['create_h5_file']
        params['main']['create_valid'] = params['opt']['create_valid']
        params['main']['tokenizer']

        create_dataset(params=params)

        results = trainer_opt(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'llama':

        params['llama']['tokenizer_name'] = 'decapoda-research/llama-7b-hf'
        params['llama']['model_name_or_path'] = 'decapoda-research/llama-7b-hf'

        results = trainer_llama(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'lora':

        params['lora']['tokenizer_name'] = 'huggyllama/llama-7b'
        params['lora']['model_name_or_path'] = 'huggyllama/llama-7b'

        results = trainer_lora(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    else:
        raise Exception("Unknown model type")


if __name__ == "__main__":
    main()
