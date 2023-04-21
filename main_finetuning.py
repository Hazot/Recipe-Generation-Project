import hydra
from omegaconf import DictConfig
import logging

from finetuning.gpt2_finetuning import trainer_gpt2

from finetuning.llama_finetuning import trainer_llama

from finetuning.lora_finetuning import trainer_lora

from utils.tokenization import tokenize
from utils.dataset2text import dataset2text

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    # add if os.path not exists or params['data']['create_txt_files'], do dataset2text()
    if params['main']['create_txt_files']:
        dataset2text(params=params)  # takes 5 minutes

    # add if os.path not exists or params['data']['create_h5_file'], do tokenize()
    if params['main']['create_h5_file']:
        tokenize(params=params)  # takes 45 minutes

    if params['main']['model_type'] == 'gpt2':

        params['gpt2']['model_type'] = 'gpt2'
        params['gpt2']['tokenizer_name'] = 'gpt2'
        params['gpt2']['model_name_or_path'] = 'gpt2'

        results = trainer_gpt2(params=params)

        logger.info("Training/evaluation results %s", results)  # takes 12 hours on A100 40GB batch_size=4
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'opt_125m':

        params['opt']['tokenizer_name'] = 'facebook/opt-125m'
        params['opt']['model_name_or_path'] = 'facebook/opt-125m'

        results = trainer_llama(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'llama':

        params['llama']['tokenizer_name'] = 'decapoda-research/llama-7b-hf'
        params['llama']['model_name_or_path'] = 'decapoda-research/llama-7b-hf'

        results = trainer_llama(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    elif params['main']['model_type'] == 'llama_lora':

        params['lora']['tokenizer_name'] = 'huggyllama/llama-7b'
        params['lora']['model_name_or_path'] = 'huggyllama/llama-7b'

        results = trainer_lora(params=params)

        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")

    else:
        raise Exception("Unknown model type")


if __name__ == "__main__":
    main()
