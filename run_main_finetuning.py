import hydra
from omegaconf import DictConfig
import logging

from generation.gpt2_finetuning import trainer_gpt2

from generation.llama_finetuning import trainer_llama

from generation.tokenization import tokenize
from generation.dataset2text import dataset2text

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    # add if os.path not exists or params['data']['create_txt_files'], do dataset2text()
    if params['data']['create_txt_files']:
        dataset2text(params=params)  # takes 5 minutes

    # add if os.path not exists or params['data']['create_h5_file'], do tokenize()
    if params['data']['create_h5_file']:
        tokenize(params=params)  # takes 45 minutes

    if params['alg']['model_type'] == 'gpt2':
        results = trainer_gpt2(params=params)
        logger.info("Training/evaluation results %s", results)  # takes 12 hours on A100 40GB batch_size=4
        logger.info("Training successfully finished!")
    elif params['alg']['model_type'] == 'llama':
        results = trainer_llama(params=params)
        logger.info("Training/evaluation results %s", results)
        logger.info("Training successfully finished!")
    else:
        raise Exception("Unknown model type")


if __name__ == "__main__":
    main()
