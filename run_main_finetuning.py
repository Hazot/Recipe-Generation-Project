import hydra
from omegaconf import DictConfig
import logging

from generation.gpt2_finetuning import trainer
from generation.tokenization import tokenize
from generation.dataset2text import dataset2text

logger = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config_gpt2")
def main(params: DictConfig):

    # setup basic logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    logger.info("Training/evaluation parameters %s", params)

    # add if os.path not exists or params['data']['create_txt_files'], do dataset2text()
    if params['data']['create_txt_files']:
        dataset2text()  # takes 5 minutes

    # add if os.path not exists or params['data']['create_h5_file'], do tokenize()
    if params['data']['create_h5_file']:
        tokenize()  # takes 45 minutes

    results = trainer(params=params)

    logger.info("Training/evaluation results %s", results)
    logger.info("Training successfully finished!")


if __name__ == "__main__":
    main()
