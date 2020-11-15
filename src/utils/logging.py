import logging


def config_logger():
    # Setup logging
    return logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p',
        level=logging.DEBUG
        # filename='log.txt' # <- optionally save logs to a txt file
    )