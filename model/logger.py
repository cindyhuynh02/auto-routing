import os
import logging
from datetime import datetime
import errno

logger_date = datetime.now().strftime("%d.log")
logger_format = "%(asctime)s | %(levelname)s | %(message)s"
logger_folder = "/home/vietphan/Documents/code/auto-routing/log/"


def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)  # Python>3.2
    except TypeError:
        try:
            os.makedirs(path)
            os.chmod(path, 0o777)
        except OSError as exc:  # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else:
                raise


class Logger:
    def __init__(self) -> None:
        pass

    def get_logger(self, name=None, logger_type=None):
        mkdir_p(logger_folder)

        logging.basicConfig(
            level=logging.INFO,
            filename=logger_folder + logger_type + logger_date,
            filemode="a",
            format=logger_format,
        )
        logger = logging.getLogger(name)

        handler = logging.StreamHandler()
        logger.addHandler(handler)
        return logger
    
    def error(self, message):
        logging.error(message)