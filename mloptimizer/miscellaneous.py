import logging
from logging import FileHandler
from logging import Formatter
import os


def init_logger(filename='mloptimizer.log', log_path="."):
    FILENAME = os.path.join(log_path, filename)
    LOG_FORMAT = (
        "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d")
    LOG_LEVEL = logging.INFO
    custom_logger = logging.getLogger(filename)
    custom_logger.setLevel(LOG_LEVEL)
    custom_logger_file_handler = FileHandler(FILENAME)
    custom_logger_file_handler.setLevel(LOG_LEVEL)
    custom_logger_file_handler.setFormatter(Formatter(LOG_FORMAT))
    custom_logger.addHandler(custom_logger_file_handler)
    custom_logger.debug("Logger configured")
    return custom_logger, FILENAME


def create_optimization_folder(folder):
    """
    Creates a folder at the given path, if it doesn't already exist.
    If the folder already exists, prompts the user to delete it.
    """
    if folder is None:
        folder = os.path.join(os.curdir, "Optimizer")
    if os.path.exists(folder):
        print("The folder already exists and it will be used")
    else:
        os.mkdir(folder)
        print("The folder has been created.")
    return folder
