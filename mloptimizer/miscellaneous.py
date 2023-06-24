import logging
import os
from logging import FileHandler
from logging import Formatter


def init_logger(filename='mloptimizer.log', log_path="."):
    """
    Initializes a logger with the given filename and log_path.
    The logger is configured to log INFO messages and above.
    :param filename: the name of the log file
    :param log_path: the path where the log file will be created
    :return: the logger object and the log file path
    :rtype: tuple
    """
    filename = os.path.join(log_path, filename)
    log_format = (
        "%(asctime)s [%(levelname)s]: %(message)s in %(pathname)s:%(lineno)d")
    log_level = logging.INFO
    custom_logger = logging.getLogger(filename)
    custom_logger.setLevel(log_level)
    custom_logger_file_handler = FileHandler(filename)
    custom_logger_file_handler.setLevel(log_level)
    custom_logger_file_handler.setFormatter(Formatter(log_format))
    custom_logger.addHandler(custom_logger_file_handler)
    custom_logger.debug("Logger configured")
    return custom_logger, filename


def create_optimization_folder(folder):
    """
    Creates a folder at the given path, if it doesn't already exist.
    If the folder already exists, prompts the user to delete it.
    :param folder: the path of the folder to be created
    :return: the path of the folder
    :rtype: str
    """
    if folder is None:
        folder = os.path.join(os.curdir, "Optimizer")
    if os.path.exists(folder):
        print("The folder {} already exists and it will be used".format(folder))
    else:
        os.mkdir(folder)
        print("The folder {} has been created.".format(folder))
    return folder
