import logging
import os
from logging import FileHandler
from logging import Formatter


def init_logger(filename='mloptimizer.log', log_path="."):
    """
    Initializes a logger and returns it.

    Parameters
    ----------
    filename : str, optional
        The name of the log file. The default is 'mloptimizer.log'.
    log_path : str, optional
        The path of the log file. The default is ".".

    Returns
    -------
    custom_logger : logging.Logger
        The logger.
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
    Creates a folder to save the results of the optimization.

    Parameters
    ----------
    folder : str
        The path of the folder to create.

    Returns
    -------
    folder : str
        The path of the folder created.
    """
    if os.path.exists(folder):
        logging.warning("The folder {} already exists and it will be used".format(folder))
    elif os.makedirs(folder, exist_ok=True):
        logging.info("The folder {} has been created.".format(folder))
    else:
        logging.error("The folder {} could not be created.".format(folder))
    return folder
