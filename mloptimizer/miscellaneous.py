import logging
import sys
import os

LOG_PATH = "/opt/project/logs/"


def init_logger(filename='optimization.log'):
    logging_params = {
        # 'stream': sys.stdout,
        'level': logging.DEBUG,
        'format': '%(asctime)s:%(levelname)s:[L%(lineno)d]:%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'filename': os.path.join(LOG_PATH, filename)
    }
    print(os.listdir("../"))
    print(os.path.abspath(__file__))
    logging.basicConfig(**logging_params)
    logging.debug('Logger configured')

