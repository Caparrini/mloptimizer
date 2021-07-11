import logging
import sys
import os


def init_logger(filename='optimization.log', log_path="."):
    logging_params = {
        # 'stream': sys.stdout,
        'level': logging.DEBUG,
        'format': '%(asctime)s:%(levelname)s:[L%(lineno)d]:%(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
        'filename': os.path.join(log_path, filename)
    }
    # print(os.listdir("../"))
    # print(os.path.abspath(__file__))
    logging.basicConfig(**logging_params)
    l = logging.getLogger("mloptimizer")
    l.debug('Logger configured')
    return l
