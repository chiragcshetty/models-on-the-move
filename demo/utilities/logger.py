from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import logging

CRITICAL=logging.CRITICAL
ERROR=logging.ERROR
WARNING=logging.WARNING
INFO=logging.INFO
DEBUG=logging.DEBUG

logging.basicConfig(filename='runtime.log', level=logging.INFO )
logging.getLogger("matplotlib").setLevel(logging.WARNING)

def get_logger(filepath, level=INFO, detailed=False):
    name = os.path.splitext(os.path.basename(filepath))[0]
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stderr)
    if detailed:
        handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'))
    else:
        handler.setFormatter(logging.Formatter(
        '%(name)s:%(lineno)d -> %(message)s'))
    logger.addHandler(handler)
    return logger
