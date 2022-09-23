import socket
import json
import torch
import sys

import config
from utilities import logger
_LOGGER = logger.get_logger(__file__, level=logger.INFO)

"""
assitant register each available node to the coordinator with
the devices available and their info
"""
machine_name = sys.argv[1]
sock_to_coordinator = socket.socket()

def ping_coordinator():
    sock_to_coordinator.connect((config.COORDINATOR_IP,\
                    config.COORDINATOR_PORT))

dev_info = {}

no_gpus = torch.cuda.device_count()
for dev in range(no_gpus):
    mem = torch.cuda.get_device_properties(dev).total_memory
    _LOGGER.info("Device {} : Memory = {}".format(dev, mem))
    dev_info[dev] = mem

ping_coordinator()
msg_out = "REGISTER-" + machine_name + "-" + json.dumps(dev_info)
sock_to_coordinator.send(msg_out.encode())

## keep listeing to recieve models moving here