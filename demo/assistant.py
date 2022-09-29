import socket
import json
import torch
import sys, os
import time

import config
from utilities import logger
_LOGGER = logger.get_logger(__file__, level=logger.INFO)

"""
assitant register each available node to the coordinator with
the devices available and their info
Run as: python assistant.py <machine_name>
"""

_LOGGER.info("Starting Assistant Listener...")
machine_name = sys.argv[1]
sock_to_coordinator = socket.socket()

dev_info = {}

no_gpus = torch.cuda.device_count()
for dev in range(no_gpus):
    mem = torch.cuda.get_device_properties(dev).total_memory
    _LOGGER.info("Device {} : Memory = {}".format(dev, mem))
    dev_info[dev] = mem

sock_to_coordinator.connect((config.COORDINATOR_IP,\
                    config.COORDINATOR_PORT))
msg_out = "REGISTER-" + machine_name + "-" + json.dumps(dev_info)
sock_to_coordinator.send(msg_out.encode())

## TODO-1:keep listeing to recieve models moving here

my_ip = socket.gethostbyname(socket.gethostname())
listener = socket.socket() 
listener.bind((my_ip, config.ASSISTANT_PORT))
listener.listen()
_LOGGER.info("Listening to any incoming jobs")

file_id = 0
while True:
    conn, addr = listener.accept()
    _LOGGER.info("Connection from: {}".format(addr))
    file_id+=1
    savefilename = 'checkpoints/model_rcvd'+str(file_id)+'.pt'
    with conn ,open(savefilename,'wb') as file:
        while True:
            recvfile = conn.recv(4096)
            if not recvfile: break
            file.write(recvfile)
    _LOGGER.info("Model recvd from {} and saved at {}. Launching training now".format(addr, savefilename))
    ## Now launch a new job with the recvd model
    os.system("python train_restart.py "+savefilename+" &")
    _LOGGER.info("Launched new job.....")
