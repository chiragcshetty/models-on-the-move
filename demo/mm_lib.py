#source 1: https://stackoverflow.com/questions/6920858/interprocess-communication-in-python
#source 2: 
import threading
import time
import torch
import socket
import json


import config
import mm_graph as mmg
from utilities import logger

_LOGGER = logger.get_logger(__file__, level=logger.INFO)


def copy_dict(from_dict, to_obj):
    for i, node in from_dict.items():
        to_node = to_obj.nodes[from_dict[i]['id']]
        to_node.device = from_dict[i]['device']


class Worker:
    def __init__(self, name, port_no=0):
        self.HOST  = 'localhost'
        self.NAME  = name

        self.reconfig_flag = False

        self.sock_to_coordinator = socket.socket() 

        self.setup_ready = threading.Event()
        self.reconfig_ready = threading.Event()
        
        self.model = None
        self.inp_size = None
        self.model_info = None
        
    def listener_daemon(self):
        #conn = self.listener.accept()
        _LOGGER.info("----> WORKER {} - is now connected to coordinator".\
                                    format(self.NAME))
        while True: 
            msg = self.sock_to_coordinator.recv(1024).decode()
            cmd = msg.split('-',1)
            if cmd[0] == 'CLOSE':
                conn.close()
                break
            elif cmd[0] == "SETUP":
                # Initial setup instruction from coordinator 
                setup = json.loads(cmd[1])
                #copy_dict(setup, self.model_info)
                self.model_info.copy_device_allotment(setup)
                self.setup_ready.set()
                
            elif cmd[0] == "RECONFIG":
                # Reconfig instruction from coordinator
                setup = json.loads(cmd[1])
                self.model_info.copy_device_allotment(setup)
                self.reconfig_ready.set()
                
        self.listener.close()

    def launch_listener(self):
        # run as daemon to stop listener when main process ends
        #self.listener.listen()
        listener_thread = threading.Thread(target=self.listener_daemon, args=(), daemon=True) 
        listener_thread.start()

    def ping_coordinator(self):
        self.sock_to_coordinator.connect((config.COORDINATOR_IP,\
                     config.COORDINATOR_PORT))

    def send_to_coordinator(self, msg):
        if isinstance(msg, (str)):
            self.sock_to_coordinator.send(msg.encode())
        elif isinstance(msg, (dict)):
            msg_out = "INFO-" + json.dumps(msg)
            self.sock_to_coordinator.send(msg_out.encode())


    def model_setup(self, model, inp_size):
        self.model = model
        self.inp_size = inp_size
        self.model_info = mmg.ModelPy(self.model, self.inp_size)
        self.model_info.trace()
        info = self.model_info.get_model_info_as_dict()
        # send info to coordinator
        _LOGGER.info("---> Sending model info to coordinator")
        self.send_to_coordinator(info)

        # wait for setup instruction from coordinator
        self.setup_ready.wait()
 
        # modify model acc. to instruction
        _LOGGER.info("Modifying the model at {}".format(self.NAME))
        modified_model, last_gpu = self.model_info.modify_model()
        
        return modified_model, last_gpu

    def model_reconfig(self):
        _LOGGER.info("Modifying the model at {}".format(self.NAME))
        modified_model, last_gpu = self.model_info.modify_model()
        return modified_model, last_gpu

    def check_reconfig_status(self):
        if self.reconfig_ready.is_set():
            modified_model, last_gpu = self.model_reconfig()
            self.reconfig_ready.clear()
            return True, modified_model,last_gpu
        else:
            return False, None, None
        #TODO: add lock for reconfig status
        #clear event

def mm_init(name, port_no=0):
    _LOGGER.info("-> Starting WORKER {}".format(name))
    w = Worker(name, port_no)
    
    # Connect to the coordinator
    w.ping_coordinator()
    msg = "JOIN-" + name
    w.send_to_coordinator(msg)
    w.launch_listener()
    return w


