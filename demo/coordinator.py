#-------- Python libraries ---------------
import socket, threading
import json   # used for transferring dicts 
              # over sockets 

#-------- project specific libraries ------
import config
from utilities import logger
import configurator
#--------------- for logging --------------
_LOGGER = logger.get_logger(__file__, level=logger.INFO)
#-----------------------------------------
############################################################################################

#------------ Coordinator setup ----------
listener = socket.socket() 
listener.bind((config.COORDINATOR_IP,\
            config.COORDINATOR_PORT))
listener.listen()
_LOGGER.info("Coordinator is now listening")

device_map = {} # map of device: {available_mem:xxx, jobs_running:[job1, job2..]}
                # TODO: Add info of memory occupied per job as well

#TODO: automatically available devices using torch.cuda.device_count()
#TODO: preferably make a 'device' class instead oof storing info in dict
MAX_GPU_MEM = 8000000000
device_map[0]  = {'available_mem': MAX_GPU_MEM, 'jobs_running':[]}
device_map[1]  = {'available_mem': MAX_GPU_MEM, 'jobs_running':[]}
device_map[2]  = {'available_mem': MAX_GPU_MEM, 'jobs_running':[]}
device_map[3]  = {'available_mem': MAX_GPU_MEM, 'jobs_running':[]}

# start the Configurator
config_chief = configurator.Configurator(device_map)

# On recieving ping from a new job, the listener_daemon
# launches a handle_worker thread to manage it thereafter
# TODO: If there are a lot of jobs, Python's threading
# will slow everthing down. 
def handle_worker(conn, addr):
    job_name = None
    job_addr = addr
    while True: 
        msg = conn.recv(1024).decode()
        if msg:
            _LOGGER.debug("Message from {}: {}".format(job_addr, msg))
            cmd = msg.split("-",1)
            # first message from job "JOIN"
            if cmd[0] == 'JOIN':
                job_name = cmd[1]
                _LOGGER.info("{} joined".format(job_name))
            # next, the job sends its model graph "INFO"
            elif cmd[0] == 'INFO':
                _LOGGER.info("{} sent model info".format(job_name))
                model_info = json.loads(cmd[1])
                # NOTE: elements in dict output of json.loads will be strings
                # TODO: Check that job_name is not none i.e JOIN has happened before
                placed_model_info = config_chief.add_new_job(\
                                    job_name, model_info, conn)

# Coordinator keeps listening to any incoming request
def listener_daemon():
    while True:
        conn, addr = listener.accept()
        worker_thread = threading.Thread(target=handle_worker, args=(conn, addr,),\
                        daemon=True) 
        worker_thread.start()
    listener.close()

# TODO: Add a main()
listener_thread = threading.Thread(target=listener_daemon, args=(), daemon=True) 
listener_thread.start()
listener_thread.join()