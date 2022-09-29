import socket, json
import random # remove this: used in dummy logic
import threading
from utilities import logger

#----------------------- for logging -----------------------
_LOGGER = logger.get_logger(__file__, level=logger.INFO)

############################################################################
# Some DEFINITONS:
# (re)configuration -> Allotment of a set of devices for a job to run on
# placement -> allotement of nodes of the job to specific devices in 
#           the given configuration
#----------------------------------------------------------
class Job:
    """
    Class representing a training job
    """
    def __init__(self, job_name, model_info, job_conn, job_addr, job_machine):
        self.name        = job_name
        self.conn        = job_conn   # connection to submittor of the job
        self.model_info  = model_info # an object of mm_graph.ModelPy class
        self.addr        = job_addr   # ipaddr of machine job is running on
        self.machine     = job_machine # name of machine job is running on
        self.dev_mem_map = {}         # device:(mem occupied by the job)

    def send_setup(self):
        # Send the initial setup configuration to the job's process,
        # stating which node goes to which device.
        # NOTE: This must be done after Configurator has
        # modified the job's model_info with placment information
        msg_out = ("SETUP-"+ self.addr + ":"+ json.dumps(self.model_info))\
                                            .encode()
        self.conn.send(msg_out)
        _LOGGER.info("Intial setup for job {} sent".format(self.name))

    def send_reconfig(self):
        # Send the reconfigured placement to the job's process
        # NOTE: This assumes the configurator has made the fed the
        # reconfigured placmeent info into model_info
        msg_out = ("RECONFIG-" + json.dumps(self.model_info))\
                                            .encode()
        self.conn.send(msg_out)
        _LOGGER.info("Reconfiguration for job {} sent".format(self.name))

    def move_reconfig(self, destination_machine):
        # When reconfig involves moving model to a different device.
        msg_out = ("MOVE-" + destination_machine).encode()
        self.conn.send(msg_out)
        _LOGGER.info("Job {} is requested to move to machine {}. It will take sometime."\
                                    .format(self.name, destination_machine))
        

class Machine:
    """
    Class representing machines in the cluster. conn connects
    to assitant running on that machine
    """
    def __init__(self, name, ip_addr, device_map):
        self.name       = name
        self.ip_addr    = ip_addr
        self.device_map = device_map # map gpu_id to {'available_mem':xx, 'jobs_running':[]}
        self.no_devices = len(device_map)

#----------------------------------------------------------------------
class Configurator:
    """
    (Re)configuration and Placement logics sits here
    """
    def __init__(self, cluster_info):
        # cluster_info[ip_addr] = {'name':machine_name,'addr':ip_addr ,'device_info' = device_map}
        # device_map[dev_id] = {'available_mem': mem_in_bytes, 'jobs_running':[]}
        # TODO: Add info of memory occupied per job as well
        self.cluster_info = cluster_info # map machine_ip_addr : Machine_object
        self.cluster_machine_by_name = {} # map machine_name : Machine_object
        for _, machine in self.cluster_info.items():
            self.cluster_machine_by_name[machine.name] = machine

        self.job_map = {}  # map job_name : Job_object

        # Thread listening to any reconfig inputs comming from the terminal
        # at the coordinator
        reconfig_listener = threading.Thread(target=self.reconfig_listener,\
                                                 args=(), daemon=True) 
        reconfig_listener.start()
        #TODO add locks to avoid conflict between add_new_job and reconfig_listener

    def add_new_machine(self, machine_name, ip_addr, device_info):
        if ip_addr not in self.cluster_info:
            device_map = {}
            for dev in device_info:
                device_map[int(dev)] = {'available_mem':int(device_info[dev]),\
                                'jobs_running':[]}
            machine = Machine(machine_name, ip_addr, device_map)
            
            self.cluster_info[ip_addr] = machine
            self.cluster_machine_by_name[machine_name] = machine
            _LOGGER.info("Machine {} joined with ip_addr {}".format(machine_name, ip_addr))
        else: 
            _LOGGER.info("Machine already registered. Ignoring request")
            #TODO: need to raise exception? 

    # When a new job pings the coordinator, creatte a Job object
    # and register it with the Configurator. But check that its machine
    # has REGISTERED first
    def add_new_job(self, job_name, model_info, job_conn, job_addr):
        if job_addr in self.cluster_info:
            job_machine = self.cluster_info[job_addr].name
            job = Job(job_name, model_info, job_conn, job_addr, job_machine)
            self.job_map[job_name] = job
            dev_config = self.get_config(job)
            job = self.get_placement(job, dev_config)
            job.send_setup()
        else:
            job_conn.send("DENIED".encode())
            _LOGGER.info("Job {} tried to join from machine {}. But the machine is not registered. Please run assistant.py on that machine first".\
                    format(job_name, job_addr))

    def reconfig_listener(self):
        #listen to terminal
        #terminal input format:
        # to reconfig within same machine-> reconfig:job_name-dev1,dev2
        # to move to a new machine->        move:job_name-destination_machine_name
        while True:
            user_inp = input()
            try:
                request_type, request = user_inp.split(':',1)
            except:
                _LOGGER.info("Invalid request format. Request format--> move:job_name-machine_name"\
                        +" or reconfig:job_name-machine_name")
            else:
                req = request.split('-') # TODO: check for inputs not following the fformat
                job_name = req[0]
                if request_type == "reconfig":
                    if job_name in self.job_map:
                        _LOGGER.info("New reconfig request recieved for job {} ".\
                                            format(job_name))
                        devices = [int(i) for i in req[1].split(',')]
                        job = self.get_placement(self.job_map[job_name], devices)
                        # TODO: add a check if placement is not possible on given
                        # devices
                        job.send_reconfig()
                    else:
                        _LOGGER.info("Job {} not found. Request format--> reconfig:job_name-dev1,dev2".format(job_name))

                elif request_type == "move":
                    if job_name in self.job_map:
                        dest_machine = req[1]
                        if dest_machine in self.cluster_machine_by_name:
                            _LOGGER.info("New request recieved to move job {} to machine {} ".\
                                            format(job_name, dest_machine))
                            self.job_map[job_name].move_reconfig(self.cluster_machine_by_name[dest_machine].ip_addr)
                            del self.job_map[job_name] # job will kill itself and restart at 
                                                # the new machine and ping coordinator
                                                # TODO: get an ack from job before forgetting it

                        else:
                            _LOGGER.info("Invalid machine name. Request format--> move:job_name-machine_name")
                    else:
                        _LOGGER.info("Invalid job name or move request. Request format--> move:job_name-machine_name")


    
    def get_config(self, job):
        #!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-
        # TODO: Add LOGIC to genearate a (re)configuration given
        # a job and device map
         #!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

        # placeholder dummy logic:
        # Randomly picks a device and the next device to split the model across
        L = self.cluster_info[job.addr].no_devices
        dev1 = random.randint(0,L-1)
        dev2 = (dev1+1)%L
        self.cluster_info[job.addr].device_map[dev1]['jobs_running'].append(job)
        self.cluster_info[job.addr].device_map[dev2]['jobs_running'].append(job)
        return [dev1, dev2]
    
    def get_placement(self, job, dev_config):
        #!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-
        # TODO: Add LOGIC to genearate a placement given
        # a reconfiguration and device map
         #!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-!-

        # placeholder dummy logic:
        # Allots first half of nodes to dev_1 and next half to dev_2
        # TODO: check dev_config is correct/valid
        no_nodes = len(job.model_info)
        for k, node in job.model_info.items():
            i=int(k)
            if i<= no_nodes/2 - 1:
                dev = dev_config[0]
            else:
                dev = dev_config[1]
            # update the model_info with device allocation
            job.model_info[k]['device'] = dev
            # update the memory of the devices according to 
            # the generated placement
            # TODO: Update this only after the job ack's the morphing. May
            # need locks to prevent other jobs from being allotted while waiting
            # for ack
            self.cluster_info[job.addr].device_map[dev]['available_mem'] -= job.model_info[k]['memory']

        _LOGGER.info("Placement generated for job {}".format(job.name))
        _LOGGER.info("Updated Device Map:")

        machine =self.cluster_info[job.addr]
        for d in machine.device_map:
            _LOGGER.info("Machine {}, Device {} : {} bytes".format(machine.name, d, machine.device_map[d]['available_mem']))
        return job
            
    # TODO remove job and update device_map and mem when job is done or reconfigured out
    # TODO: First time a job touches a device, it occupies ~800MB to setup its cuda 
    #       runtime. So update the device_map accordingly





