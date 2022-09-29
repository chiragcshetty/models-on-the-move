import torch
from torch import optim, nn
import sys, logging
import numpy as np
import time

import model_nursery as mn
import mm_lib as mm
from utilities import logger

# run on termminal as -> python train.py job_name 2000
# 2000 here is the number of training steps to run the job for

# tag "MODIFICATION" in code below = additions required to 
# a normal training script for it to run with mm (model mobility)

#----------------------- for logging -----------------------
_LOGGER = logger.get_logger(__file__, level=logger.INFO)
#------------------------------------------------------------

def main():
    
    # -- Read terminal input ---
    worker_name = sys.argv[1]
    Nrun        = int(sys.argv[2])

    # -- init the mm worker ----
    # MODIFICATION
    worker = mm.mm_init(worker_name) 

    # ---------------------- training ------------------------------------
    batch_size = 128
    
    #-- some knobs for experiemmnting (no real use) -
    factor     = 100  # to scale model size
    inpt_factor = 0.0001   # to scale  input size
    inp_size =  (int(64*factor),)
    opt_size =  (int(10*factor),)

    #--- import (and initialize) the model ------------
    model = mn.basicModel(factor)
    end = time.time()
    
    # --- setup training ------
    _LOGGER.info("Beginning Setup.....")
    start = time.time()

    # MODIFICATION
    # Creates a model_info, send to coordinator, get the placement
    # and outputs the modified distributed model accordng to the placement
    # out_gpu = gpu of the last layer. Labels must be moved to this device
    model, out_gpu = worker.model_setup(model, inp_size, opt_size, batch_size)

    optimizer = optim.SGD(model.parameters(), lr = 10); optimizer.zero_grad()
    criterion = nn.MSELoss()
    #TODO: some optimizer like adagrad maynot work if model is moved across
    # devices after the optimizer has been defined.
    
    end = time.time()
    _LOGGER.info("Time taken by {} for setup: {} ms".\
            format(worker.NAME, 1000*(end-start)) )

    #----------------------------------------------------------------------------------------------
        #-- for experiment, setup dummy data ---------
    _LOGGER.info("Generating random inputs (may take a few minutes)....")

    inp_size = (batch_size,) + inp_size
    opt_size = (batch_size,) + opt_size

    start = time.time()
    torch.random.manual_seed(0)
    #inp_data   = torch.randn((Nrun,) + inp_size)*(inpt_factor)
    #labels_data = torch.randn((Nrun,) +opt_size)
    inp_data   = torch.ones((Nrun,) + inp_size)*(inpt_factor)
    labels_data = torch.ones((Nrun,) +opt_size)
    _LOGGER.info("Time taken by {} for input generation: {} ms".\
            format(worker.NAME, 1000*(end-start)) )

    #-----------------------------------------------------------------------------------------------------------

    out_sum = 0 # running sum of outputs. For correctness check of outputs
    times =[]
    _LOGGER.info("Starting training at {}".format(worker.NAME))

    for run_no in range(Nrun):
        if run_no%100==0:
            _LOGGER.info("{} ---> Run number: {}".format(worker.NAME, run_no))
        start = time.time()
        inp = inp_data[run_no]
        labels = labels_data[run_no].to(out_gpu)

        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output,labels)
        loss.backward(loss)
        optimizer.step()

        out_sum = out_sum + torch.sum(output.detach().clone())
        
        times.append(1000*(time.time()-start))
        
        # MODIFICATION
        # After each step, check if a modifiaction request was sent by the coordinator
        # If so, morph the model according to new placement and proceed
        # CHECK: will 'model_out' coexist with 'model' oor are they same model?
        #         If former, it may take more memory # TODO
        #--pass optimizer, loss and no of remianing runs,  incase the model needs to be
        #--reconfiged out to different device
        reconfig_happened, model_out, out_gpu_temp\
                    = worker.check_reconfig_status(optimizer, criterion, Nrun-run_no)  
        if reconfig_happened:
            model = model_out
            out_gpu = out_gpu_temp
            _LOGGER.info("Reconfig request at {} ".format(worker.NAME))
            torch.cuda.empty_cache() # Necessary to recover free memory
            _LOGGER.info("{} at run no {} ->\nSo far median step time = {}\nmax step time= {} at run no: {}\n"\
                            .format(worker.NAME, run_no, np.median(times), np.max(times),  np.argmax(times)) )
            times = []
            
        out_sum = out_sum.to(out_gpu)
        
    _LOGGER.info("END of TRAINING at {} ->\nfrom last reconfig median step time = {}\nmax step time\
                         = {} at run no: {}\n".format(worker.NAME, run_no,\
                         np.median(times), np.max(times),  np.argmax(times)) )

    

if __name__ == '__main__':   # important to have to work as a child process of the coordinator (check python multit-processing)
    main()
    torch.cuda.empty_cache()

