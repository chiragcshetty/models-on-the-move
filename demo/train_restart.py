import torch
from torch import optim, nn
import sys, logging
import numpy as np
import time

import model_nursery as mn
import mm_lib as mm
from utilities import logger

# To restart a training job that has just been
# moved to the current device

#----------------------- for logging -----------------------
_LOGGER = logger.get_logger(__file__, level=logger.INFO)
#------------------------------------------------------------

def main():
    
    # -- Read terminal input ---
    checkpoint_path = sys.argv[1]
    #-- some knobs for experiemmnting (no real use) -
    factor     = 100  # to scale model size
    inpt_factor = 1   # to scale  input size

    #------------- Setup training from the checkpoint -------------------
    _LOGGER.info("Beginning loading model and setup.....")
    start = time.time()
    
    checkpoint  = torch.load(checkpoint_path)
    #TODO: new name is not required. But currently coordinator doesn't remove dead jobs from its dict
    # After ensuring the jobs dict maintianed at coordinator is updated after a job dies, remove this
    worker_name = checkpoint['worker_name'].split('_')[0]+"_new_"+str(id(checkpoint))
    # -- init the mm worker ----
    worker = mm.mm_init(worker_name) 

    batch_size  = checkpoint['batch_size']
    inp_size    = checkpoint['inp_size']
    opt_size    = checkpoint['opt_size']
    Nrun        = checkpoint['Nrun']
    model = mn.get_model(checkpoint['model_name'], factor)
    optimizer = mn.get_optimizer(checkpoint['optimizer'], model.parameters(), lr = 10); optimizer.zero_grad()
    criterion = mn.get_criterion(checkpoint['criterion'])
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Creates a model_info, send to coordinator, get the placement
    # and outputs the modified distributed model accordng to the placement
    # out_gpu = gpu of the last layer. Labels must be moved to this device
    model, out_gpu = worker.model_setup(model, inp_size, opt_size, batch_size)
    #TODO: some optimizer like adagrad maynot work if model is moved across
    # devices after the optimizer has been defined.
    
    end = time.time()
    _LOGGER.info("Time taken by {} for setup: {} ms".\
            format(worker.NAME, 1000*(end-start)) )

    inp_size = (batch_size,) + inp_size
    opt_size = (batch_size,) + opt_size
    
    #-- for experiment, setup dummy data ---------
    _LOGGER.info("Generating random inputs (may take a few minutes)....")
    start = time.time()
    torch.random.manual_seed(0)
    inp_data   = torch.randn((Nrun,) + inp_size)*(inpt_factor)
    labels_data = torch.randn((Nrun,) +opt_size)

    end = time.time()
    _LOGGER.info("Time taken by {} for input generation: {} ms".\
            format(worker.NAME, 1000*(end-start)) )


    #-----------------------------------------------------------------------------

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
        
        # After each step, check if a modifiaction request was sent by the coordinator
        # If so, morph the model according to new placement and proceed
        # CHECK: will 'model_out' coexist with 'model' oor are they same model?
        #         If former, it may take more memory # TODO
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

