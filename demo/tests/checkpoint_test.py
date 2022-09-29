#source: https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
import torch
from torch import optim, nn
import time
import socket
import config

import model_nursery as mn
print("TRANSFER TYPE:", config.TRANSFER_TYPE)

def _get_modified_forward( original_forward, device):
    # Important: do not pass node and use node.forward,node.device
    # in modified_forward, since changing node's device in some 
    # other method may modify the forward on the fly 
    # (eg: in between a training run) due to reference to node 
    # within the modified_forward
    def modified_forward(self,*inputs):
        #########################################################
        # move all inputs to the modules gpu
        input_list = list(inputs)
        for i, inp in enumerate(input_list):
            input_list[i] = inp.to(device)
        inputs = tuple(input_list)
        ########################################################
        output = original_forward(*inputs) 
        return output
    return modified_forward

### define model and move to devices
factor = config.FACTOR
model = mn.basicModel(factor)

###############################################################3
# save before modifying the model
dummy_inp = torch.ones((32,64*factor))*0.0001
with torch.no_grad():
    print(torch.sum(model(dummy_inp)))
    traced_cell = torch.jit.trace(model, (dummy_inp))
torch.jit.save(traced_cell, "model_torchscript.pth")
################################################################

dev = [0,1,2,3]

i=0
sub_modules = model.__dict__['_modules']
for name, sub_module in sub_modules.items():
    sub_module.forward=_get_modified_forward(sub_module.forward, dev[i]).__get__(sub_module, sub_module.__class__) 
    sub_module.to(dev[i])
    i = i + 1


## training
optimizer = optim.SGD(model.parameters(), lr = 10); optimizer.zero_grad()
criterion = nn.MSELoss()

inp   = torch.randn((32,64*factor))
labels = torch.randn((32,10*factor)).to(dev[-1])

for run_no in range(10):
    print(run_no)
    optimizer.zero_grad()
    output = model(inp)
    loss = criterion(output,labels)
    loss.backward(loss)
    optimizer.step()

EPOCH = 10
PATH = "model.pt"
LOSS = 0.4

torch.save({
            'model_name': 'basicModel',
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'criterion': 'MSELoss',
            'optimizer': 'SGD',
            'loss': LOSS,
            'batch_size':32,
            'Nrun': 2000,
            'inp_size': (1,6400),
            'worker_name': 'job_k'
            }, PATH)



del model
del optimizer
del criterion
torch.cuda.empty_cache()
print("Done saving and cleaning")
time.sleep(5)

##########################3
model = mn.basicModel(factor)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.MSELoss()
labels = torch.randn((32,10*factor))

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

dummy_inp = torch.ones((1,64*factor))*0.0001
print("value:", torch.sum(model(dummy_inp)))

print("Loading done. Now training")
for run_no in range(10):
    print(run_no)
    optimizer.zero_grad()
    output = model(inp)
    loss = criterion(output,labels)
    loss.backward(loss)
    optimizer.step()

# Checkpointing does not retain device information. The 
# reloaded model runs on the cpu


# https://stackoverflow.com/questions/56194446/send-big-file-over-socket
print("sending now")
sock_to_coordinator = socket.socket() 
sock_to_coordinator.connect((config.COORDINATOR_IP,\
                     config.COORDINATOR_PORT))


if config.TRANSFER_TYPE == 'chkpt':
    PATH = "model.pt"
elif config.TRANSFER_TYPE == 'torchscript':
    PATH = "model_torchscript.pth"


with sock_to_coordinator:
    with open(PATH, 'rb') as file:
        sendfile = file.read()
    sock_to_coordinator.sendall(sendfile)
    print('file sent')