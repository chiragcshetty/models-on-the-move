import socket
import config
import torch
import model_nursery as mn
from torch import optim, nn
import time

def _get_modified_forward( original_forward, device):
    # Important: do not pass node and use node.forward,node.device
    # in modified_forward, since changing node's device in some 
    # other method may modify the forward on the fly 
    # (eg: in between a training run) due to reference to node 
    # within the modified_forward
    def modified_forward(self,*inputs):
        #########################################################
        # move all inputs to the modules gpu
        print("inside")
        input_list = list(inputs)
        for i, inp in enumerate(input_list):
            input_list[i] = inp.to(device)
        inputs = tuple(input_list)
        ########################################################
        output = original_forward(*inputs) 
        return output
    return modified_forward
    ############################################

print("TRANSFER TYPE:", config.TRANSFER_TYPE)

listener = socket.socket() 
listener.bind((config.COORDINATOR_IP,\
            config.COORDINATOR_PORT))
listener.listen()
print("listening")

#https://stackoverflow.com/questions/56194446/send-big-file-over-socket

conn, addr = listener.accept()
savefilename = 'model_rcvd.pt'
with conn ,open(savefilename,'wb') as file:
    print("rcving")
    while True:
        recvfile = conn.recv(4096)
        if not recvfile: break
        file.write(recvfile)
print("File has been received.")

##########################################################
factor = config.FACTOR

if config.TRANSFER_TYPE == 'chkpt':

    model = mn.basicModel(factor)
    checkpoint = torch.load(savefilename)
    model.load_state_dict(checkpoint['model_state_dict'])
    batch_size = checkpoint['batch_size']
    inp_size = checkpoint['inp_size']
    print("sleeping");time.sleep(10)


    dummy_inp = torch.ones((1,64*factor))*0.0001
    print("value", torch.sum(model(dummy_inp)))

# Note: destination must have access to the model code to be able to load it. Else it can't
# you can use torch-script, but only for inference:
# https://stackoverflow.com/questions/59287728/saving-pytorch-model-with-no-access-to-model-class-code

elif config.TRANSFER_TYPE=='torchscript':

    dummy_inp = torch.ones((1,64*factor))*0.0001
    model = torch.jit.load(savefilename)
    with torch.no_grad():
        print("value", torch.sum(model(dummy_inp)))

    print("train")

    dev = [0,1,2,3]

    '''
    # this doesn't work
    i=0
    sub_modules = model.__dict__['_modules']
    for name, sub_module in sub_modules.items():
        sub_module.forward=_get_modified_forward(sub_module.forward, dev[i]).__get__(sub_module, sub_module.__class__) 
        sub_module.to(dev[i])
        i = i + 1
    inp   = torch.randn((32,64*factor)).to(0)
    labels = torch.randn((32,10*factor)).to(3)
    '''

    sub_modules = model.__dict__['_modules']
    for name, sub_module in sub_modules.items():
        print(sub_module.forward)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()
    inp   = torch.randn((32,64*factor))
    labels = torch.randn((32,10*factor))
    print(model)
   

    for run_no in range(5000):
        if (run_no%100)==0:
            print(run_no)
        optimizer.zero_grad()
        output = model(inp)
        loss = criterion(output,labels)
        loss.backward(loss)
        optimizer.step()