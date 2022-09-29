import torch
import time
from torch import optim, nn

from utilities import utilsf, logger
import config
import copy

_LOGGER = logger.get_logger(__file__, level=logger.INFO)

class SubModuleNode:
    """
    Each layer (or block of layers as specified in config.BASIC_BLOCKS)
    in the model is representing as a node using this class
    """
    def __init__(self):
        # the module this node represents
        self.module = None
        self.module_id = None
        # submodule name
        self.name = None

        # submodule memory requirement
        self.memory = 0 
        # forward funtion of the module will be
        # wrapped when the the model morphs to
        # fit into the devices. original forward
        # method, which defines the operation 
        # performed by the submodule is stored here
        self.original_forward = None
        
        # device alloted to the layer
        self.device = None
        # topological order of the layer
        # during model forward pass
        self.topo_order = None


class ModelPy:
    """
    Info about a pytorch model and methods to trace, and morph it
    """
    def __init__(self, model, inp_size):
        self.model      = model
        self.inp_size   = inp_size # input size including batch

        # map (sub_module id):(node representing it)
        self.nodes = {} 

        self._topo_count = 0
        self._hook_handles =[]

    def recur_get_nodes(self, module):
        """
        Models are composed of nested modules. This function
        recursively gets the layers comprising the model, creates
        a node for each of it and adds a forward hook to get the 
        topo_order of the layer duing trace()
        """
        this_context = self
        sub_modules = module.__dict__['_modules']
        for name, sub_module in sub_modules.items():
            # if there are more than 1 submodules of sub_module, we need further recursion
            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0 and (sub_module.__class__.__name__\
                                             not in config.BASIC_BLOCKS):
                this_context.recur_get_nodes(sub_module)
                continue

            #---------------Define a node for the submodule-------------------------
            node        = SubModuleNode()
            node.module = sub_module
            node.module_id = id(sub_module)
            node.name   = name
            # get meomory for layer parameters
            node.memory = utilsf.estimate_model_size(sub_module)
            node.original_forward = sub_module.forward
            #TODO: check node for module hasn't laready been created
            # Can happen if a same module is used multiple times 
            # in the model
            this_context.nodes[node.module_id] = node

            #-----------Add a forward hook to the submodule-------------------------
            def hook(cur_module, inputs, output):
                cur_node = this_context.nodes[id(cur_module)]
                # add modules' output mem to its mem requirement
                cur_node.memory += utilsf.estimate_tensor_size(output)
                cur_node.topo_order = this_context._topo_count
                this_context._topo_count += 1

            handle = sub_module.register_forward_hook(hook)
            this_context._hook_handles.append(handle)


    def trace(self):
        """
        Runs a dummy forward pass to get topological
        order of execution of layers
        TODO: Add backward run to get computation graph
        (will need to add backward hook in recur_get_nodes)
        """
        self._topo_count = 0
        self.recur_get_nodes(self.model)
        dummy_input = torch.rand(self.inp_size)
        dummy_out = self.model(dummy_input)
        for handle in self._hook_handles:
            handle.remove()

    def get_model_info_as_dict(self):
        '''
        returns info in 'nodes' as dict of dict
        to be sent over network to coordinator
        '''
        # (module topo_order): (dict of module's node info)
        model_info = {}
        for node_id, node in self.nodes.items():
            node_info = copy.deepcopy(node.__dict__)
            # module not required at coordinator (also not json-able)
            del node_info["module"]
            del node_info["original_forward"]
            model_info[node.topo_order] = node_info 
        return model_info

    def copy_device_allotment(self, allotted_model_info):
        '''
        device placement for each node from coordinator is 
        copied to node.device
        '''
        for _, node in allotted_model_info.items():
            # "int" because dict recvd from cordinator is 
            # a json dump, hence all info is string
            self.nodes[node["module_id"]].device = int(node["device"])

    def _get_modified_forward(self, original_forward, device):
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

    def modify_model(self):
        '''
        Assumes a device allotment has been rcvd from coordinator 
        and copied to node.device's . Modify each layer/node
        in the module with forward method according to the
        device allocation
        '''
        final_layer_topo = len(self.nodes)-1
        gpus_list = []
        for node_id, node in self.nodes.items():
            submodule = node.module
            device = node.device
            _LOGGER.info("{} goes to gpu-{}".format(node.name, device))
            # move module to allotted device
            submodule.to(device)
            # modify module's forward
            submodule.forward =  self._get_modified_forward(node.original_forward, node.device).\
                                    __get__(submodule, submodule.__class__) 
            if node.topo_order == final_layer_topo:
                last_gpu = node.device
            if device not in gpus_list:
                gpus_list.append(device)

        # wait for all transfers to devices to complete
        for dev in gpus_list:
            torch.cuda.synchronize(dev)
        return self.model, last_gpu



'''
# TODO: For tensorflow
class ModelTF:
    def __init__(model, inp_size):
        pass

    def trace(self):
        pass
    
    def modify_model(self):
        pass

    def get_model_info_as_dict(self):
        pass

'''