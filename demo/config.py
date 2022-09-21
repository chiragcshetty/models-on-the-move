# modules which should be treated as a single node in graph
# although they are composed of further submodules. 
# Typically they are user defined blocks and repeated 
# througout the model (eg: Basic inception block)
BASIC_BLOCKS     = []

#-------------------------------------------------------
COORDINATOR_IP   = 'localhost'
COORDINATOR_PORT = 6000