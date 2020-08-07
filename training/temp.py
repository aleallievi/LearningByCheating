import numpy as np

model_params = []
with open('/v/filer5b/l_pstone/agents6/brahma/CARLA/carla_lbc/training/model_seed_params.txt', 'r') as f:
    line = f.readline()
    while line:
        print (line)
        model_params.append(float(line))
        line = f.readline()

m = np.array(model_params)
print (m.shape)
