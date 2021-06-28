import itertools
import sys
import os

source_list = sys.argv[1]
target_list = ['W4633070102', 'W4633080200', 'W4662FM0400', 'W4662FM0507', 'W4662FM0605', 'W4662FM0606']

epoch_list = range(20, 200+20, 20)
timestep_list = range(4, 32+4, 4)
layer_1 = range(128, 512+32, 32)
layer_2 = range(16, 128+16, 16)

hyperparameters = [source_list, target_list, epoch_list, timestep_list, layer_1, layer_2]
hp_list = list(itertools.product(*hyperparameters))

for hp in hp_list:

    source = hp[0]
    target = hp[1]
    epoch = str(hp[2])
    timesteps = str(hp[3])
    layer1 = str(hp[4])
    layer2 = str(hp[5])
    gpu = int(sys.argv[2])

    if (source != target) and (layer1 > layer2):

        command = 'python csc_transformer.py {} {} {} {} {} {} {}'.format(source, 
                                                                        target,
                                                                        epoch,
                                                                        timesteps,
                                                                        layer1,
                                                                        layer2,
                                                                        gpu)

        print(command)
        os.system(command)