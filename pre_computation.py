import numpy as np
from scipy.special import erf
import multiprocessing
from functools import partial
import sys, io, time, os
import subprocess

class pre_computation:
    def pre_compute(self, U, bw, events, Aij):
        N = len(events)
        tune_parameter = 4
        div_ver = int(U/tune_parameter)
        # cmd = 'truncate -s 0 pre_com.sh'
        # os.system(cmd)
 
        # with open('pre_com.sh', 'r+') as script:
        #     script.write('#!/bin/sh \n')
        #     script.write('rm Pre_Files/* \n')
        #     for i in range(tune_parameter - 1):
        #         script.write('python2 precompute_text.py '+str(div_ver*i)+' '+ str(int(div_ver*(i+1))) + ' &\n')

        #     script.write('python2 precompute_text.py '+str(div_ver*(i+1))+' end \n')
        #     script.write('wait\n')
        #     script.write('cat $(find ./Pre_Files/ -name "pre_compute_map_*" | sort -V) > ./Pre_Files/pre_compute_map.txt \n')
        #     script.write('cat $(find ./Pre_Files/ -name "pre_compute_Aij_*" | sort -V) > ./Pre_Files/pre_compute_Aij.txt\n')

        # script.close()

        # subprocess.call(['./pre_com.sh'])
        pre_compute_map = np.loadtxt('Pre_Files/pre_compute_map.txt', delimiter=' ')
        pre_compute_Aij = np.loadtxt('Pre_Files/pre_compute_Aij.txt', delimiter=' ')
        
        print(pre_compute_map.shape, pre_compute_Aij.shape)
        return (pre_compute_map, pre_compute_Aij)
