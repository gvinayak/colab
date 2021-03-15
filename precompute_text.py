from __future__ import print_function
import numpy as np
from scipy.special import erf
import multiprocessing
from functools import partial
import sys, io, time
import pickle, pdb
import time

def parallel_pre_compute (i, k, bw, events, Aij, pre_compute_map, pre_compute_Aij):
    nu = - 0.01
    N = len(events)
    s = 0.0
    ik = events[k][0]
    for n in range(k + 1, N):
        prod1 = (np.exp(nu * (events[n][1] - events[k][1])) - np.exp(nu *
                                                                       (events[n - 1][1] -
                                                                        events[k][1]))) / nu
        prod2 = erf((events[n][2] - events[k][2]) / np.sqrt(2 * bw[i])) - erf((
            events[n - 1][2] - events[k][2]) / np.sqrt(2 * bw[i]))
        prod3 = erf((events[n][3] - events[k][3]) / np.sqrt(2 * bw[i])) - erf((
            events[n - 1][3] - events[k][3]) / np.sqrt(2 * bw[i]))
        s = s + (prod1 * prod2 * prod3) / 4.0

    return [ik, s]

def mycallback(ret, i, k, Aij, pre_compute_map, pre_compute_Aij, div1):
    ik = ret[0]
    s = ret[1]
    i = i - div1
    pre_compute_map[(i,k)] = s
    pre_compute_Aij[(i,k)] = Aij[(i,ik)]


def pre_compute():
    [U, bw, events, Aij] = pickle.load(open("pre_compute.p", "rb" ))
    all_sig = pickle.load(open("SA_1_Sig.p", "rb"))
    
    div1 = sys.argv[1]
    div2 = sys.argv[2]
    
    nu = - 0.01
    N = len(events)
    Aij = np.reshape(Aij, (U, U))

    if(div2 == 'end'):
        div2 = U

    div1 = int(div1)
    div2 = int(div2)

    pre_compute_map = np.zeros((div2 - div1,N))
    pre_compute_Aij = np.zeros((div2 - div1,N))

    for i in range(div1, div2):
        pool = multiprocessing.Pool(20)
        for k in range(0, N-1):
            ik = events[k][0]
            if(ik not in all_sig[i]):
                new_callback_function = partial(mycallback, i=i, k=k, Aij=Aij, pre_compute_map=pre_compute_map, pre_compute_Aij=pre_compute_Aij, div1=div1)
                pool.apply_async(parallel_pre_compute, args=(i, k, bw, events, Aij, pre_compute_map, pre_compute_Aij), callback=new_callback_function)

            else:
                pre_compute_map[(i-div1,k)] = 0
                pre_compute_Aij[(i-div1,k)] = 0 

        pool.close()
        pool.join()
    
    np.savetxt('Pre_Files/pre_compute_map_'+str(div1)+'.txt', pre_compute_map, delimiter=' ', fmt='%.5f')
    np.savetxt('Pre_Files/pre_compute_Aij_'+str(div1)+'.txt', pre_compute_Aij, delimiter=' ', fmt='%.5f')
    # print(pre_compute_map)
    # return (pre_compute_map, pre_compute_Aij)

if __name__ == '__main__':
    # t0 = time.time()
    pre_compute()
    # t1 = time.time()

    # total = t1-t0
    # print("Total time taken", total)
