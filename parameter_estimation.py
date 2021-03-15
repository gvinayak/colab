import numpy as np
import sys, io, time, subprocess, os
from simulation.utility import utility
from synthetic_data_processing import synthetic_data_processing
from simulation.kernelParameters import kernelParameters
from parameter_estimation_mu import parameter_estimation_mu
from parameter_estimation_theta import parameter_estimation_theta
from parameter_estimation_aij import parameter_estimation_aij
from parameter_estimation_phi import parameter_estimation_phi
from pre_computation import pre_computation
import pickle
import pdb

class parameter_estimation:
    S = 0
    M = 0
    U = 0
    V = 0
    I = 0
    T = 0
    X = 0
    Y = 0

    # event is a tuple of (user, time, lat, lon, category, community)
    events = []
    checkins = {}
    dump_name = ""

    def __init__(self, num_of_samples, number_of_iter, dump_name):
        self.S = num_of_samples
        self.I = number_of_iter
        self.dump_name = dump_name
    
    def fix_Aij(self, Aij):
        all_sig = pickle.load(open(self.dump_name, "rb"))
        for i in range(Aij.shape[0]):
            for j in range(Aij.shape[1]):
                if(j not in all_sig[i] and i!=j):
                    Aij[i,j] = 0
        return Aij

    def estimate_parameters(self, checkin_file, connections_file, num_community):

        # get data
        (self.events, self.checkins, self.T, self.X, self.Y) = synthetic_data_processing().get_checkins(checkin_file)

        [U_real, V_real] = synthetic_data_processing().get_params_from_file(checkin_file, connections_file)
        M_real = num_community
        num_iteration = 5
            
        mu = np.random.uniform(0,1,(1, U_real))
        sum_mu = np.sum(mu) 
        mu = [i/sum_mu for i in mu]

        # Aij = np.random.uniform(0,1,(U_real, U_real))
        Aij = np.ones((U_real, U_real))
        
        # Set for only significant users
        Aij = self.fix_Aij(Aij)
        
        sum_Aij = np.sum(Aij, axis = 0)
        for k in range(Aij.shape[0]):
            if(sum_Aij[k]) != 0:
                Aij[:,k] = [i/sum_Aij[k] for i in Aij[:,k]]

        pi = np.random.uniform(0,1,(U_real, M_real))
        sum_pi = np.sum(pi, axis = 1)
        for k in range(pi.shape[0]):
            pi[k, :] = [i/sum_pi[k] for i in pi[k, :]]

        theta = np.random.uniform(0,1,(M_real, V_real))
        sum_theta = np.sum(theta, axis=1)
        for k in range(theta.shape[0]):
            theta[k, :] = [i/sum_theta[k] for i in theta[k, :]]

        phi = np.random.uniform(0, 1, (U_real, M_real))
        phi = utility().normalize_2d_array(phi)

        mu = np.asarray(mu)
        Aij = np.asarray(Aij)
        pi = np.asarray(pi)
        theta = np.asarray(theta)

        N = len(self.events)
        print ("N : ", N)
       
        print("The shape is", mu.shape, Aij.shape, pi.shape, theta.shape)
        
        print ("theta : ", theta)
        print ("mu : ", mu)
        print ("Aij : ", Aij)
        print ("phi : ", phi)

        self.U = U_real
        self.V = V_real
        self.M = M_real

        graph = synthetic_data_processing().get_graph(checkin_file, connections_file, self.U)
        # Modifications end --------------------------------------------------------------------

        for iteration in range(num_iteration):
            
            print("Iteration in consideration", iteration)

            # learn bandwidth of each user
            bw_start_time = time.time()
            bw = np.empty((self.U))
            bw = [kernelParameters().kde(self.checkins[user][0], self.checkins[user][1]).bandwidth for user in range(0,self.U)]

            print "bandwidth learn time : ",(time.time() - bw_start_time)

            # pre computation step
            pre_comp_time = time.time()

            # Changes are made here----------------------------------------------------------------------------------------------------------
            pickle.dump([self.U, bw, self.events, Aij], open("pre_compute.p", "wb"))
            (pre_compute_map, pre_compute_Aij) = pre_computation().pre_compute(self.U, bw, self.events, Aij)
            # Changes end here---------------------------------------------------------------------------------------------------------------

            print "pre compute map : ", pre_compute_map

            print "pre compute Aij : ", pre_compute_Aij

            print "pre compute time : ", (time.time() - pre_comp_time)

            obj_cre_time = time.time()
            
            par_phi_obj = parameter_estimation_phi(self.S, self.M, self.U, self.V, self.I, self.T, self.X, self.Y, self.events,
                                                   self.checkins, pre_compute_map, pre_compute_Aij)

            par_mu_obj = parameter_estimation_mu(self.S, self.M, self.U, self.V, self.I, self.T, self.X, self.Y, self.events,
                                                 self.checkins, pre_compute_map, pre_compute_Aij)

            par_theta_obj = parameter_estimation_theta(self.S, self.M, self.U, self.V, self.I, self.T, self.X, self.Y,
                                                       self.events, self.checkins, pre_compute_map, pre_compute_Aij)

            par_aij_obj = parameter_estimation_aij(self.S, self.M, self.U, self.V, self.I, self.T, self.X, self.Y, self.events,
                                                   self.checkins, pre_compute_map, pre_compute_Aij)

            print "object creation time :", (time.time() - obj_cre_time)
            samples = par_phi_obj.getSamples(pi)

            main_dict = {'mu': mu[0, :], 'Aij': Aij, 'phi': phi, 'theta': theta, 'bw': bw, 'pi': pi, 'samples': samples}
            print("The original values are:----------------")

            # mu estimation
            start_time = time.time()
            print("Started the mu optimization----------------------------------->checkpoint 1")
            param_mu = par_mu_obj.optimize(main_dict)
            mu = main_dict['mu']
            main_dict['mu'] = mu[0, :]
            print "MU time :", (time.time() - start_time)

            # theta estimation
            start_time = time.time()
            temp_theta = main_dict['theta']
            theta = np.reshape(main_dict['theta'], (1, np.product(main_dict['theta'].shape)))
            main_dict['theta'] = theta
            print("Started the theta optimization----------------------------------->checkpoint 2")
            par_theta_obj.optimize(main_dict)
            theta = np.reshape(main_dict['theta'], (temp_theta.shape[0], temp_theta.shape[1]))
            main_dict['theta'] = theta
            print "THETA time :", (time.time() - start_time)
            
            # phi parameter estimation
            start_time = time.time()
            temp_phi = main_dict['phi']
            phi = np.reshape(main_dict['phi'], (1, np.product(main_dict['phi'].shape)))
            main_dict['phi'] = phi
            print("Started the phi optimization----------------------------------->checkpoint 3")
            par_phi_obj.optimize(main_dict)
            phi = np.reshape(main_dict['phi'], (temp_phi.shape[0], temp_phi.shape[1]))
            main_dict['phi'] = phi
            print " PHI time :", (time.time() - start_time)

            # aij estimation
            pdb.set_trace()
            start_time = time.time()
            temp_Aij = main_dict['Aij']
            Aij = np.reshape(main_dict['Aij'], (1, np.product(main_dict['Aij'].shape)))
            main_dict['Aij'] = Aij
            print("Started the Aij optimization----------------------------------->checkpoint 4")
            par_aij_obj.optimize(main_dict)
            Aij = np.reshape(main_dict['Aij'], (temp_Aij.shape[0], temp_Aij.shape[1]))
            main_dict['Aij'] = Aij
            print "AIJ time :", (time.time() - start_time)
            
            pickle.dump(main_dict, open("Iteration/Itr_"+str(iteration)+"_"+str(os.path.basename(checkin_file)[:2])+"_results.p", "wb"))

        pickle.dump(main_dict, open("Iteration/Final_"+str(os.path.basename(checkin_file)[:2])+"_results.p", "wb"))



    def main(self, checkin_file, connections_file, num_community):
        self.estimate_parameters(checkin_file, connections_file, num_community)

if __name__ == '__main__':
    # s = raw_input("number of communities to sample : ")
    # i = raw_input("number of iterations for custom opt routine : ")
    s = sys.argv[3]
    i = sys.argv[4]
    comm = sys.argv[5]

    start_time = time.time()
    parameter_estimation(int(s), int(i), sys.argv[6]).main(sys.argv[1], sys.argv[2], int(comm))

    print "execution time :", (time.time() - start_time)
