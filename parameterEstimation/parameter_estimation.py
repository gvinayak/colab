import numpy as np
import sys, io, time
from simulation.utility import utility
from synthetic_data_processing import synthetic_data_processing
from simulation.kernelParameters import kernelParameters
from parameter_estimation_mu import parameter_estimation_mu
from parameter_estimation_theta import parameter_estimation_theta
from parameter_estimation_aij import parameter_estimation_aij
from parameter_estimation_phi import parameter_estimation_phi
from pre_computation import pre_computation
import pickle

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

    def __init__(self, num_of_samples, number_of_iter):
        self.S = num_of_samples
        self.I = number_of_iter


    def estimate_parameters(self, checkin_file, connections_file, params_file):

        # get data
        (self.events, self.checkins, self.T, self.X, self.Y) = synthetic_data_processing().get_checkins(checkin_file)

        # get real parameters
        (mu_real, Aij_real, pi_real, theta_real, U_real, V_real, M_real) = synthetic_data_processing().get_params(params_file)
        self.U = U_real
        self.V = V_real
        self.M = M_real

        # get graph
        graph = synthetic_data_processing().get_graph(connections_file, self.U)

        print ("graph : ", graph)
        print ("S : ", self.S)

        print ("T : " , self.T)
        print (" X : " , self.X)
        print (" Y : " , self.Y)


        # initialize theta (category to community distribution)

        theta = np.random.uniform(0, 1, (self.M, self.V))
        theta = utility().normalize_2d_array(theta)
        theta_trans = np.array(theta)
        theta_trans = theta.T

        print ("theta : ", theta)


        # initialize phi (posterior distribution of user to community)

        phi = np.random.uniform(0, 1, (self.U, self.M))
        phi = utility().normalize_2d_array(phi)
        print ("phi : ", phi)

        # initialize pi (prior distribution of user to community)

        pi = np.random.uniform(0, 1, (self.U, self.M))
        pi = utility().normalize_2d_array(pi)

        print ("pi : ", pi)


        # initialize mu (base distribution of user)

        N = len(self.events)
        print ("N : ", N)
        mu = np.random.uniform(0, 1, (1, self.U))
        mu = utility().normalize_2d_array(mu)

        print ("mu : ", mu)

        mu_checkin_prop = np.random.uniform(0, 1, (1, self.U))
        for k, v in self.checkins.iteritems():
            # print len(v[0])
            mu_checkin_prop[0, k] = float(len(v[0])) / float(N)

        print ("mu checkin propotional : " , mu_checkin_prop)

        # initialize Aij (user to user influence)

        # Aji = simulate_data().init_influence(graph)
        # Aij = np.array(Aji)
        Aij = np.random.rand(self.U, self.U)

        print ("Aij : ", Aij)

        # learn bandwidth of each user

        bw_start_time = time.time()
        bw = np.empty((self.U))
        bw = [kernelParameters().kde(self.checkins[user][0], self.checkins[user][1]).bandwidth for user in range(0,self.U)]
        # for user in self.checkins.iterkeys():
        #     bw[user] = kernelParameters().kde(self.checkins[user][0], self.checkins[user][1]).bandwidth
        print "bandwidth learn time : ",(time.time() - bw_start_time)
        # pre computation step

        pre_comp_time = time.time()
        # (pre_compute_map, pre_compute_Aij) = pre_computation().pre_compute(self.U, bw, self.events, Aij_real)

        # Changes are made here----------------------------------------------------------------------------------------------------------

        pickle.dump([self.U, bw, self.events, Aij_real], open("pre_compute.p", "wb"))
        (pre_compute_map, pre_compute_Aij) = pre_computation().pre_compute(self.U, bw, self.events, Aij_real)

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

        samples = par_phi_obj.getSamples(pi_real)



        # mu estimation

        # dict = {'mu': mu[0, :], 'Aij': Aij_real, 'phi': pi_real, 'theta': theta_real, 'bw': bw, 'pi': pi_real,
        #         'samples': samples}
        #
        # param_mu = par_mu_obj.optimize(dict)
        #
        # # theta estimation
        #
        # theta = np.reshape(theta, (1, np.product(theta.shape)))
        #
        # dict = {'mu': mu_real[0,:], 'Aij': Aij_real, 'phi': pi_real, 'theta': theta, 'bw': bw, 'pi': pi_real,
        #         'samples': samples}
        #
        # param_theta = par_theta_obj.optimize(dict)
        #
        # # phi parameter estimation

        phi = pi_real
        phi_var = np.random.uniform(0.0,0.3,(self.U,self.M))
        print "phi var : ", phi_var
        phi = phi + phi_var
        phi = utility().normalize_2d_array(phi)

        phi = np.reshape(phi, (1, np.product(phi.shape)))

        dict = {'mu': mu_real[0, :], 'Aij': Aij_real, 'phi': phi, 'theta': theta_real, 'bw': bw, 'pi': pi,
                'samples': samples}

        par_phi_obj.optimize(dict)

        # aij estimation

        # Aij = Aij_real
        # Aij_var = np.random.uniform(0,0.2,(self.U,self.U))
        # Aij = Aij + Aij_var
        # print "Aij with var : ", Aij
        #
        # Aij = np.reshape(Aij, (1, np.product(Aij.shape)))
        #
        # dict = {'mu': mu_real[0, :], 'Aij': Aij, 'phi': pi_real, 'theta': theta_real, 'bw': bw, 'pi': pi,
        #         'samples': samples}
        #
        # param_aij = par_aij_obj.optimize(dict)


    def main(self, checkin_file, connections_file, params_file):

        self.estimate_parameters(checkin_file, connections_file, params_file)



if __name__ == '__main__':
    # s = raw_input("number of communities to sample : ")
    # i = raw_input("number of iterations for custom opt routine : ")
    s = sys.argv[4]
    i = sys.argv[5]
    start_time = time.time()
    parameter_estimation(int(s), int(i)).main(sys.argv[1], sys.argv[2], sys.argv[3])

    print "execution time :", (time.time() - start_time)