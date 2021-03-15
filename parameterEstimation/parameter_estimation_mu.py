import numpy as np
from datetime import datetime
from simulation.hawkes_process import hawkes_process
from likelihood import likelihood
from scipy.optimize import minimize
from simulation.utility import utility
from collections import deque
from multiprocessing import Process, Queue

class parameter_estimation_mu:
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

    likelihood = 0

    def __init__(self, S, M, U, V, I, T, X, Y, events, checkins, pre_compute_map, pre_compute_Aij):
        self.S = S
        self.M = M
        self.U = U
        self.V = V
        self.I = I
        self.T = T
        self.X = X
        self.Y = Y
        self.events = events
        self.checkins = checkins
        self.likelihood = likelihood(S, M, U, V, I, T, X, Y, events, checkins, pre_compute_map, pre_compute_Aij)

    def constraint_mu(self, mu):
        return np.sum(mu) - 1

    def constraint_gamma_mu(self, gamma_mu):
        mu = np.exp(gamma_mu)
        return np.sum(mu) - 1

    def likelihood_mu(self, args):
        first_term = self.likelihood.first_term(args[0]['mu'], args[0]['samples'], args[0]['Aij'], args[0]['bw'])
        third_term = np.sum(args[0]['mu']) * self.T * self.X * self.Y + args[0]['K']
        # print "third term : " , third_term
        # print "T : ", self.T
        # print "X : ", self.X
        # print "Y : ", self.Y
        # print "mu : ", args[0]['mu']
        # print "first term : " , first_term
        # print "third term : " , third_term
        return (first_term, third_term, first_term + args[0]['st'] - third_term - args[0]['ft'])

    def func_mu(self, gamma_mu, *args):
        mu = np.exp(gamma_mu)
        args = list(args)
        args[0]['mu'] = mu
        args = tuple(args)
        (ft, tt, ld) = self.likelihood_mu(args)
        f = -1.0 * ld
        return f

    def grad_mu(self, gamma_mu, *args):
        mu = np.exp(gamma_mu)
        args = list(args)
        args[0]['mu'] = mu
        args = tuple(args)
        grad_mu = self.gradient_mu(mu, args)
        # # for i in range(0,len(grad_mu)):
        # #     grad_mu[i] = -1.0 * grad_mu[i] * mu[i]
        #
        # print "grad mu : ", -1.0 * grad_mu * mu
        return -1.0 * grad_mu * mu

    def gradient_mu(self, mu, args):

        """gradient of objective function w.r.t. mu:
           sum_n ( (1/S) * sum( 1 / lamda_gs(t_n, x_n, y_n) ) ) - T*X*Y
        """

        aij = args[0]['Aij']
        bw = args[0]['bw']
        samples = args[0]['samples']

        gradient_user = np.zeros((self.U, 1))
        # gradient_user = {}
        # for i in range(0, self.U):
        #     gradient_user[i] = []

        community_points = dict((key, deque(maxlen=10)) for key in range(0, self.M))

        for event in self.events:
            user = event[0]
            s = 0.0
            for g in samples[user]:
                # g = simulate_data().sample_multinomial(phi[user])
                lamda = hawkes_process().hawkes_intensity(mu[user], aij[user], community_points[g],
                                                              event[1], event[2], event[3], bw[user])
                s = s + (1.0 / max(0.0001,lamda))

                if (len(community_points[g]) <= 10):
                    community_points[g].append((event[1], event[2], event[3], user))
                else:
                    community_points[g].popleft()
                    community_points[g].append((event[1], event[2], event[3], user))

            gradient_user[user] = gradient_user[user] + ((1.0 / self.S) * s)


        gradient = np.zeros((self.U,1))
        gradient = gradient + gradient_user - (self.T * self.X * self.Y)

        # print gradient.T
        return gradient.T[0,:]

    def optimize(self, dict):

        second_term = self.likelihood.second_term(dict['phi'], dict['theta'], dict['pi'])
        fourth_term = self.likelihood.E_q_q_g(dict['phi'])
        K= self.likelihood.integral_pre_computed(dict['bw'])
        args = {'mu' : dict['mu'], 'samples' : dict['samples'], 'Aij' : dict['Aij'], 'bw' : dict['bw'], 'st' : second_term, 'ft' : fourth_term, 'K' : K}
        start_time = datetime.now()
        gamma_mu = np.log(dict['mu'])
        print "gamma_mu : ", gamma_mu
        end_time = datetime.now()
        print('Duration log op : {}'.format(end_time - start_time))

        start_time = datetime.now()
        estimated_mu = np.exp(minimize(self.func_mu, x0=gamma_mu, jac = self.grad_mu, args=(args), options={'disp': True, 'maxiter' : 100}).x)
        estimated_mu = np.reshape(estimated_mu,(1, self.U))
        estimated_mu = utility().normalize_2d_array(estimated_mu)
        print estimated_mu
        end_time = datetime.now()
        print 'minimize operation duration : {} '.format(end_time - start_time)