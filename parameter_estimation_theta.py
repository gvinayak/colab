import numpy as np
from likelihood import likelihood
import io
from scipy.optimize import minimize


class parameter_estimation_theta:
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


    def likelihood_theta(self, args):

        second_term = self.likelihood.second_term(args[0]['phi'], args[0]['theta'], args[0]['pi'])
        # print "T : ", self.T
        # print "X : ", self.X
        # print "Y : ", self.Y
        # print "mu : ", args[0]['mu']
        # print "second term : " , second_term
        # print "third term : " , third_term
        return (args[0]['first_term'] + second_term - args[0]['K'] - args[0]['fourth_term'])

    def constraint_theta(self, theta):
        theta = np.reshape(theta, (self.M, self.V))
        return np.sum(theta, 1) - 1

    def constraint_gamma_theta(self, gamma_theta):
        gamma_theta = np.reshape(gamma_theta, (self.M, self.V))
        # gamma_theta = np.reshape(gamma_theta, (self.V, self.M))
        # gamma_theta_trans = np.array(gamma_theta)
        # gamma_theta_trans = gamma_theta.T
        theta = np.exp(gamma_theta)
        return np.sum(theta, 1) - 1

    def func_theta(self, gamma_theta, *args):
        args = list(args)
        theta = np.exp(gamma_theta)
        args[0]['theta'] = theta
        args = tuple(args)
        f = -1.0 * self.likelihood_theta(args)
        return f

    def grad_theta(self, gamma_theta, *args):
        args = list(args)
        theta = np.exp(gamma_theta)
        args[0]['theta'] = theta
        args = tuple(args)
        grad_theta = self.gradient_theta(theta, args)
        # for i in range(0, len(grad_theta)):
        #     grad_theta[i] = -1.0 * grad_theta[i] * theta[i]
        return -1.0 * grad_theta * theta

    def gradient_theta(self, theta, args):
        """gradient of objective function w.r.t. theta:
           sum_n ( phi_i,g_n * (sum_m (1 / theta_g_m,c_n) ) )
        """
        theta = np.reshape(theta, (self.M, self.V))
        # theta_trans = np.array(theta)
        # theta_trans = theta.T

        phi = args[0]['phi']

        gradient = np.zeros((self.M, self.V))

        for event in self.events:
            c_n = event[4]
            # gradient[c_n] = gradient[c_n] + phi[event[0]] / theta_trans[c_n]
            for m in range(0,self.M):
                gradient[(m, c_n)] = gradient[(m,c_n)] + (phi[(event[0], m)] / theta[(m,c_n)])


        gradient = np.reshape(gradient, (1, np.product(gradient.shape)))

        # print gradient[0,:]
        return gradient[0,:]

    def optimize(self, dict):
        first_term = self.likelihood.first_term(dict['mu'], dict['samples'], dict['Aij'], dict['bw'])
        fourth_term = self.likelihood.E_q_q_g(dict['phi'])
        K = self.likelihood.integral_full_pre_computed(dict['mu'], dict['bw'])

        print "first term : " , first_term

        print "fourth term : " , fourth_term

        print "K : ", K

        args = {'theta' : dict['theta'], 'phi' : dict['phi'], 'pi': dict['pi'], 'mu' : dict['mu'], 'samples' : dict['samples'], 'Aij' : dict['Aij'], 'bw' : dict['bw'], 'first_term' : first_term, 'fourth_term' : fourth_term, 'K' : K}

        gamma_theta = np.log(dict['theta'])
        print "gamma_theta : ", gamma_theta

        cons = [{'type': 'eq', 'fun': self.constraint_gamma_theta}]

        estimated_theta = np.exp(minimize(self.func_theta, x0=gamma_theta, jac = self.grad_theta, args=(args), options={'disp': True, 'maxiter' : 100}, constraints=cons).x)

        dict['theta'] = estimated_theta

        estimated_theta = np.reshape(estimated_theta, (self.M, self.V))

        print estimated_theta
