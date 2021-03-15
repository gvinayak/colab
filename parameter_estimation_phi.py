import numpy as np
from simulation.hawkes_process import hawkes_process
from simulation.simulate_data import simulate_data
from likelihood import likelihood
import io
from scipy.optimize import minimize
from simulation.utility import utility
from collections import deque
import pickle

class parameter_estimation_phi:
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


    def likelihood_phi(self, args):

        second_term = self.likelihood.second_term(args[0]['phi'], args[0]['theta'], args[0]['pi'])
        fourth_term = self.likelihood.E_q_q_g(args[0]['phi'])
        # print "second term : " , second_term
        # print "fourth term : " , fourth_term
        # print "likelihood : ", (args[0]['first_term'] + second_term - args[0]['K'] - fourth_term)
        return (args[0]['first_term'] + second_term - args[0]['K'] - fourth_term)

    def gradient_phi(self, phi, args):

        """gradient of objective function w.r.t. phi:
            sum_n ( (1/S) * sum( lamda_gs(t_n, x_n, y_n) *  ) ) +
            sum_n sum_m ( log theta_[m,c_n] + log pi[i_n,m] )
        """

        Aij = args[0]['Aij']
        mu = args[0]['mu']
        bw = args[0]['bw']
        pi = args[0]['pi']
        theta = args[0]['theta']
        samples = args[0]['samples']
        phi = np.reshape(phi, (self.U, self.M))
        theta = np.reshape(theta, (self.M, self.V))
        Aij = np.reshape(Aij, (self.U, self.U) )

        # print phi

        n = 0
        gradient_phi = np.zeros((self.U,self.M))
        gradient_phi_theta_pi = np.zeros((self.U,self.M))
        user_community_points = {}

        community_points = dict((key, deque(maxlen=10)) for key in range(0, self.M))

        for event in self.events:
            user = event[0]
            # first summation
            gradient = np.zeros((1,self.M))

            for g in samples[user]:
                if user_community_points.has_key(user):
                    if user_community_points[user].has_key(g):
                        user_community_points[user][g] += 1
                    else:
                        user_community_points[user][g] = 1
                else:
                    user_community_points[user] = {}
                    user_community_points[user][g] = 1

                lamda = hawkes_process().hawkes_intensity(mu[user], Aij[user], community_points[g], self.events[n][1],
                                                          self.events[n][2], self.events[n][3], bw[user])
                grad_phi_q = user_community_points[user][g] / max(0.0001,phi[user, g])
                if (len(community_points[g]) <= 10):
                    community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))
                else:
                    community_points[g].popleft()
                    community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))

                gradient[0,g] = gradient[0,g] + np.log(max(0.0001,lamda)) * grad_phi_q


            gradient_phi[user] = gradient_phi[user] + (1.0 / self.S) * gradient[0]
                # gradient_phi[(user,g)] = gradient_phi[(user,g)] + ((1.0 / self.S) * v)

            # second summation

            c_n = self.events[n][4]

            for m in range(0,self.M):
                gradient_phi_theta_pi[(user,m)] = gradient_phi_theta_pi[(user,m)] + np.log(max(0.0001,theta[(m,c_n)])) + np.log(max(0.0001,pi[user, m]))

            n += 1

        # third summation

        gradient_phi_eq = 1.0 + np.log(phi)

        # for i in range(0, gradient_phi.shape[0]):
        #     for m in range(0, gradient_phi.shape[1]):
        #         # print "gradient_phi[(i,m)] : " + str(gradient_phi[(i, m)])
        #         # print "gradient_phi_theta_pi[(i,m)] : " + str(gradient_phi_theta_pi[(i, m)])
        #         # print "gradient_phi_eq[(i,m)] : "+str(gradient_phi_eq[(i,m)])
        #
        #         gradient_phi[(i,m)] = gradient_phi[(i,m)] + gradient_phi_theta_pi[(i,m)] - gradient_phi_eq[(i,m)]

        gradient_phi = gradient_phi + gradient_phi_theta_pi - gradient_phi_eq

        gradient_phi = np.reshape(gradient_phi, (1, np.product(gradient_phi.shape)))

        # print "grad phi"
        # print gradient_phi[0,:]
        return gradient_phi[0,:]

    def func_phi(self, gamma_phi, *args):
        # print("The value of gamma is ------------------->", gamma_phi.shape)
        temp_phi = []
        for k in range(len(gamma_phi)):
            if(gamma_phi[k] < 700):
                temp_phi.append(np.exp(gamma_phi[k]))
            else:
                temp_phi.append(1.0142320547350045e+304)

        temp_phi = np.asarray(temp_phi)
        phi = temp_phi
        args = list(args)
        args[0]['phi'] = phi
        args = tuple(args)
        f = -1.0 * self.likelihood_phi(args)
        return f

    def grad_phi(self, gamma_phi, *args):
        phi = np.exp(gamma_phi)
        args = list(args)
        args[0]['phi'] = phi
        args = tuple(args)
        grd = self.gradient_phi(phi, args)
        # for i in range(0, len(phi)):
        #     grd[i] = -1.0 * grd[i] * phi[i]
        return -1.0 * grd * phi

    def constraint_phi(self, gamma_phi):
        temp_phi = []
        for k in range(len(gamma_phi)):
            if(gamma_phi[k] < 700):
                temp_phi.append(np.exp(gamma_phi[k]))
            else:
                temp_phi.append(1.0142320547350045e+304)

        temp_phi = np.asarray(temp_phi)
        phi = temp_phi
        
        phi = np.reshape(phi, (self.U, self.M))
        # print ("phi in cons : ", phi)
        return np.sum(phi, 1) - 1

    def getSamples(self, phi):
        phi = np.reshape(phi, (self.U, self.M))
        # print phi
        samples = {}
        # sampling of g's
        for i in range(0, self.U):
            samples[i] = []
            for s in range(0, self.S):
                samples[i].append(simulate_data().sample_multinomial(phi[i]))
        return samples


    def custom_gradient_phi(self, *args):

        G = np.zeros(((self.U*self.M),(self.U*self.M)))
        eta = 1.0
        phis = np.zeros((self.I, self.U * self.M))
        delta_lambda = np.zeros(self.I)
        obj_func_values = []
        phi = args[0]['phi']

        for t in range(0, self.I):

            phis[t] = phi

            if t % 50 == 0:
                print "Iteration: ", t
                print "phi: ", phi

            # new samples

            phi = np.reshape(phi, (self.U, self.M))
            phi = utility().normalize_2d_array(phi)
            phi = np.reshape(phi, (1, self.U * self.M))

            samples = self.getSamples(phi)

            # updated arguments
            args = list(args)
            args[0]['samples'] = samples
            args = tuple(args)

            # terminating condition
            obj_fun = self.likelihood_phi(args)
            # print "obj fun : " , obj_fun
            obj_func_values.append(obj_fun)
            if t != 0 and (np.abs(obj_func_values[t] - obj_func_values[t - 1])) < 0.01:
                print obj_func_values[t]
                print obj_func_values[t - 1]
                print "phi : ", phi
                break

            cons = [{'type': 'eq', 'fun': self.constraint_phi}]

            gamma_phi = np.log(phi)

            res = minimize(self.func_phi, x0=gamma_phi, args=args, method = 'SLSQP',jac= self.grad_phi, constraints=cons, options={'maxiter' : 1, 'eps' : 3.0})

            phi_new = np.exp(res.x)

            # print ("phi new" , phi_new)

            phi = phi_new

        print "obj func value : " , (obj_func_values[len(obj_func_values)-1])
        print "final phi : ", phi
        
        pickle.dump([self.U,self.M,phi],open("estimated_phi.p", "wb"))
        return phi_new

    def optimize(self, dict):
        first_term = self.likelihood.first_term(dict['mu'], dict['samples'], dict['Aij'], dict['bw'])
        K = self.likelihood.integral_full_pre_computed(dict['mu'], dict['bw'])

        print "first term : " , first_term

        print "K : ", K

        # print(" phi : ",dict['phi'])

        args = {'theta' : dict['theta'], 'phi' : dict['phi'], 'pi': dict['pi'], 'mu' : dict['mu'], 'samples' : dict['samples'], 'Aij' : dict['Aij'], 'bw' : dict['bw'], 'first_term' : first_term, 'K' : K}

        x = self.custom_gradient_phi(args)
        dict['phi'] = x
        # print minimize(self.func_theta, x0=gamma_theta, jac = self.grad_theta, args=(args), options={'disp': True}, constraints=cons)
