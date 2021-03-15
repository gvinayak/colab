import numpy as np
from likelihood import likelihood
import io
from scipy.optimize import minimize
from simulation.hawkes_process import hawkes_process
from collections import deque

class parameter_estimation_aij:
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
    const = 5.0

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

    def likelihood_aij(self, args):
        aij = args[0]['Aij']
        first_term = self.likelihood.first_term(args[0]['mu'], args[0]['samples'], aij, args[0]['bw'])
        K = self.likelihood.integral(args[0]['mu'], aij, args[0]['bw'])
        # print "first term : " , first_term
        l2_reg = self.const * np.sum(aij**2)
        # print "fourth term : " , K
        return (first_term + args[0]['second_term'] - K - args[0]['fourth_term'] - l2_reg)

    def constraint_Aij(self, Aij):
        # Aij = np.exp(gamma_Aij)
        Aij = np.reshape(Aij, (self.U, self.U))
        return np.sum(Aij, 1) - 1

    def func_Aij(self, Aij, *args):
        # Aij = np.exp(gamma_Aij)
        args = list(args)
        args[0]['Aij'] = Aij
        args = tuple(args)
        f = -1.0 * self.likelihood_aij(args)
        return f

    def grad_Aij(self, Aij, *args):
        # Aij = np.exp(gamma_Aij)
        args = list(args)
        args[0]['Aij'] = Aij
        args = tuple(args)
        g = self.gradient_A_ij(Aij, args)
        g = -1.0 * g * Aij
        # print "grad :", g
        return g[0,:]

    def grad_intensity_aij(self, lamda, ai, points, t, x, y, h):

        """gradient of hawkes intensity w.r.t. Aij:
             sum ( ( np.exp(- nu (t- t_k) - ((x - x_k)^2 + (y- y_k)^2)/ (2*h) ) for t_k in points if t_kk<=t)
        """

        nu = 0.01  # time decay parameter
        p = np.zeros((self.U))
        # p = {}

        for entry in points: # point is a tuple <t,x,y,j>
            if (entry[0] < t):
                j = entry[3]
                # if(p.has_key(j)):
                p[j] = p[j] + (np.exp(nu * (entry[0] - t) - (np.square(x - entry[1]) + np.square(y - entry[2])) / (2 * h))
                         ) / (2 * np.pi * h * lamda)
                # else:
                #     p[j] = (np.exp(nu * (entry[0] - t) - (np.square(x - entry[1]) + np.square(y - entry[2])) / (2 * h))
                #          ) / (2 * np.pi * h * lamda)

        # p = p / lamda
        # for j,v in p.iteritems(): # gradient w.r.t. jth connection of i
        #     gradient[j] = (sum(v) / lamda)

        return p

    def gradient_A_ij(self, Aij, args):

        """gradient of objective function w.r.t. aij:
            sum_n ( (1/S) * sum(lamda_gs(t_n, x_n, y_n) * (sum_g sum_tk<t kappa_fxn )) )  I(ij) -
            sum_n ( integral ) I(ij)
        """

        mu = args[0]['mu']
        bw = args[0]['bw']
        samples = args[0]['samples']

        Aij = np.reshape(Aij, (self.U, self.U))

        gradient_aij = np.zeros((self.U, self.U))

        community_points = dict((key, deque(maxlen=10)) for key in range(0, self.M))


        for event in self.events:
            user = event[0]
            gradient = []
            for g in samples[user]:

                lamda = hawkes_process().hawkes_intensity(mu[user], Aij[user], community_points[g], event[1],
                                                          event[2], event[3], bw[user])

                gradient.append(self.grad_intensity_aij(lamda, Aij[user], community_points[g], event[1],
                                            event[2], event[3], bw[user]))
                if (len(community_points[g]) <= 10):
                    community_points[g].append((event[1], event[2], event[3], user))
                else:
                    community_points[g].popleft()
                    community_points[g].append((event[1], event[2], event[3], user))

            for k in range(0,len(gradient)):
                gradient_aij[user] = gradient_aij[user] + ((1 / self.S) * gradient[k][0])
                # for (j,v) in gradient[k].iteritems():
                #     gradient_aij[(user,j)] = gradient_aij[(user,j)] + ((1 / self.S) * v)


        (K, Kij) = self.likelihood.integral_full(Aij, bw)

        # final_gradient = []
        # for i in range(0, self.U):
        #     for j in range(0,self.U):
        #         final_gradient.append(gradient_aij[(i,j)] - Kij[(i,j)])
        #
        # final_gradient = np.array(final_gradient)

        gradient_aij = gradient_aij - Kij
        gradient_aij = np.reshape(gradient_aij, (1, np.product(gradient_aij.shape)))

        return gradient_aij

    def custom_gradient_Aij(self, Aij, *args):


            # G = np.zeros(((self.U * self.U), (self.U * self.U)))
            # eta = 1.0

            fixed_gradient = 0.1
            Aijs = np.zeros((self.I, self.U * self.U))

            obj_func_values = []

            cons = [{'type': 'eq', 'fun': self.constraint_Aij}]

            bnds = ()
            for i in range(0, np.product(Aij.shape)):
                bnds = bnds + ((0, np.Infinity),)

            for t in range(0, self.I):

                Aijs[t] = Aij

                if t % 50 == 0:
                    print "Iteration: ", t
                    print "Aij: ", Aij

                # updated arguments
                args = list(args)
                args[0]['Aij'] = Aij
                args = tuple(args)

                # terminating condition
                obj_func_values.append(self.likelihood_aij(args))
                if t != 0 and (np.abs(obj_func_values[t] - obj_func_values[t - 1])) < 0.01:
                    print obj_func_values[t]
                    print obj_func_values[t - 1]
                    break



                Aij_new = minimize(self.func_Aij, x0=Aij[0, :], jac=self.grad_Aij, args=(args), options={'maxiter': 1, 'eps': 1.0}, bounds=bnds, constraints=cons).x

                # call to gradient
                # grad_estimate = self.gradient_A_ij(Aij, args)
                #
                # for i in range(0, len(grad_estimate)):
                #     grad_estimate[i] = grad_estimate[i] * Aij[i]
                #
                # Aij_new = Aij + (fixed_gradient * grad_estimate)

                print Aij_new

                Aij = Aij_new

            fout = io.open("Aij"+str(self.U), "w")

            for i in Aij:
                fout.write(unicode(i)+u"\n")

            fout.close()

    def optimize(self, dict):

        second_term = self.likelihood.second_term(dict['phi'], dict['theta'], dict['pi'])
        fourth_term = self.likelihood.E_q_q_g(dict['phi'])
        args = {'mu' : dict['mu'], 'samples' : dict['samples'], 'Aij' : dict['Aij'], 'bw' : dict['bw'], 'second_term' : second_term, 'fourth_term' : fourth_term}

        # gamma_aij = np.log(dict['Aij'])
        # print "gamma_aij : ", gamma_aij

        Aij = dict['Aij']

        self.custom_gradient_Aij(Aij, args)

        # bnds = ()
        # for i in range(0, np.product(Aij.shape)):
        #     bnds = bnds + ((0,np.Infinity),)
        #
        # print bnds
        #
        # cons = [{'type': 'eq', 'fun': self.constraint_Aij}]
        #
        # print minimize(self.func_Aij, x0= Aij[0,:], jac = self.grad_Aij, args=(args), options={'disp': True, 'maxiter' : 100, 'eps': 1.0, 'maxfev': 1}, bounds=bnds, constraints=cons)

