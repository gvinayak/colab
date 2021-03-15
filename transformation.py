import math
import numpy as np
from simulation.hawkes_process import hawkes_process
from simulation.simulate_data import simulate_data
from scipy.special import erf

class transformation:

    S = 0
    M = 0
    U = 0
    V = 0
    T = 0
    X = 0
    Y = 0

    # event is a tuple of (user, time, lat, lon, category, community)
    events = []
    checkins = {}


    def __init__(self, t, x, y, m, u, v, s, e):
        self.T = t
        self.X = x
        self.Y = y
        self.M = m
        self.U = u
        self.V = v
        self.S = s
        self.events = e

    def integral(self, gamma, Aij, bw):
        """survival thing:
           TXY * sum_i(mu_i) +
           sum_i sum_k A[i_k,i] sum_n ( np.exp(- nu(t_n - t_k) ) (erf((x[n],y[n] - x[k],y[k])/(sqrt(2*h)) - erf((x[n-1],y[n-1] - x[k],y[k])/(sqrt(2*h)) ) )
                """

        nu = 0.01
        N = len(self.events)
        K = 0

        for i in range(0, self.U):
            for k in range(0,N):
                ik = self.events[k][0]
                s = 0.0
                # print "k : "+str(k)+" N : "+str(N)
                for n in range(k, N):
                    prod1 = np.exp(nu * (self.events[n][1] - self.events[k][1])) - np.exp(nu * (self.events[n-1][1] - self.events[k][1]))
                    prod2 = erf((self.events[n][2] - self.events[k][2])/ math.sqrt(2 * bw[i])) - erf((self.events[n-1][2] - self.events[k][2])/ math.sqrt(2 * bw[i]))
                    prod3 = erf((self.events[n][3] - self.events[k][3]) / math.sqrt(2 * bw[i])) - erf((self.events[n - 1][3] - self.events[k][3]) / math.sqrt(2 * bw[i]))
                    # print "prod1 : "+str(prod1)
                    # print "prod2 : "+str(prod2)
                    # print "prod3 : "+str(prod3)
                    s = s + ((prod1 * prod2 * prod3) / (2 * np.pi * bw[i]))
                # print "i : "+str(i)+" ik : "+str(ik)+" Aij[i,ik] : "+str(Aij[(i,ik)])+" s : "+str(s)
                K = K + (Aij[(i,ik)] * s)

        gamma = np.exp(gamma)
        return sum(gamma) * self.T * self.X * self.Y + K

    def E_q_q_g(self, phi):

        s = 0.0
        for i in range(0,self.U):
            for m in range(0,self.M):
                s = s + (phi[(i,m)] * math.log(phi[(i,m)]))
        return s

    def obj_func(self, gamma, args):
        """Likelihood expression:
        sum_n ( (1/S) * sum( log (lamda_gs(t_n, x_n, y_n) ) ) ) +
        sum_n sum_m phi[i,g]( log theta_[m,c_n] + log pi[i_n,m] ) -
        TXY * \sum_i(mu_i) + sum_k A[i_k,i] sum_n ( np.exp(- nu(t_n - t_k) ) (erf((x[n],y[n] - x[k],y[k])/(sqrt(2*h)) - erf((x[n-1],y[n-1] - x[k],y[k])/(sqrt(2*h)) ) )
        - sum_i sum_g (q(g) log q(g))"""

        aij = args[0]['aij']
        phi = args[0]['phi']
        theta = args[0]['theta']
        bw = args[0]['bw']
        pi = args[0]['pi']
        samples = args[0]['samples']

        likelihood = 0
        first_sum = []
        second_sum = []
        community_points = {}

        for community in range(0, self.M):
            community_points[community] = []

        n = 0
        for event in self.events:
            user = event[0]
            # first summation
            sampled_lamda = []
            for g in samples[user]:
                #g = simulate_data().sample_multinomial(phi[user])
                community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))
                lamda = hawkes_process().hawkes_intensity(np.exp(gamma[user]), aij[user], community_points[g], self.events[n][1], self.events[n][2], self.events[n][3], bw[user])
                if lamda != 0.0:
                    log_lamda = math.log(lamda)
                    sampled_lamda.append(log_lamda)

            first_sum.append(sum(sampled_lamda)/ self.S)

            # second summation
            c_n = self.events[n][4]
            expectation_m = []
            for m in range(0, self.M):
                expectation_m.append(phi[user, m] * (math.log(theta[c_n, m]) + math.log(pi[user, m])))
            second_sum.append(sum(expectation_m))

            n += 1

        third_sum = self.integral(gamma, aij, bw)
        # print "integral : " + str(third_sum)

        fourth_sum = self.E_q_q_g(phi)
        # print "e_q_q_g : "+str(fourth_sum)

        likelihood = sum(first_sum) + sum(second_sum) - third_sum - fourth_sum
        # print "gamma : " + str(gamma)
        return likelihood


    def gradient_gamma(self, gamma, args):
        """gradient of objective function w.r.t. mu:
               sum_n ( (1/S) * sum( 1 / lamda_gs(t_n, x_n, y_n) ) ) - T*X*Y
                    """

        phi = args[0]['phi']
        aij = args[0]['aij']
        bw = args[0]['bw']
        samples = args[0]['samples']

        # print phi
        # print aij
        # print bw
        # print gamma

        gradient_user = {}
        for i in range(0, self.U):
            gradient_user[i] = []

        community_points = {}
        for community in range(0, self.M):
            community_points[community] = []

        n = 0
        for event in self.events:
            user = event[0]
            sampled_lamda = []
            for g in samples[user]:
                #g = simulate_data().sample_multinomial(phi[user])
                lamda = hawkes_process().hawkes_intensity(np.exp(gamma[user]), aij[user], community_points[g], self.events[n][1], self.events[n][2], self.events[n][3], bw[user])
                if lamda != 0.0:
                    sampled_lamda.append(1.0 / lamda)
                community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))
            gradient_user[user].append((1 / self.S) * sum(sampled_lamda))
            n += 1

        gradient = []
        for i in range(0, self.U):
            # print "gradient of user : "+str(i)+" is : "+str(sum(gradient_user[i]))
            # print sum(gradient_user[i]) + (self.T * self.X * self.Y)
            gradient.append((sum(gradient_user[i]) - (self.T * self.X * self.Y)) * np.exp(gamma[i]))

        gradient = np.array(gradient)
        # gradient = gradient/sum(gradient)
        print gradient
        return gradient