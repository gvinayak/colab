import numpy as np
from simulation.hawkes_process import hawkes_process
from scipy.special import erf
from collections import deque

class likelihood:
    """Likelihood expression:
            sum_n ( (1/S) * sum( log (lamda_gs(t_n, x_n, y_n) ) ) )
            + sum_n sum_m phi[i,g]( log theta_[m,c_n] + log pi[i_n,m] )
            - TXY * \sum_i(mu_i) - sum_k A[i_k,i] sum_n ( np.exp(- nu(t_n - t_k) ) (erf((x[n],y[n] - x[k],y[k])/(sqrt(2*h)) - erf((x[n-1],y[n-1] - x[k],y[k])/(sqrt(2*h)) ) )
            - sum_i sum_g (q(g) log q(g))
            """
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
    pre_compute_map = np.empty((U, len(events)))
    pre_compute_Aij = np.empty((U, len(events)))


    def __init__(self, num_of_samples, number_of_communities, number_of_users, number_of_categories, number_of_iter, T,
                 X, Y, events, checkins, pre_compute_map, pre_compute_Aij):
        self.S = num_of_samples
        self.M = number_of_communities
        self.U = number_of_users
        self.V = number_of_categories
        self.I = number_of_iter
        self.T = T
        self.X = X
        self.Y = Y
        self.events = events
        self.checkins = checkins
        self.pre_compute_map = pre_compute_map
        self.pre_compute_Aij = pre_compute_Aij

    def integral_pre_computed(self, bw):
        nu = -1 * 0.01
        N = len(self.events)
        K = 0

        for i in range(0, self.U):
            K = K + self.pre_compute_map[i] * self.pre_compute_Aij[i]

        return np.sum(K)

    def integral_full_pre_computed(self, mu, bw):
        return (sum(mu) * self.T * self.X * self.Y) + self.integral_pre_computed(bw)

    def integral_full(self, Aij, bw):
        """survival thing:
        TXY * sum_i(mu_i) +
        sum_i sum_k A[i_k,i] sum_n ( - np.exp(- nu(t_n - t_k) + np.exp(-nu(t_n-1 - t_k))) (erf((x[n],y[n] - x[k],y[k])/(sqrt(2*h)) - erf((x[n-1],y[n-1] - x[k],y[k])/(sqrt(2*h)) ) )
            """
        Aij = np.reshape(Aij, (self.U, self.U))
        nu = -1 * 0.01
        N = len(self.events)
        K = 0

        # initialize sum term w.r.t. each i and j

        Kij = np.zeros((self.U, self.U))

        # Kij = {}
        # for i in range(0, self.U):
        #     for j in range(0, self.U):
        #         pair = (i,j)
        #         Kij[pair] = 0

        for i in range(0, self.U):
            for k in range(0, N - 1):
                ik = self.events[k][0]
                s = self.pre_compute_map[(i,k)]
                K = K + (Aij[(i, ik)] * s)
                Kij[(i, ik)] = Kij[(i, ik)] + (Aij[(i, ik)] * s)

        return (K, Kij)


    def integral(self, mu, Aij, bw):
        (K, Kij) = self.integral_full(Aij, bw)
        # print "mu * TXY : ", (sum(mu) * self.T * self.X * self.Y)
        # print "second last term : " , K
        return (sum(mu) * self.T * self.X * self.Y) + K


    def E_q_q_g(self, phi):
        phi = np.reshape(phi , (1,np.product(phi.shape)))
        s = phi * np.log(phi)
        # print "E_q_q_g : ", s.sum()
        return s.sum()

    def first_term(self, mu, samples, Aij, bw):

        # mu = args[0]['mu']
        # samples =  args[0]['samples']
        # Aij =  args[0]['Aij']
        # bw = args[0]['bw']

        Aij = np.reshape(Aij, (self.U, self.U))

        community_points = dict((key, deque(maxlen=10)) for key in range(0,self.M))

        first_sum = []
        n = 0
        for event in self.events:
            user = event[0]
            # first summation
            sampled_lamda = []
            for g in samples[user]:
                # g = simulate_data().sample_multinomial(phi[user])
                if (mu[user] > 0):
                    lamda = hawkes_process().hawkes_intensity(mu[user], Aij[user], community_points[g],
                                                          self.events[n][1],
                                                          self.events[n][2], self.events[n][3], bw[user])

                    log_lamda = np.log(max(0.0001, lamda))
                    # print "log lamda : ", log_lamda
                    sampled_lamda.append(log_lamda)
                    if(len(community_points[g]) <= 10):
                        community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))
                    else:
                        community_points[g].popleft()
                        community_points[g].append((self.events[n][1], self.events[n][2], self.events[n][3], user))
                    # print " community_points : ", len(community_points[g])
            first_sum.append((1.0 / self.S) * sum(sampled_lamda))

            n +=1

        return sum(first_sum)

    def second_term(self, phi, theta, pi):

        theta = np.reshape(theta, (self.M, self.V))

        phi = np.reshape(phi, (self.U, self.M))

        second_sum = []

        for event in self.events:
            user = event[0]
            c_n = event[4]

            # expectation_m = phi[user] * (np.log(theta_trans[c_n]) + np.log(pi[user]))

            expectation_m = []
            for m in range(0, self.M):
                expectation_m.append(
                    (phi[user, m] * (np.log(max(0.0001, theta[m, c_n])) + np.log(max(0.0001, pi[user, m])))))


            second_sum.append(sum(expectation_m))
        # print sum(second_sum)
        return sum(second_sum)

    def third_term(self, args, result_queue):
        result_queue.put(np.sum(args[0]['mu']) * self.T * self.X * self.Y + args[0]['K'])

    # def obj_func(self, args):
    #
    #     phi = args[0]['phi']
    #     Aij = args[0]['Aij']
    #     mu = args[0]['mu']
    #     bw = args[0]['bw']
    #     pi = args[0]['pi']
    #     theta = args[0]['theta']
    #     samples = args[0]['samples']
    #
    #     likelihood = 0
    #
    #
    #     likelihood = sum(first_sum) + sum(second_sum) - third_sum - fourth_sum
    #     # theta = np.reshape(theta, (1, np.product(theta.shape)))
    #     print "likelihood : " + str(likelihood)
    #     return likelihood