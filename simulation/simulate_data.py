import sys
import random
import io
import numpy as np
from user_checkin_dist import user_checkin_dist
from hawkes_process import hawkes_process
from kernelParameters import kernelParameters
from datetime import datetime

class simulate_data(object):

    n = 1000
    u = 100
    m = 10
    c = 20

    def __init__(self, num_of_users = 10, num_of_communities = 2, num_of_categories=4):
        self.u = num_of_users
        self.m = num_of_communities
        self.c = num_of_categories

    def generate_user_checkins(self, mean, var):
        user_checkin_count = [int(i) for i in user_checkin_dist().gamma_random_sample(mean, var, self.u)]
        self.n = sum(user_checkin_count)
        return user_checkin_count

    def generate_Graph(self, connections):
        adjacency = np.random.randint(0, 2, (self.u, self.u))
        fin = io.open(connections, "w", encoding="utf-8")

        for i in range(0, adjacency.shape[0]):
            for j in range(0, adjacency.shape[1]):
                if(adjacency[(i,j)] == 1):
                    fin.write(unicode(i)+u","+unicode(j))
                    fin.write(u"\n")
        fin.close()
        return adjacency

    def get_user_kde_estimate(self, kde_file, venues_file):
        userTrainingData = kernelParameters().get_training_data(kde_file, venues_file)
        kde_user_estimate = kernelParameters().get_user_parameters(userTrainingData)
        return kde_user_estimate

    def generate_region(self, kde_file, venues_file, user_points):
        discrete_points = []
        bandwidth_users = []
        kde_user_estimate = self.get_user_kde_estimate(kde_file, venues_file)

        for i in range(0,self.u):
            KDE = kde_user_estimate[random.choice(kde_user_estimate.keys())]
            discrete_points.append(kernelParameters().sample_kde(KDE,user_points))
            bandwidth_users.append(KDE.bandwidth)
        return discrete_points, bandwidth_users


    def init_influence(self, graph):
        A_ij = np.zeros(shape=graph.shape)
        links = np.sum(graph, axis=1).tolist()
        for i in range(0,len(links)):
            for j in range(0,graph.shape[1]):
                if(links[i] != 0.0):
                    A_ij[i,j] = (np.abs(graph[i,j])/float(links[i]))
                else:
                    A_ij[i,j] = 0.0
        return A_ij

    def init_intensity(self, user_checkin_count):
        mu = np.empty((0))
        for checkin_count in user_checkin_count:
            mu = np.append(mu, checkin_count/float(self.n))
        return mu

    def update_intensity(self, lamda, i, m_i):
        lamda[i] = m_i
        s = np.sum(lamda)
        lamda = np.divide(lamda,s)
        return lamda

    def sample_multinomial(self, probs):
        i = 0
        for user in np.random.multinomial(1, probs):
            if user == 1:
                return i
            i += 1
        return None

    def sample_dirichlet(self,i,j):
        alphas = 2 ** np.random.randint(0, 4, size=(i,j))
        np.random.seed(1234)
        d = np.empty(alphas.shape)
        for k in range(len(alphas)):
            d[k] = np.random.dirichlet(alphas[k])
        return d

    def user_initial_xy(self, discrete_points):
        initital_xy = []
        for i in range(0,self.u):
            initital_xy.append(discrete_points[i][random.randint(0, discrete_points[i].shape[0]-1)])
        return initital_xy

    def  generate_data(self, mean, var, kde_file, venues_file, candidate_points):

        # file to write parameters of synthetic data
        fout_synthetic_data_parameters = io.open(kde_file + "synthetic_data_parameters", "w", encoding="utf-8")

        # user to checkin count
        user_checkin_count = self.generate_user_checkins(mean, var)
        # print "each user's checkin count : "
        # print user_checkin_count

        # user generate discrete points
        discrete_points, user_bandwidths = self.generate_region(kde_file, venues_file, candidate_points)
        # print "discrete points : "
        # print discrete_points

        # graph
        graph = self.generate_Graph(kde_file+"synthetic_connections") # social connections generated
        # print "graph : "
        # print graph

        # influence matrix
        A_ji = self.init_influence(graph) # initialize influence matrix
        A_ij = np.array(A_ji)
        A_ij = A_ji.T

        fout_synthetic_data_parameters.write(u"U="+unicode(self.u))
        fout_synthetic_data_parameters.write(u"\n")
        fout_synthetic_data_parameters.write(u"V="+unicode(self.c))
        fout_synthetic_data_parameters.write(u"\n")
        fout_synthetic_data_parameters.write(u"M="+unicode(self.m))
        fout_synthetic_data_parameters.write(u"\n")

        fout_synthetic_data_parameters.write(u"Aij=")
        for i in range(0,A_ij.shape[0]):
            for j in range(0,A_ij.shape[1]):
                fout_synthetic_data_parameters.write(unicode(A_ij[(i,j)])+u",")
        fout_synthetic_data_parameters.write(u"\n")

        # print "influence matrix : "
        # print A_ji
        # print "transpose : "
        # print A_ij

        # user to community distribution
        pi = self.sample_dirichlet(self.u,self.m)

        fout_synthetic_data_parameters.write(u"pi=")
        for i in range(0,pi.shape[0]):
            for j in range(0,pi.shape[1]):
                fout_synthetic_data_parameters.write(unicode(pi[(i,j)])+u",")
        fout_synthetic_data_parameters.write(u"\n")

        # print "user-community matrix : "
        # print pi

        # category to community distribution
        theta = self.sample_dirichlet(self.m,self.c)

        fout_synthetic_data_parameters.write(u"theta=")
        for i in range(0,theta.shape[0]):
            for j in range(0,theta.shape[1]):
                fout_synthetic_data_parameters.write(unicode(theta[(i,j)])+u",")
        fout_synthetic_data_parameters.write(u"\n")

        # print "community-category matrix : "
        # print theta

        # base intensity
        mu = self.init_intensity(user_checkin_count)

        fout_synthetic_data_parameters.write(u"mu=")
        for i in range(0,mu.shape[0]):
            fout_synthetic_data_parameters.write(unicode(mu[(i)])+u",")
        fout_synthetic_data_parameters.write(u"\n")

        # print "base intensity of each user : "
        # print mu

        # initialize
        t = 0
        x = 0
        y = 0
        init_xy = self.user_initial_xy(discrete_points)
        # print "initial x y : "
        # print  init_xy
        lamda = np.empty_like(mu)
        lamda[:] = mu
        community_points = {}
        for community in range(0,self.m):
            community_points[community] = []

        all_samples = []
        prev_t = t

        # populate (t,x,y,c,g)
        # print self.n

        for n in range(0,self.n):
            # print "n : "+str(n)
            prev_t = t
            # sample user i
            i = self.sample_multinomial(lamda)
            # print "user : "+str(i)
            # print "lambda_i : "+str(lamda[i])

            # sample g from pi_i
            g = self.sample_multinomial(pi[i,:])
            # print "pi_i,g : "+str(pi[i,g])

            if t == 0:
                x = init_xy[i][0]
                y = init_xy[i][1]

            # sample t from lambda_i
            (t,x,y,m_i) = hawkes_process().generate_point(t, x, y, mu[i], A_ij[i,:], self.n, community_points[g], discrete_points[i], user_bandwidths[i])
            # print "t : "+str(t)
            if prev_t != t:
                community_points[g].append((t,x,y,i))
                # sample c from theta_g
                c = self.sample_multinomial(theta[g,:])
                # print "theta_g,c : "+str(theta[g,c])

                # update operations
                all_samples.append((i, t, x, y, c, g))
                # print "m_i : "+str(m_i)
                lamda = self.update_intensity(lamda,i,m_i)
        # print "data generated : "
        # print all_samples

        # write sysnthetic data generated in file

        fout_synthetic_data = io.open(kde_file+"synthetic_data", "w", encoding="utf-8")
        for (i, t, x, y, c, g) in all_samples:
            fout_synthetic_data.write(unicode(i) + u"," + unicode(t) + u"," + unicode(x) + u"," + unicode(y) + u"," + unicode(c) + u"," + unicode(g) + u"\n")

        fout_synthetic_data_parameters.close()
        fout_synthetic_data.close()

        return all_samples


    def main(self):
        num_of_users = raw_input("enter number of users : ")
        num_of_communities = raw_input("enter number of communities : ")
        num_of_categories = raw_input("enter number of categories : ")
        candidate_points = raw_input("enter number of candidate points to be generated : ")
        mean = raw_input("enter mean for user to checkin dist : ")
        var = raw_input("enter variance for user to checkin dist : ")

        obj = simulate_data(int(num_of_users), int(num_of_communities), int(num_of_categories))
        obj.generate_data(float(mean), float(var), sys.argv[1], sys.argv[2], int(candidate_points))


if __name__ == '__main__':
    simulate_data().main()
