import numpy as np
from utility import utility
import matplotlib.pyplot as plt
import scipy.stats

class hawkes_process:

    def hawkes_intensity(self, mu, ai, points, t, x, y, h):
        """Find the hawkes intensity:
        mu_i + sum (a_ji * ( np.exp(- nu (t- t_k) - ((x - x_k)^2 + (y- y_k)^2)/ (2*h) ) for t_k in points if t_kk<=t )
        """

        nu = 0.01  # time decay parameter
        p = [((np.exp( nu * (entry[0] - t) - (np.square(x - entry[1]) + np.square(y - entry[2])) / (2 * h)) *
                          ai[entry[3]]) / (2 * np.pi * h)) for entry in points]
        # for entry in points:
        #     if (entry[0] < t):
        #         if(ai[entry[3]] > 0):
        #             p.append((np.exp( nu * (entry[0] - t) - (np.square(x - entry[1]) + np.square(y - entry[2])) / (2 * h)) *
        #                   ai[entry[3]]) / (2 * np.pi * h))
        # print "p : ", p
        return mu + sum(p)

    def generate_point(self, t, x, y, mu, ai, window, points, region, bandwidth):
        # intensity value at current t
        m = self.hawkes_intensity(mu, ai, points, t, x, y, bandwidth)

        # generate time lag from homogeneous exp distribution with this intensity
        s = np.random.exponential(scale=1 / m)

        # generate random point from discrete points
        # index = np.random.randint(0, region.shape[0])
        # x_new = region[index][0]
        # y_new = region[index][1]

        mean = [x,y]
        cov = [[bandwidth, 0],[0, bandwidth]]
        point = np.random.multivariate_normal(mean, cov, 1)
        x_new = point[0][0]
        y_new = point[0][1]

        # intensity with new t', x' and y'
        m_new = self.hawkes_intensity(mu, ai, points, t + s, x_new, y_new, bandwidth)

        ratio = m_new / m
        if ratio >= np.random.uniform():
            x_new, y_new = utility().closest(region, x_new, y_new)
            t = t + s
            x = x_new
            y = y_new

        return (t, x, y, m)




    # def simulate_hawkes(self, mu, alpha, window):
    #     t = 0
    #     user_points = {}
    #     for user in range(0,5):
    #         print user
    #         user_points[user] = []
    #     all_samples = []
    #     prev_t = t
    #     while t < window:
    #         prev_t = t
    #         user = randint(0, 4)
    #         t = self.generate_point(t, mu, alpha, window, user_points[user])
    #         if prev_t != t:
    #             user_points[user].append(t)
    #             all_samples.append(t)
    #
    #     print user_points
    #     print all_samples
    #
    # def main(self):
    #     self.simulate_hawkes(0.2, 0.05, 50)


if __name__ == '__main__':
    hawkes_process().main()
