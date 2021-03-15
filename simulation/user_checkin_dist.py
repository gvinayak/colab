import sys
import matplotlib.pyplot as plt
import numpy as np
import random

from Initialize import importCheckins, importVenues

class user_checkin_dist:

    def gamma_random_sample(self, mean, variance, size):
        """Yields a list of random numbers following a gamma distribution defined by mean and variance"""
        g_alpha = mean*mean/variance
        g_beta = mean/variance
        for i in range(size):
            yield random.gammavariate(g_alpha,1/g_beta)

    def fit_distribution(self, checkin_file):
        #get checkins per user and venue information
        checkins = importCheckins(checkin_file)
        freq_dist = []
        for user in checkins.iterkeys():
            key = len(checkins.get(user))
            if(key>=10):
                freq_dist.append(key)

        #Fit gamma distribution through mean and average
        mean_of_distribution = np.mean(freq_dist)
        variance_of_distribution = np.var(freq_dist)

        print "mean : "+str(mean_of_distribution)
        print "var : "+str(variance_of_distribution)

        # force integer values to get integer sample
        grs = [int(i) for i in self.gamma_random_sample(mean_of_distribution,variance_of_distribution,len(freq_dist))]
        self.plot_data(freq_dist, grs)

    def plot_data(self, freq_dist, grs):
        #print("Original data: ", sorted(data))
        #print("Random sample: ", sorted(grs))

        #data = stats.gamma.rvs(0.55707, loc=loc, scale=165.7216, size=78000)
        #data.sort()
        freq_dist.sort()
        freq_dist.reverse()
        grs.sort()
        grs.reverse()
        plt.plot(freq_dist, label = 'data')
        plt.plot(grs, color='r', label = 'fitted gamma dist')
        plt.xlabel("users")
        plt.ylabel("#checkins")
        plt.title("Best fit gamma distribution")
        plt.text(45000,2500,"alpha = 0.474 , theta = 0.004")
        plt.legend(loc='best', frameon=False)
        plt.show()

    def get_user_checkin_count(self, checkin_file):
        checkins = importCheckins(checkin_file)
        freq_dist_10 = 0
        freq_dist_50 = 0
        freq_dist_100 = 0
        for user in checkins.iterkeys():
            key = len(checkins.get(user))
            if(key >= 100):
                freq_dist_100 += 1
            if(key >= 50):
                freq_dist_50 += 1
            if(key >= 10):
                freq_dist_10 += 1

        print "> 10 : ", freq_dist_10
        print "> 50 : ", freq_dist_50
        print "> 100 : ", freq_dist_100

    def main(self):
        self.get_user_checkin_count(sys.argv[1])
        # self.fit_distribution(sys.argv[1])

if __name__ == '__main__':
    user_checkin_dist().main()


