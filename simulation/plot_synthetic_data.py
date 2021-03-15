from parameterEstimation.gradientAscent import gradientAscent
from hawkes_process import hawkes_process
from kernelParameters import kernelParameters
import matplotlib.pyplot as plt
import io, sys
import numpy as np

class plot_synthetic_data:

    U = 0
    V = 0
    M = 0

    def plot_samples(self, all_samples, bw, Aji, mu):

        Aij = np.array(Aji)
        Aij = Aji.T

        # print "Aij"
        # print Aij

        time_samples = {}
        intensity = {}

        community_points = {}
        for community in range(0, self.M):
            community_points[community] = []

        for (i, t, x, y, c, g) in all_samples:
            if(time_samples.has_key(i)):
                pass
            else:
                time_samples[i] = []
                intensity[i] = []
            time_samples[i].append(t)
            intensity[i].append(hawkes_process().hawkes_intensity(mu[i], Aij[i], community_points[g], t,x,y, bw[i]))
            community_points[g].append((t,x,y,i))
        return (time_samples, intensity)

    def read_data(self, checkins_file, parameters_file):
        (events, checkins) = gradientAscent(1, self.M, self.U, self.V, 50.0).synthetic_data_processing(checkins_file)

        (Aji, pi, theta, mu) = self.read_parameters(parameters_file)

        bw = {}
        for user in checkins.iterkeys():
            bw[user] = kernelParameters().kde(checkins[user][0], checkins[user][1]).bandwidth

        (time_samples, intensity) = self.plot_samples(events, bw, Aji, mu[0,:])
        self.plot_data(time_samples[3], intensity[3])

    def read_parameters(self, parameters_file):

        fin = io.open(parameters_file, "r", encoding="utf-8")

        Aji = None
        pi = None
        theta = None
        mu = None

        line = fin.readline()
        while line:
            line = line.rstrip()
            if("U" in line):
                self.U = int(line.split("=")[1])

            if("V" in line):
                self.V = int(line.split("=")[1])

            if("M" in line):
                self.M = int(line.split("=")[1])

            if("Aji" in line):
                Aji_str = np.array(line.split("=")[1].split(","))
                Aji_str = np.reshape(Aji_str, (self.U, self.U))
                Aji = Aji_str.astype(np.float)

            # print "Aji"
            # print Aji

            if("pi" in line):
                pi_str = np.array(line.split("=")[1].split(","))
                pi_str = np.reshape(pi_str, (self.U, self.M))
                pi = pi_str.astype(np.float)

            # print "pi"
            # print pi

            if("theta" in line):
                theta_str = np.array(line.split("=")[1].split(","))
                theta_str = np.reshape(theta_str, (self.M, self.V))
                theta = theta_str.astype(np.float)

            # print "theta"
            # print theta

            if("mu" in line):
                mu_str = np.array(line.split("=")[1].split(","))
                mu_str = np.reshape(mu_str, (1, self.U))
                mu = mu_str.astype(np.float)

            # print "mu"
            # print mu

            line = fin.readline()

        return (Aji, pi, theta, mu)

    def plot_data(self, t, intensity):

        print t
        print intensity

        plt.plot(t, intensity, label='data')
        plt.xlabel("time")
        plt.ylabel("intensity")
        plt.title("User's Intensity of checkins")
        # plt.text(45000, 2500, "alpha = 0.474 , theta = 0.004")
        # plt.legend(loc='best', frameon=False)
        plt.show()

    def main(self):
        self.read_data(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    plot_synthetic_data().main()
