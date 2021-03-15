import numpy as np
import io
from simulation.utility import utility
import pdb

class synthetic_data_processing:

    # event is a tuple of (user, time, lat, lon, category, community)
    events = []
    checkins = {}

    T = 0
    X = 0
    Y = 0


    def get_checkins(self, checkins_file):
        fin = io.open(checkins_file, "r", encoding="utf-8")
        line = fin.readline()

        # get min and max arguements
        t_min = np.infty
        t_max = 0.0
        x_min = np.infty
        x_max = 0.0
        y_min = np.infty
        y_max = 0.0

        while line:
            line = line.rstrip()
            event = line.split(",")
            # user, time, lat, lon, cat, com
            tuple = (int(event[0]), float(event[1]), float(event[2]), float(event[3]))
            for i in range(4, len(event)):
                entry = int(event[i])
                tuple = tuple + (entry,)

            self.events.append(tuple)

            (t_min, t_max) = utility().get_check(t_min, t_max, float(event[1]))
            (x_min, x_max) = utility().get_check(x_min, x_max, float(event[2]))
            (y_min, y_max) = utility().get_check(y_min, y_max, float(event[3]))

            # if (self.checkins.has_key(int(event[0]))):
            #     self.checkins[int(event[0])][0].append(float(event[2]))
            #     self.checkins[int(event[0])][1].append(float(event[3]))
            #     self.checkins[int(event[0])][2].append(float(event[1]))
            # else:
            #     lat = []
            #     lon = []
            #     t = []
            #     self.checkins[int(event[0])] = (lat, lon, t)
            #     self.checkins[int(event[0])][0].append(float(event[2]))
            #     self.checkins[int(event[0])][1].append(float(event[3]))
            #     self.checkins[int(event[0])][2].append(float(event[1]))

            line = fin.readline()

        fin.close()
        # self.T = t_max - t_min
        # self.X = x_max - x_min
        # self.Y = y_max - y_min

        # scaling of t,x,y

        for i in range(0,len(self.events)):
            event = list(self.events[i])

            event[1] = (event[1] - t_min)/ (t_max - t_min)
            event[2] = (event[2] - x_min) / (x_max - x_min)
            event[3] = (event[3] - y_min) / (y_max - y_min)

            if (self.checkins.has_key(event[0])):
                self.checkins[event[0]][0].append(event[2])
                self.checkins[event[0]][1].append(event[3])
                self.checkins[event[0]][2].append(event[1])
            else:
                lat = []
                lon = []
                t = []
                self.checkins[event[0]] = (lat, lon, t)
                self.checkins[event[0]][0].append(event[2])
                self.checkins[event[0]][1].append(event[3])
                self.checkins[event[0]][2].append(event[1])

            self.events[i] = event

        self.T = 1.0
        self.X = 1.0
        self.Y = 1.0
        # print "checkins : "
        # print self.checkins
        #
        # print "events : "
        # print self.events
        return (self.events, self.checkins, self.T, self.X, self.Y)

    def get_graph(self, checkin_file, connections_file, U):
        # social connections file
        fin = io.open(connections_file, "r", encoding="utf-8")
        # read header line
        line = fin.readline()

        connections = np.zeros((U, U))

        while line:
            line = line.rstrip()
            connection = line.split(",")
            connections[int(connection[0]), int(connection[1])] = 1
            line = fin.readline()

        fin.close()
        return connections

    def get_params(self, params_file):

        # social connections file
        fin = io.open(params_file, "r", encoding="utf-8")
        mu = []
        Aij = []
        pi = []
        theta = []
        U = ''
        V = ''
        M = ''

        line = fin.readline()
        while line:
            line = line.rstrip()
            data = line.split("=")
            # get mu
            if('mu' in data[0]):
                mu = data[1].split(",")
            if('Aij' in data[0]):
                Aij = data[1].split(",")
            if('pi' in data[0]):
                pi = data[1].split(",")
            if('theta' in data[0]):
                theta = data[1].split(",")
            if('U' in data[0]):
                U = int(data[1])
            if ('V' in data[0]):
                V = int(data[1])
            if ('M' in data[0]):
                M = int(data[1])
            line = fin.readline()

        for i in range(0,len(mu)):
            mu[i] = float(mu[i])

        for i in range(0,len(Aij)):
            Aij[i] = float(Aij[i])

        for i in range(0,len(pi)):
            pi[i] = float(pi[i])

        for i in range(0,len(theta)):
            theta[i] = float(theta[i])


        mu = np.array(mu)
        Aij = np.array(Aij)
        pi = np.array(pi)
        theta = np.array(theta)

        mu = np.reshape(mu, (1,U))
        Aij = np.reshape(Aij, (U,U))
        pi = np.reshape(pi, (U,M))
        theta = np.reshape(theta,(M,V))

        print ("mu real", mu)

        print ("Aij real" , Aij)

        print ("pi real" , pi)

        print ("theta real" , theta)

        pdb.set_trace()
        return (mu, Aij, pi, theta, U, V, M)

# Modifications are made here------------------------------
    def get_params_from_file(self, checkin_file, connections_file):
        cat_list = []
        # com_list = []
        usr_list = []

        temp_list = []
        with open(checkin_file) as f:
            for line in f:
                temp_list = line.split(',')
                # com_list.append(int(temp_list[len(temp_list) - 1]))
                for k in range(4,len(temp_list)):
                    cat_list.append(temp_list[k])

        cat_list = np.asarray(cat_list)
        # com_list = np.asarray(com_list)

        num_cat = int(len(np.unique(cat_list)))
        # num_com = int(len(np.unique(com_list)))


        with open(checkin_file) as f:
            for line in f:
                temp_list = line.split(',')
                usr_list.append(int(temp_list[0]))

        usr_list = np.asarray(usr_list)
        num_usr = int(len(np.unique(usr_list)))

        # result = [num_usr, num_cat, num_com]
        result = [num_usr, num_cat]

        print(result)

        return result