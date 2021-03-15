import numpy as np
from sklearn.neighbors import KernelDensity
import sys
from datetime import datetime
from Initialize import importCheckins, importVenues

class kernelParameters:

    def kde(self, x, y):
        xy = np.vstack((x, y))

        d = xy.shape[0]
        n = xy.shape[1]
        bw = (n * (d + 2) / 4.) ** (-1. / (d + 4))  # silverman
        # bw = n**(-1./(d+4)) # scott
        #print('bw: {}'.format(bw))

        kde = KernelDensity(bandwidth=bw, metric='euclidean',
                        kernel='gaussian', algorithm='ball_tree')
        kde.fit(xy.T)

        return kde

    def sample_kde(self, KDE, n):
        points = KDE.sample(n)
        return points

    def getTimeLag(self, time_stamps):
        time_lags = []
        for t in time_stamps:
            k = 0
            date_t = datetime.strptime(t, '%Y-%m-%d %H:%M:%S')
            date_tk = datetime.strptime(time_stamps[k], '%Y-%m-%d %H:%M:%S')
            while date_tk < date_t:
                time_lags.append(date_t - date_tk)
                k +=1
                date_tk = datetime.strptime(time_stamps[k], '%Y-%m-%d %H:%M:%S')

        return time_lags

    def get_training_data(self, checkin_file, venues_file):

        #get checkins per user and venue information
        checkins = importCheckins(checkin_file)
        venues = importVenues(venues_file)

        # get user, lat, lon and timestamp
        userTrainingData = []

        for user in checkins.iterkeys():
            lat = []
            lon = []
            time_stamps = []
            for user_checkins in checkins.get(user):
                # get geo information
                geo_cordinates = venues.get(user_checkins[0])[0][0]
                lat.append(geo_cordinates[0])
                lon.append(geo_cordinates[1])

                #get time stamps
                time_stamps.append(user_checkins[1])

            userTrainingData.append((user,lat,lon,time_stamps))
        return userTrainingData


    def get_user_parameters(self, userTrainingData):

        bandwidthUsers = {}
        nuUsers = {}

        for entry in userTrainingData:
            user = entry[0]
            lat = entry[1]
            lon = entry[2]
            time_stamps = entry[3]

            # time lags for single user
            time_lags = self.getTimeLag(time_stamps)

            # store kernel parameters per user
            bandwidthUsers[user] = self.kde(lat, lon)

        return bandwidthUsers


    def main(self):
        userTrainingData = self.get_training_data(sys.argv[1], sys.argv[2])

        bandwidthUsers = self.get_user_parameters(userTrainingData)


if __name__ == '__main__':
    kernelParameters().main()

