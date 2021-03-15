from math import cos, asin, sqrt
import numpy as np

class utility:
    def distance(self, lat1, lon1, lat2, lon2):
        p = 0.017453292519943295
        a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p)*cos(lat2*p) * (1-cos((lon2-lon1)*p)) / 2
        return 12742 * asin(sqrt(a))

    def closest(self, region, x, y):
        return min(region, key=lambda p: self.distance(x,y,p[0],p[1]))

    def get_check(self, x_min, x_max, v):
        if v >= x_max:
            x_max = v
        if v <= x_min:
            x_min = v
        return (x_min, x_max)

    def get_unique_values(self, list):
        print set(list)
        return len(set(list))

    def normalize_2d_array(self, array):
        for i in range(0, array.shape[0]):
            array[i,:] = self.normalize(array[i,:])
        return array

    def normalize(self, list):
        s = sum(list)
        n = [i / s for i in list]
        return n