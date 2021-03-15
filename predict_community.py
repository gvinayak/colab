import io, sys
import numpy as np
from simulation.simulate_data import simulate_data
from sklearn.metrics.cluster import normalized_mutual_info_score
from parameterEstimation.synthetic_data_processing import synthetic_data_processing
from wordcloud import WordCloud, STOPWORDS 
import pdb, pickle, datetime
import matplotlib.pyplot as plt

class predict_community:
    events = []
    checkins = {}
    pred_com = []

    def __init__(self, checkins_file):
        # get data
        (self.events, self.checkins, T, X, Y) = synthetic_data_processing().get_checkins(checkins_file)
    def get_predicted_phi(self, estimated_phi):

        [U,M, predicted_phi] = pickle.load(open("estimated_phi.p", "rb" ))
        
        # fin = io.open(estimated_phi,"r")
        # line = fin.readline()
        # U = int(line)

        # line = fin.readline()
        # M = int(line)

        # print "U : ", U
        # print "M : ", M
        # line = fin.readline()
        # predicted_phi = []

        # while line:
        #     line = line.split(" ")
        #     if(line[0] == '['):
        #         line = line[1:]

        #     if(line[len(line) - 1] == ']'):
        #         line = line[:-1]
                
        #     # for j in len(line):
        #     #     if()
        #     predicted_phi.append(float(line))
        #     line = fin.readline()

        phi = np.array(predicted_phi)
        phi = np.reshape(phi, (U, M))
        print "phi : ",phi.shape
        return phi

    def sample_community(self, estimated_phi):

        phi = self.get_predicted_phi(estimated_phi)

        predicted_community = []
        ground_truth_community = []

        for event in self.events:
            # print "phi[event[0]] ", phi[event[0]]
            predicted_community.append(simulate_data().sample_multinomial(phi[event[0]]))
            # ground_truth_community.append(event[5])

        # print "ground truth : ", ground_truth_community
        # print "predicted : ", predicted_community
        self.pred_com = predicted_community
        unique, count = np.unique(np.asarray(predicted_community), return_counts=True)
        # score = normalized_mutual_info_score(ground_truth_community, predicted_community)
        # print "nmi score : ", score1
        # word_cloud(predicted_community).make_string()

        self.word_cloud()
        self.reverse_date()

    def reverse_date(self):
        self.read_checkin(sys.argv[1])
        print(len(self.pred_com))

    def read_checkin(self, checkins_file):
        # First Checkin: 6294689,0.0,9.32935377537,-83.9571622394,135,134
        # u6294689,l639040,2014-12-31 12:35:05
        # Day 0 is Monday

        format = 'week'
        orig_checkin = datetime.datetime.strptime("2014-12-31 12:35:05", '%Y-%m-%d %H:%M:%S')

        check_time = []
        fin = io.open(checkins_file, "r", encoding="utf-8")
        line = fin.readline()
        while line:
            line = line.rstrip()
            arr = line.split(",")
            delta = datetime.timedelta(hours=float(arr[1]))
            if(format == "week"):
                check_time.append((orig_checkin + delta).weekday())

            else:
                check_time.append((orig_checkin + delta).hour)
            line = fin.readline()

        self.plotting(np.asarray(check_time))

    def plotting(self, check_time):
        day_list = ['Monday','Tuesday','Wednesday','Thursday', 'Friday','Saturday', 'Sunday']
        comm = np.asarray(self.pred_com)
        num_comm = len(np.unique(comm))
        
        for k in np.unique(check_time):
            indices = np.where(check_time == k)
            data = comm[indices]

            temp_x, temp_y = np.unique(data, return_counts=True)
            temp = dict(zip(temp_x, temp_y))

            x = np.zeros(num_comm)
            y = np.zeros(num_comm)

            x = range(num_comm)
            for key, value in temp.items():
                y[key] = value

            y = (y/np.sum(y)).astype(float)
            
            plt.bar(x, y)
            plt.xlabel('Community')
            plt.ylabel('Ratio')
            plt.title("Day: "+day_list[k])
            plt.savefig("Images/CT_"+str(k)) 

    def word_cloud(self):
        stopwords = set(STOPWORDS) 
        venues = []
        fin = io.open("Venues.txt", "r", encoding="utf-8")
        line = fin.readline()
        while line:
            line = line.rstrip()
            venues.append(line)
            line = fin.readline()

        venues = np.asarray(venues)
        print(len(venues))

        cats = self.read_checkin_word(sys.argv[1])
        comm = np.asarray(self.pred_com)
        for k in np.unique(comm):
            indices = np.where(comm == k)
            categories = np.hstack(cats[indices])
            plot_cats = venues[np.asarray(categories)]
            print(plot_cats)

            wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords, 
                min_font_size = 10).generate(" ".join(plot_cats))

            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title("For Community "+ str(k))
            plt.savefig("Images/WC_"+str(k))  
        


    def read_checkin_word(self, checkins_file):
        user_dict, cat_dict = pickle.load(open('Input_Data/dicts.p', 'rb'))
        cats = []
        fin = io.open(checkins_file, "r", encoding="utf-8")
        line = fin.readline()
        while line:
            temp = []
            line = line.rstrip()
            arr = line.split(",")
            for k in range(4, len(arr)):
                for key,val in cat_dict.items():
                    if(val == int(arr[k])):
                        temp.append(int(key))

            cats.append(temp)
            line = fin.readline()

        return np.asarray(cats)

    def main(self, estimated_phi):
        self.sample_community(estimated_phi)

if __name__ == '__main__':
    predict_community(sys.argv[1]).main(sys.argv[2])