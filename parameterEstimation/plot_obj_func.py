import io, sys
import numpy as np
import matplotlib.pyplot as plt


def plot_estimated_values():

    aij_real = [0.2       , 0.25      , 0.        , 0.        , 0.        ,
       0.2       , 0.25      , 0.        , 0.        , 0.        ,
       0.2       , 0.25      , 0.5       , 0.33333333, 0.        ,
       0.2       , 0.        , 0.5       , 0.33333333, 1.        ,
       0.2       , 0.25      , 0.        , 0.33333333, 0.        ]

    aij_estimated = [64.16903442e-01, 4.01194943e+03, 2.49178897e-01, 1.46193874e-01,
       2.13556839e-01, 3.85846598e-01, 3.74126893e-01, 1.96096935e-01,
       2.14272068e-01, 1.33837148e-01, 3.03091797e-01, 3.79372828e-01,
       2.66872318e+03, 4.90760198e-01, 1.90070507e-01, 3.95691953e-01,
       1.32713117e-01, 8.43432679e-01, 6.02739594e-01, 1.14160578e+00,
       4.05743009e-01, 3.60812033e-01, 1.06516136e-01, 3.43928219e+03,
       5.40380566e-02]

    theta_real = []
    theta_estimated = []
    aij_r = np.array(aij_real)
    aij_e = np.array(aij_estimated)
    rel_err = abs(aij_r - aij_e)

    print rel_err
    print np.average(rel_err)

    plt.plot(aij_real, color='blue' ,label="phi real")
    plt.plot(aij_estimated, color='red', label="phi estimated")
    # plt.plot(mu_checkin_prop, color = 'black', label="mu checkins propotional")
    plt.xlabel("user")
    plt.ylabel("phi")
    plt.title("50 users")
    plt.text(0.1,0.5, "")
    plt.legend(loc='best', frameon=False)
    plt.show()


def plot_data(data, text):

    plt.plot(data, label='objective function values')
    plt.xlabel("iteration")
    plt.ylabel("obj func value")
    plt.title(text)

    # plt.text(45000, 2500, "alpha =,,,,,,,, 0.474 , theta =,,,,,,,, 0.004")
    # plt.legend(loc='best', frameon=False)
    plt.show()


def main():
    plot_estimated_values()
    # fin = io.open(sys.argv[1],"r")
    # line = fin.readline()
    #
    # data = []
    # while line:
    #     line = line.rstrip()
    #     data.append(-1.0 * float(line))
    #     line = fin.readline()
    #
    # plot_data(data, "theta")


if __name__ == '__main__':
    main()