import argparse
import matplotlib.pyplot as plt
import csv
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot Reward File")

    parser.add_argument('--file', default='')
    args = vars(parser.parse_args())

    data = []
    with open(args['file'], newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append([float(num) for num in row])

    data = np.asarray(data)
    data = np.abs(data)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Average reward per epoch')

    ax1.set_title('Training')
    ax1.plot(data[:, 0])
    ax1.set(xlabel='Epoch', ylabel='Average reward')

    ax2.set_title('Validation')
    ax2.plot(data[:, 1])
    ax2.set(xlabel='Epoch', ylabel='Average reward')

    plt.show()