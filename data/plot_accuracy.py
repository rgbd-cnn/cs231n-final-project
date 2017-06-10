import argparse
import csv

import json
import os
import matplotlib.pyplot as plt
import sys


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--json_file_dir', type=str,
                        help='Directory of the json data files.',
                        default='./accuracies')
    return parser.parse_args(argv)


def plot_accuracies(dir):
    j = {}
    for f in os.listdir(dir):
        if 'csv' in f:
            with open(os.path.join(dir, f),'r') as csvfile:
                data = [l for l in csv.reader(csvfile)]
            ind = f.find(',')
            string = f[:ind]
            indx = string.rfind('-')
            run = string[:indx]
            if run not in j:
                j[run] = {"train": [], "val": []}
            if "train" in f:
                j[run]['train'] = [float(i[2]) * 100 for i in data[1:]]
            else:
                j[run]['val'] = [float(i[2]) * 100 for i in data[1:]]


    i = 1
    for title in j:
        plt.figure(i)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.ylim([50.0,105.0])
        plt.yticks(range(50, 120, 5))
        plt.plot(range(len(j[title]['train'])), j[title]['train'], label="Train", linewidth=3)
        plt.plot(range(len(j[title]['train'])), j[title]['val'], label="Validation", color="red", linewidth=3)
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.grid(b=True, axis='y')
        i += 1

        plt.savefig(os.path.join(dir, title + '.png'))

if __name__ == '__main__':
    dir = parse_arguments(sys.argv[1:]).json_file_dir
    plot_accuracies(os.path.expanduser(dir))
