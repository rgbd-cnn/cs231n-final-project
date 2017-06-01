import argparse
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
        if 'json' in f:
            run = f[:f.find('@')]
            if run not in j:
                j[run] = {"train": [], "val": []}
            mode = f[f.find('@') + 1:-5]
            with open(os.path.join(dir, f)) as data_file:
                data = json.load(data_file)
            j[run][mode] = [i[2] * 100 for i in data]

    num_plots = len(j)
    i = 1
    for title in j:
        plt.figure(i)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(title)
        plt.ylim([50.0,105.0])
        plt.yticks(range(50, 120, 5))
        plt.plot(range(100), j[title]['train'], label="Train", linewidth=3)
        plt.plot(range(100), j[title]['val'], label="Validation", color="red", linewidth=3)
        plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
        plt.grid(b=True, axis='y')
        i += 1

        plt.savefig(os.path.join(dir, title + '.png'))

if __name__ == '__main__':
    dir = parse_arguments(sys.argv[1:]).json_file_dir
    plot_accuracies(os.path.expanduser(dir))
