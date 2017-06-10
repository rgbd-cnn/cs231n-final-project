import os

import matplotlib.pyplot as plt
import numpy as np

import json

labels = {"0": "apple", "1": "toothpaste", "2": "glue_stick",
          "3": "notebook", "4": "pliers", "5": "peach",
          "6": "food_box", "7": "pitcher", "8": "instant_noodles",
          "9": "mushroom", "10": "bowl", "11": "bell_pepper",
          "12": "food_can", "13": "hand_towel", "14": "onion",
          "15": "lemon", "16": "garlic", "17": "potato",
          "18": "food_cup", "19": "soda_can", "20": "tomato",
          "21": "coffee_mug", "22": "scissors", "23": "calculator",
          "24": "sponge", "25": "lightbulb", "26": "cell_phone",
          "27": "toothbrush", "28": "rubber_eraser", "29": "plate",
          "30": "binder", "31": "stapler", "32": "lime",
          "33": "dry_battery", "34": "keyboard", "35": "food_jar",
          "36": "kleenex", "37": "banana", "38": "water_bottle",
          "39": "shampoo", "40": "orange", "41": "cereal_box",
          "42": "camera", "43": "pear", "44": "food_bag",
          "45": "marker", "46": "ball", "47": "comb",
          "48": "flashlight", "49": "cap", "50": "greens"}


def visualize(f):
    with open(f, 'r') as fp:
        yinp = json.load(fp)

    plt.figure(1, figsize=(30, 30))

    for i in range(20):
        plt.subplot(7, 6, 2 * i + 1)
        plt.title(labels[str(yinp['label'][i])] + "-RGB")
        plt.imshow(np.array(yinp['X'][i]).astype('uint8'))
        plt.subplot(7, 6, 2 * i + 2)
        plt.title(labels[str(yinp['label'][i])] + "-D")
        plt.imshow(np.array(yinp['data'][i])[:, :, 0])
    plt.savefig(f[:-5] + '.png')


dir = './depth_maps/depth_maps'

for f in os.listdir(dir):
    if 'json' in f:
        print(f)
        visualize(os.path.join(dir, f))
