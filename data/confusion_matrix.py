import matplotlib.pyplot as plt
import numpy as np
import json
conf_arr = np.zeros((51, 51))

with open('confusion.json', 'r') as f:
    j = json.load(f)

labels = j['labels']
data = j['data']

for i in data:
    conf_arr[i[1], i[0]] += 1

conf_arr = conf_arr.astype('uint8')

norm_conf = []

print(conf_arr)
for i in conf_arr:
    a = 0
    tmp_arr = []
    a = sum(i, 0)
    for j in i:
        tmp_arr.append(float(j) / float(a) if a > 0 else 0)
    norm_conf.append(tmp_arr)

fig = plt.figure(figsize=(60, 60))
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf).T, cmap=plt.cm.jet, interpolation='nearest')

conf_arr = conf_arr.T

norm_conf = np.array(norm_conf).T

width, height = np.array(conf_arr).shape

for x in range(width):
    for y in range(height):
        ax.annotate(str(conf_arr[x][y]) + "\n(%.1f" % (norm_conf[x,y]*100) + "%)", xy=(y, x),
                    horizontalalignment='center',
                    verticalalignment='center')

cb = fig.colorbar(res)
cb.ax.tick_params(labelsize=50)
plt.title("Depth-Enhanced CNN Confusion Matrix on UWASH-2D Dataset", fontsize=75)
plt.ylabel('Predicted Class', fontsize=60)
plt.xlabel('True Class', fontsize=60)
plt.xticks(range(width), [labels[str(i)] for i in range(51)], rotation=90, fontsize=35)
plt.yticks(range(height), [labels[str(i)] for i in range(51)], fontsize=35)
# plt.show()
plt.savefig('confusion_matrix.pdf', format='pdf')
