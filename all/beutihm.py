from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use("Agg")
data = pd.read_csv("RPL36_metascores.csv")
print(data)
tick_labels = [f'{i} Class' for i in range(0,10)]
figsize = (8,8)
ticks_size = 9
fontsize_ax = 9
fontsize_title = 12
annotation_size = 8

fig, ax = plt.subplots(figsize=figsize)
plt.title('Confusion Matrix', fontsize=fontsize_title)
hmap = sns.heatmap(data,
                     ax=ax,
                     annot=True,
                     square=True,
                     fmt='.1%',
                     cmap='GnBu',
                     cbar=False,
                     annot_kws={'size':str(annotation_size)},
                     xticklabels=tick_labels,
                     yticklabels=tick_labels)
plt.ylabel('Actual', fontsize=fontsize_ax)
plt.xlabel('Predicted', fontsize=fontsize_ax)
"""
hmap.set_xticklabels(hm.get_xmajorticklabels(), fontsize=ticks_size)
hmap.set_yticklabels(hm.get_ymajorticklabels(), fontsize=ticks_size)
"""

plt.show()