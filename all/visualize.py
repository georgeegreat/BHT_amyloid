import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")

savedir = "graphics/"
files = glob("*.csv")
plt.rcParams['font.size'] = 12
for file in files:
    frame = pd.read_csv(file)
    cols = frame.columns
    needle = [i for i in cols if i.endswith("_score")]
    bins = [i for i in cols if i.endswith("_score")]
    bsum = [i for i in frame[bins[0]].values]
    for c in bins[1:]:
        for i in range(len(bsum)):
            bsum[i] += frame[c].values[i]

    bsum = [1 if i > 4 else 0 for i in bsum]
    plt.figure(figsize=(50, 22))
    coef = 1
    c = 0
    legs = []
    for col in needle:
        legs.append(col)
        plt.plot([i for i in range(len(frame[col].values))], frame[col].values)

    plt.legend(legs)

    tmp = file.split(".")[0]
    plt.savefig(f"{savedir}{tmp}_coef={coef}_all.jpg")
    plt.close()