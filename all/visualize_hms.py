import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd
from glob import glob

for file in glob("*.csv"):
    df = pd.read_csv(file)
    cols = [c for c in df.columns if "bin" in c]

    data = df[cols].values
    plt.figure(figsize=(7, 180))
    plt.imshow(data, cmap='hot')
    plt.colorbar()
    tmp = file.split("\\")[-1].split(".")[0]
    plt.savefig(f"hm\\{tmp}_heatmap.jpg")
    plt.close()