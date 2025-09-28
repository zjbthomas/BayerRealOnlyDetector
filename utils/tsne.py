from cProfile import label
import sklearn.manifold
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class TSNE():
    def __init__(self, label_strs):
        self.tsne = sklearn.manifold.TSNE(n_components = 2, random_state = 3721, init = 'pca', learning_rate = 'auto')

        self.label_strs = label_strs

    def gen_tsne_plt(self, ws, y_true, n_classes):
        z = self.tsne.fit_transform(np.array(ws))

        df = pd.DataFrame()
        df["y"] = y_true
        df["x1"] = z[:,0]
        df["x2"] = z[:,1]

        plt.figure()
        sp = sns.scatterplot(x="x1", y="x2", hue=df.y.tolist(),
                        palette=sns.color_palette("hls", n_classes),
                        data=df,
                        s=1)
        sp.set(xlabel=None)
        sp.set(ylabel=None)
        sp.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # replace labels
        for ix, t in enumerate(sp.legend_.texts):
            t.set_text(self.label_strs[ix])

        plt.tight_layout()

        return plt