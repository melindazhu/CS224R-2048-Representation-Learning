import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

def plot_gini_vs_reward_hexbin(gini_values, rewards):

    plt.figure(figsize=(8, 6))
    hb = plt.hexbin(gini_values, rewards, gridsize=25, cmap='viridis', bins='log')
    plt.colorbar(hb, label='Log Count')
    plt.xlabel("Gini Coefficient")
    plt.ylabel("Reward")
    plt.title("Hexbin Heatmap: Gini vs Reward")
    plt.tight_layout()
    plt.savefig("gini_reward_hexbin.png")
    plt.show()


def plot_gini_vs_reward_heatmap(gini_values, rewards):
    df = pd.DataFrame({
        'Gini': np.round(gini_values, 2),
        'Reward': rewards
    })

    pivot = df.groupby('Gini')['Reward'].mean().reset_index()
    pivot['Dummy'] = 'avg'
    heatmap_data = pivot.pivot(index='Gini', columns='Dummy', values='Reward')

    plt.figure(figsize=(6, 8))
    sns.heatmap(heatmap_data, annot=False, cmap='YlGnBu', cbar_kws={'label': 'Avg Reward'})
    plt.title("Heatmap of Avg Reward by Gini Score")
    plt.xlabel("")
    plt.ylabel("Gini Coefficient")
    plt.tight_layout()
    plt.savefig("gini_reward_heatmap.png")
    plt.show()
