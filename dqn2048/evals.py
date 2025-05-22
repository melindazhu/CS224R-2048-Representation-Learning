import matplotlib.pyplot as plt
import numpy as np

def moving_average(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_metrics(total_scores, max_tiles, illegal_counts):
    episodes = list(range(1, len(total_scores) + 1))

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, total_scores, label='Total Score')
    plt.plot(episodes[len(episodes)-len(moving_average(total_scores)):], moving_average(total_scores), label='Smoothed Score', color='green')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Total Score per Episode")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, max_tiles, label='Max Tile', color='orange')
    plt.xlabel("Episode")
    plt.ylabel("Max Tile")
    plt.title("Max Tile per Episode")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(episodes, illegal_counts, label='Illegal Moves', color='red')
    plt.xlabel("Episode")
    plt.ylabel("Illegal Moves")
    plt.title("Illegal Moves per Episode")
    plt.grid(True)
    plt.legend()
    plt.show()
