import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, df):
        self.df = df

    def make_grid(self, grid_size, features):
        """Create 2D grid of neurons with random weights"""
        grid = np.random.normal(
            loc=0.0, scale=1.0,
            size=(grid_size, grid_size, features)
        )
        return grid

    def BMU(self, sample, grid):
        """Find Best Matching Unit (BMU) for a sample"""
        best_coords = (0, 0)
        best_point = grid[0, 0]
        distance = np.inf

        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                neuron = grid[i, j]
                dis = np.linalg.norm(sample - neuron)
                if dis < distance:
                    distance = dis
                    best_point = neuron
                    best_coords = (i, j)

        return best_coords, best_point

    def train(self, learning_rate, neighborhood_radius,
              lr_decay_rate, nr_decay, percent_df,
              T, grid_size):
        """Train the SOM"""
        df_sample, _ = train_test_split(
            self.df, train_size=percent_df, random_state=42
        )
        df_norm = (df_sample - df_sample.mean()) / df_sample.std()

        grid = self.make_grid(grid_size, len(self.df.columns))

        for t in range(T):
            lr_t = learning_rate * np.exp(-t * lr_decay_rate)
            nr_t = neighborhood_radius * np.exp(-t * nr_decay)

            for _, row in df_norm.iterrows():
                sample = row.values
                (bmu_i, bmu_j), _ = self.BMU(sample, grid)

                for i in range(grid.shape[0]):
                    for j in range(grid.shape[1]):
                        dist_to_bmu = np.sqrt((i - bmu_i) ** 2 + (j - bmu_j) ** 2)
                        if dist_to_bmu <= nr_t:
                            influence = np.exp(-(dist_to_bmu ** 2) / (2 * (nr_t ** 2)))
                            grid[i, j] += lr_t * influence * (sample - grid[i, j])

        self.grid = grid
        return grid

    def map_samples(self, df):
        """Map each sample to its BMU coordinates"""
        coords = []
        df_norm = (df - df.mean()) / df.std()
        for _, row in df_norm.iterrows():
            (bmu_i, bmu_j), _ = self.BMU(row.values, self.grid)
            coords.append((bmu_i, bmu_j))
        return coords

    def plot_mapping(self, df, labels=None):
        """Plot the samples on the grid"""
        coords = self.map_samples(df)
        plt.figure(figsize=(7, 7))
        for idx, (i, j) in enumerate(coords):
            if labels is not None:
                plt.text(j, i, str(labels.iloc[idx]),
                         ha='center', va='center',
                         bbox=dict(facecolor='white', alpha=0.6, lw=0))
            else:
                plt.plot(j, i, 'ko')
        plt.title("SOM Mapping")
        plt.gca().invert_yaxis()
        plt.show()