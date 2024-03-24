import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio

class GraphUtils:
    @staticmethod
    def plot_graph_from_adjacency(adj_matrix, filename, pos=None, title='Graph'):
        """Plot the graph from an adjacency matrix and save the plot as an image."""
        G = nx.from_numpy_matrix(adj_matrix)
        fig = plt.figure()
        plt.title(title)
        nx.draw(G, pos=pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
        plt.show()
        plt.savefig(filename, format='png')
        plt.close(fig)
        return fig

    @staticmethod
    def plot_graph_and_spectrum(adj_matrix, spectrum, filename, pos=None, title=None):
        """Plot the graph and its spectrum side by side, and save the plot as an image."""
        G = nx.from_numpy_matrix(adj_matrix)
        fig, axs = plt.subplots(1, 2, figsize=(15, 10))
        nx.draw(G, pos=pos, with_labels=True, ax=axs[0], node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
        axs[0].set_title('Graph')
        axs[1].hist(spectrum, bins=60)
        axs[1].set_title('Spectrum')
        if title:
            plt.suptitle(title)
        plt.savefig(filename, format='png')
        plt.close(fig)
        return fig

    @staticmethod
    def plot_degree_distribution(adj_matrix, filename, title='Degree Distribution'):
        """Plot the degree distribution of the graph."""
        G = nx.from_numpy_matrix(adj_matrix)
        degrees = [G.degree(n) for n in G.nodes()]
        fig = plt.figure()
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left')
        plt.title(title)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.savefig(filename, format='png')
        plt.close(fig)
        return fig




