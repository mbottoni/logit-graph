import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import imageio
import pickle
from scipy.stats import gaussian_kde
from pyvis.network import Network


class GraphUtils:
    @staticmethod
    def plot_graph_from_adjacency(adj_matrix: np.array, pos=None, title='Graph', size=(10, 10), node_size=700, font_size=10):
        """Plot the graph from an adjacency matrix and save the plot as an image."""
        G = nx.from_numpy_array(adj_matrix)
        fig = plt.figure(figsize=size)
        plt.title(title)
        nx.draw(G, pos=pos, with_labels=True, node_size=node_size, node_color="skyblue", font_size=font_size, font_weight='bold')
        #plt.close(fig)
        plt.show(fig)
        return fig

    @staticmethod
    def plot_spectrum_and_zoom(spectrum, zoom_scale=0.1, size=(15, 5), title=None):
        """Plot the spectrum of a graph and a zoomed view around the mean of the spectrum."""
        #G = nx.from_numpy_array(adj_matrix)
        #spectrum = np.linalg.eigvalsh(nx.laplacian_matrix(G).toarray())

        # Calculate the mean and standard deviation of the spectrum
        mean_spectrum = np.mean(spectrum)
        std_spectrum = np.std(spectrum)

        # Define the zoom range around the mean
        zoom_range = zoom_scale * std_spectrum
        zoom_min, zoom_max = mean_spectrum - zoom_range, mean_spectrum + zoom_range

        fig, axs = plt.subplots(1, 2, figsize=size)

        # Plot the full spectrum histogram
        axs[0].hist(spectrum, bins=60, density=True, alpha=0.75, color='skyblue', edgecolor='black')
        axs[0].set_title('Full Spectrum')

        # Plot the zoomed-in spectrum
        mask = (spectrum >= zoom_min) & (spectrum <= zoom_max)
        axs[1].hist(spectrum[mask], bins=30, density=True, alpha=0.75, color='skyblue', edgecolor='black')
        axs[1].set_title(f'Zoom Around Mean (±{zoom_scale*100:.0f}% of Std Dev)')

        # Optionally add KDE plots
        kde = gaussian_kde(spectrum)
        x_range_full = np.linspace(min(spectrum), max(spectrum), 500)
        kde_values_full = kde(x_range_full)
        axs[0].plot(x_range_full, kde_values_full, color='darkblue', lw=2, label='KDE')

        x_range_zoom = np.linspace(zoom_min, zoom_max, 500)
        kde_values_zoom = kde(x_range_zoom)
        axs[1].plot(x_range_zoom, kde_values_zoom, color='darkblue', lw=2, label='KDE')

        if title:
            plt.suptitle(title)

        plt.show()
        return fig

    @staticmethod
    def plot_graph_and_spectrum(adj_matrix: np.array, spectrum, pos=None, title=None, size=(15, 10)):
        """Plot the graph and its spectrum side by side, and save the plot as an image."""
        G = nx.from_numpy_array(adj_matrix)
        fig, axs = plt.subplots(1, 2, figsize=size)
        nx.draw(G, pos=pos, with_labels=True, ax=axs[0], node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
        axs[0].set_title('Graph')
        axs[1].hist(spectrum, bins=60, density=True, alpha=0.75, color='skyblue', edgecolor='black')

        # Calculate and plot the KDE
        kde = gaussian_kde(spectrum)
        x_range = np.linspace(min(spectrum), max(spectrum), 500)
        kde_values = kde(x_range)
        axs[1].plot(x_range, kde_values, color='darkblue', lw=2, label='KDE')

        axs[1].set_title('Spectrum')
        if title:
            plt.suptitle(title)
        #plt.close(fig)
        plt.show()
        return fig

    @staticmethod
    def plot_degree_distribution(adj_matrix: np.array, title='Degree Distribution', size=(10,10)):
        """Plot the degree distribution of the graph along with its KDE."""
        G = nx.from_numpy_array(adj_matrix)
        degrees = [G.degree(n) for n in G.nodes()]
        #kde = gaussian_kde(degrees)
        x_range = np.linspace(min(degrees), max(degrees), 500)
        #kde_values = gaussian_kde(x_range)
        fig = plt.figure(figsize=size)
        # Plot the histogram
        plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), density=True,
                 align='left', alpha=0.75, color='skyblue', edgecolor='black')
        # Plot the KDE
        #plt.plot(x_range, kde_values, color='darkblue', lw=2, label='KDE')
        # Set titles and labels
        max_degree = np.max(degrees)
        avg_degree = np.mean(degrees)
        info = f" | max_degree: {max_degree}, avg_degree: {avg_degree}"
        plt.title(title+info)
        plt.xlabel('Degree')
        plt.ylabel('Frequency')
        plt.legend()
        # Prevent the plot from showing automatically in Jupyter notebooks
        #plt.close(fig)
        plt.show()
        return fig

    @staticmethod
    def save_graph_html(g: np.array, params_dict):
        # Plot with pyvis
        net = Network(
            directed = False,
            select_menu = True, # Show part 1 in the plot (optional)
            filter_menu = True, # Show part 2 in the plot (optional)
            notebook = True,
            cdn_resources='in_line'
        )
        path = '../data/output/'
        params_str = "_".join(f"{k}={v}" for k, v in params_dict.items())
        graph_filename = f"{path}graph_{params_str}.html"
        net.show_buttons() # Show part 3 in the plot (optional)
        net.from_nx(nx.from_numpy_array(g)) # Create directly from nx graph
        net.show(graph_filename)

    @staticmethod
    def loading_graph_artifacts(params_dict):
        path = '../data/input/'
        params_str = "_".join(f"{k}={v}" for k, v in params_dict.items())
        graph_filename = f"{path}graph_data_{params_str}.pickle"
        spec_filename = f"{path}spec_data_{params_str}.pickle"
        
        # Load graphs and spec from pickle files
        with open(graph_filename, 'rb') as f:
            graphs = pickle.load(f)
        with open(spec_filename, 'rb') as f:
            spec = pickle.load(f)
        
        return graphs, spec

    @staticmethod
    def saving_graph_artifacts(params_dict, graphs, spec):
        path = '../data/input/'
        params_str = "_".join(f"{k}={v}" for k, v in params_dict.items())
        graph_filename = f"{path}graph_data_{params_str}.pickle"
        spec_filename = f"{path}spec_data_{params_str}.pickle"
        
        print(graph_filename)
        print(spec_filename)
        
        # Save graphs and spec in pickle files
        with open(graph_filename, 'wb') as f:
            pickle.dump(graphs, f)
        with open(spec_filename, 'wb') as f:
            pickle.dump(spec, f)

