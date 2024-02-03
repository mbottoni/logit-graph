import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import os
import imageio
np.random.seed(42)


def plot_graph_from_adjacency(adj_matrix, filename, pos=None, title='graph'):
    G = nx.Graph()  # Initialize an undirected graph
    n = len(adj_matrix)

    # Add edges to the graph
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    # Draw the graph
    #fig = plt.figure()
    #plt.title(title)
    #nx.draw(G, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
    #plt.show()
    fig = plt.figure()
    plt.title(title)
    nx.draw(G, pos=pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
    
    plt.show()
    # Save the figure as an image
    #plt.savefig(filename, format='png')
    #plt.close(fig)  # Close the figure to release resources
    return fig


def plot_graph_and_spectrum(adj_matrix, spectrum, filename, pos=None, title=None):
    G = nx.Graph()  # Initialize an undirected graph
    n = len(adj_matrix)

    # Add edges to the graph
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)

    """Plot the graph and its spectrum side by side."""
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    
    # Plot the graph
    #nx.draw(G, pos, ax=axs[0], with_labels=True)
    nx.draw(G, pos=pos, with_labels=True, ax=axs[0], node_size=700, node_color="skyblue", font_size=15, font_weight='bold')
    axs[0].set_title('Graph')

    # Plot the spectrum
    axs[1].hist(spectrum, bins=60)
    axs[1].set_title('Spectrum')

    # Set the overall title
    if title:
        plt.suptitle(title)
    
    # Save the plot to a file
    plt.savefig(filename, format='png')
    plt.close(fig)  # Close the figure to release resources
    #plt.show()
    return fig


def create_gif_from_images(image_folder, gif_filename='graph_animation.gif', duration=500):
    images = []
    sorted_list = sorted(os.listdir('images'), key=lambda x: int(x.split('.')[0]))
    #for filename in os.listdir(image_folder):
    for filename in sorted_list:
        if filename.endswith(".png"):
            filepath = os.path.join(image_folder, filename)
            images.append(imageio.imread(filepath))

    imageio.mimsave(gif_filename, images, duration=duration)
