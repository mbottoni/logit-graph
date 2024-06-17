import numpy as np
import networkx as nx
from scipy.stats import ks_2samp
from scipy.special import expit

from src.degrees_counts import degree_vertex, get_sum_degrees

'''
1. Sigma é o parâmetro para definir a probabilidade de colocar aresta quando os graus de i e j são ambos zero, certo?

2. Para que você precisa de alpha e beta? Aliás, por que alpha pode ser diferente de beta. Se for diferente, 
dependendo de como você escolhe i e j pode dar diferença (não é simétrico). Eu deixaria alfa=beta=1.
Acho que isso só complica o modelo, pelo menos por enquanto.

3. O algoritmo como um todo está estranho. Acho que basta definir o modelo como 1/(1 + exp( - ( sigma + |i| + |j|) ) ) (estou considerando
para fins de explicação, p = 0, ou seja, apenas i e j sem considerar seus vizinhos).

OK. Passo 0. Eu acho que tanto faz inicializar o grafo com um ER e p bem pequeno ou grafo vazio.
OK Passo 1. Sorteia i e j quaisquer, com reposição.
OK Passo 2. Aplica o modelo. Se der mais que 0.5, coloca aresta. Se der menos, remove aresta.
OK Repete passos 1 e 2 até convergência. Para definir convergência eu usaria algo simples, como se o número de arestas totais não varia muito.
'''

class GraphModel:
    def __init__(self, n, d, sigma):
        self.n = n # number of nodes
        self.d = d # number of neighbors to consider 
        self.sigma = sigma
        self.graph = self.generate_empty_graph(n)

    def generate_empty_graph(self, n):
        return np.zeros((n, n))

    @classmethod
    def calculate_spectrum(cls, graph):
        G = nx.from_numpy_array(graph)
        eigenvalues = nx.laplacian_spectrum(G)
        return np.sort(eigenvalues)

    def logistic_regression(self, sum_degrees):
        return expit(sum_degrees)

    def get_edge_logit(self, sum_degrees):
        val_log = self.logistic_regression(sum_degrees)
        # Randomly choose 1 or 0 based on the probability
        random_choice = np.random.choice([1, 0], p=[val_log, 1 - val_log])

        #if val_log > 0.5:
        #    random_choice = 1
        #else:
        #    random_choice = 0

        return random_choice

    def add_remove_edge(self):
        # Pre compute
        # TODO: I dont need to pick every node
        #sum_degrees = np.zeros(self.n)
        #for i in range(self.n):
        #    sum_degrees[i] = get_sum_degrees(self.graph, vertex=i, d = self.d)

        i, j = np.random.choice(self.n, 2, replace=False)
        #total_degree = (sum_degrees[i] + sum_degrees[j]) + self.sigma
        total_degree = ( get_sum_degrees(self.graph, vertex=i, d=self.d) + get_sum_degrees(self.graph, vertex=j, d=self.d) ) + self.sigma
        self.graph[j, i] = self.graph[i, j] = self.get_edge_logit(total_degree) # here we can add or remove vertex

    def check_convergence_hist(self, graphs, stability_window=5, degree_dist_threshold=0.05):
        def degree_distribution_stability(graph1, graph2):
            # Calculate degree sequences
            degrees1 = np.sum(graph1, axis=1)
            degrees2 = np.sum(graph2, axis=1)
            # Compute the Kolmogorov-Smirnov statistic
            ks_stat, _ = ks_2samp(degrees1, degrees2)
            print(f"KS Statistic: {ks_stat}")
            return ks_stat

        if len(graphs) <= stability_window:
            print("Not enough graphs for stability check.")
            return False

        # Degree Distribution Stability: Check if the KS distance between degree distributions is below the threshold
        degree_dist_stable = all(
            degree_distribution_stability(graphs[-i - 1], graphs[-i]) < degree_dist_threshold
            for i in range(1, stability_window)
        )
        print(f"Degree Distribution Stable: {degree_dist_stable}")

        # Final convergence check
        is_converged = degree_dist_stable and True
        print(f"Graph Converged: {is_converged}")
        print('\n'*3)
        return is_converged

    def check_convergence(self, graphs, threshold_edges, stability_window):
        # Check only the last stability_window graphs
        graphs_to_check = graphs[-stability_window:]
        
        prev_total_edges = None
        for graph in graphs_to_check:
            total_edges = np.sum(np.triu(graph))
            if prev_total_edges is not None:
                if abs(total_edges - prev_total_edges) > threshold_edges:
                    return False
            prev_total_edges = total_edges
        return True

    def populate_edges(self, warm_up, max_iterations, threshold, stability_window):
        i = 0
        stop_condition = False
        graphs = [self.graph.copy()]  # List to store the graphs
        #spectra = []  # List to store the spectrum at each iteration

        while i < max_iterations and (i < warm_up or not stop_condition):
            print(f'iteration: {i}')
            self.add_remove_edge()  # add or remove vertex
            graphs.append(self.graph.copy())

            if len(graphs) > 1000:
                graphs.pop(0)

            if i > warm_up:
                #stop_condition = self.check_convergence(graphs, stability_window=stability_window, degree_dist_threshold=degree_dist_threshold)
                stop_condition = self.check_convergence(graphs, threshold_edges=threshold, stability_window=stability_window)

            i += 1

        spectra = self.calculate_spectrum(self.graph)
        return graphs, spectra






