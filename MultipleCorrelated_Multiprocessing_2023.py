from multiprocessing import Pool
import random
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
import numpy as np
import time
from math import sqrt, pi, exp

# 
start_time = time.time()
G = nx.barabasi_albert_graph(1000, 2)
model = ep.SIModel(G)
mean = 5 
std = 2 
for (u, v) in G.edges():
    delay = max(1, int(np.random.normal(mean, std)))
    G[u][v]["delay"] = delay

edge_delays = [G[u][v]["delay"] for (u, v) in G.edges()]

# mean_delay = np.mean(edge_delays)
cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.1)
cfg.add_model_parameter("fraction_infected", 0.1)
model.set_initial_status(cfg)

iterations = model.iteration_bunch(200)
# print(iterations)

source_node = np.random.choice(G.nodes())
observers = random.sample(list(G.nodes()), 4)
def histo():
    plt.hist(edge_delays, bins=20, color='lightblue')
    plt.xlabel("Delay")
    plt.ylabel("Frequency")
    plt.title("Distribution of Edge Delays")
    plt.show()
    
    plt.hist(edge_delays, density=True, bins=20, alpha=0.5, color='lightblue')
    plt.xlabel("Delay")
    plt.ylabel("Probability Density")
    plt.title("Probability Density Function of Edge Delays")
    plt.show()
    
histo()
# 

def calculate_reception_time(observer_node):
    reception_times={}
    path = nx.shortest_path(G, source=source_node, target=observer_node, weight='delay')
    reception_time = sum(G[u][v]['delay'] for u, v in zip(path, path[1:]))
    reception_times[observer_node] = reception_time
    return reception_times

node_colors = ['red' if node == source_node else 'g' if node in observers else 'lightblue' for node in G.nodes()]

def plot_network():
    pos = nx.spring_layout(G)
    nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()}, font_color='black',font_size=4)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30)
    nx.draw_networkx_edges(G, pos, width=0.1)
    plt.show()

plot_network()


edge_delays={}
def cal_edges():
    global edge_delays
    for (u, v) in G.edges():
        delay = max(1, int(np.random.normal(mean, std)))
        G[u][v]["delay"] = delay
    edge_delays = [G[u][v]["delay"] for (u, v) in G.edges()]

def calculate_mean_std(obs1):
    delays = []
    for obs2 in observers:
        if obs1 != obs2:
            shortest_path = nx.shortest_path(G, source=obs1, target=obs2, weight='delay')
            delay = sum([G[shortest_path[j]][shortest_path[j+1]]['delay'] for j in range(len(shortest_path)-1)])
            delays.append(delay)
    mean = np.mean(delays)
    std = np.std(delays)
    return mean, std

def find_correlated_paths(node1, node2):
    paths = list(nx.all_simple_paths(G, node1, node2))
    correlated_paths = []
    for i, path1 in enumerate(paths):
        for j, path2 in enumerate(paths[i + 1:], i + 1):
            common_edges = set(path1) & set(path2)
            if len(common_edges) > 0:
                correlated_paths.append((path1, path2))
        
    if len(correlated_paths) == 0:
        return None
    else:
        return correlated_paths
  



def correlation_coefficient(R1, R2):
    intersection = set(R1).intersection(set(R2))
    length = max(len(R1), len(R2))
    rho = len(intersection) / length
    return rho


def correlated_paths_coefficient(node1, node2,correlated_paths):
    pair = (node1, node2)
    paths = correlated_paths[pair]
    if paths is None:
        return None
    else:
        result = []
        for path1, path2 in paths:
            rho = correlation_coefficient(path1, path2)
            result.append((path1, path2, rho))
        return rho ,result


def expected_minimum(µ,σ,ρ):
    σ1 = σ[0]
    σ2 = σ[1]
    µ1 = µ[0]
    µ2 = µ[1]
    θ = sqrt(σ1**2 + σ2**2 - 2*ρ*σ1*σ2)
    φ = lambda x: 1/(sqrt(2*pi)) * exp(-x**2/2)
    EY = µ1*φ((µ2 - µ1)/θ) + µ2*φ((µ1 - µ2)/θ) - θ*φ((µ1 + µ2)/θ)
    return EY

def shortest_distance(G, source_node, nodes):
    min_distance = float('inf')
    min_node = None    
    for node in nodes:
        if node == source_node:
            return 0,node
        if node == nodes[0] or node == nodes[-1]:
            continue       
        distance = nx.shortest_path_length(G,source_node,node)        
        if distance < min_distance:
            min_distance = distance
            min_node = node
    
    return min_distance, min_node

if __name__ == "__main__":
    p = Pool(processes=3)
    observed_info = p.map(calculate_reception_time,(observers)) 
    observed_info = {node: reception_time for node, reception_time in zip(observers, observed_info)}
    # print(observed_info)
    # # print(observed_dict[observers[0]])
    # print(f"source node : {source_node}")
    # for observer, reception_time in observed_info.items():
    #     print(f"Observer node {observer} -> {reception_time[observer]} units.")
    
    # set edge values
    cal_edges()
    
    p = Pool(processes=3)
    means, stds = zip(*p.map(calculate_mean_std, observers))
    # print(means)
    # print("----------")
    # print(stds)
    
    observer_pairs = []
    correlated_paths = {}
    for i in range(len(observers)):
        for j in range(i+1, len(observers)):
            pair = (observers[i], observers[j])
            observer_pairs.append(pair)
            correlated_paths[pair] = []

    p = Pool(processes=3)
    results = p.starmap(find_correlated_paths, observer_pairs)

    for pair, paths in zip(observer_pairs, results):
        if paths is not None:
            correlated_paths[pair] = paths
    
    # pair = (observers[0], observers[1])
    # print(pair)
    # print(correlated_paths[pair])    



    # min_ans = None
    # min_result = None
    # for i in range(len(observers)):
    #     for j in range(i+1, len(observers)):
    #         µ = [means[i], means[j]]
    #         σ = [stds[i], stds[j]]
    #         rho , result = correlated_paths_coefficient(observers[i],observers[j],correlated_paths)
    #         ans = expected_minimum(µ, σ, rho)
    #         if min_ans is None or ans < min_ans:
    #             min_ans = ans
    #             min_result = result   
                
    
    # merged_list = list(set(min_result[0][0]).union(set(min_result[0][1])))
    # min1, node1 = shortest_distance(G, source_node, merged_list)
    # print(f"\nsource_node : {node1} ,error distance : {min1}")    
    end_time = time.time()    
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")



