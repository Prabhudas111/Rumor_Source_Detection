import random
import networkx as nx
import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import matplotlib.pyplot as plt
import numpy as np
import time
from math import sqrt, pi, exp
from multiprocessing import Pool


start_time = time.time()

G = nx.barabasi_albert_graph(20, 2)
model = ep.SIModel(G)

mean = 5 
std = 2 
for (u, v) in G.edges():
    delay = max(1, int(np.random.normal(mean, std)))
    G[u][v]["delay"] = delay

edge_delays = [G[u][v]["delay"] for (u, v) in G.edges()]


cfg = mc.Configuration()
cfg.add_model_parameter('beta', 0.1)
cfg.add_model_parameter("fraction_infected", 0.1)
model.set_initial_status(cfg)

iterations = model.iteration_bunch(200)




source_node = np.random.choice(G.nodes())

observers = random.sample(list(G.nodes()), 3)

    


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


def calculate_reception_time(graph, source_node, observer_nodes):
    reception_times = {}
    for observer in observer_nodes:
        path = nx.shortest_path(graph, source=source_node, target=observer, weight='delay')
        reception_time = sum(graph[u][v]['delay'] for u, v in zip(path, path[1:]))
        reception_times[observer] = reception_time
    return reception_times

observed_info = calculate_reception_time(G, source_node, observers)
print(f"source node : {source_node}")
for observer, reception_time in observed_info.items():
    print(f"Observer node {observer} -> {reception_time} units.")
   
    

def calculate_delay_vector(observed_info):
    observer_nodes = list(observed_info.keys())
    n = len(observer_nodes)
    delay_times = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            delay_times[i][j] = observed_info[observer_nodes[j]] - observed_info[observer_nodes[i]]
    return delay_times


observed_delay_vector = calculate_delay_vector(observed_info)


node_colors = ['red' if node == source_node else 'g' if node in observers else 'lightblue' for node in G.nodes()]



def plot_network():
    pos = nx.spring_layout(G)
    nx.draw_networkx_labels(G, pos, labels={node: str(node) for node in G.nodes()}, font_color='black',font_size=4)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=30)
    nx.draw_networkx_edges(G, pos, width=0.1)
    plt.show()

plot_network()

for (u, v) in G.edges():
    delay = max(1, int(np.random.normal(mean, std)))
    G[u][v]["delay"] = delay

edge_delays = [G[u][v]["delay"] for (u, v) in G.edges()]

def get_all_paths_edge_delays(G, node1, node2, edge_delays):
    global min_edge_delay_paths
    all_paths = nx.all_simple_paths(G, source=node1, target=node2)
    edge_delays_dict = {}
    min_delay = float('inf')
    for path in all_paths:
        delay = sum(G[u][v]['delay'] for u, v in zip(path, path[1:]))
        edge_delays_dict[tuple(path)] = delay
        if delay < min_delay:
            min_delay = delay
            min_edge_delay_paths = [path]
        elif delay == min_delay:
            min_edge_delay_paths.append(path)
    print(min_edge_delay_paths)
    return edge_delays_dict


def calculate_mean_std(G, edge_delays, observers) :
    n = len(observers)
    means = np.zeros(n)
    stds = np.zeros(n)
    for i, obs1 in enumerate(observers):
        delays = []
        for obs2 in observers:
            if obs1 != obs2:
                shortest_path = nx.shortest_path(G, source=obs1, target=obs2, weight='delay')
                delay = sum([G[shortest_path[j]][shortest_path[j+1]]['delay'] for j in range(len(shortest_path)-1)])
                delays.append(delay)
        means[i] = np.mean(delays)
        stds[i] = np.std(delays)
    return means, stds


means, stds = calculate_mean_std(G, edge_delays, observers)
print(f"\nmean: {means}")
print(f"standard deviation: {stds}")


def find_correlated_paths(G, node1, node2, num_processes=1):
    paths = list(nx.all_simple_paths(G, node1, node2))
    correlated_paths = []
    
    def find_common_edges(paths_chunk):
        correlated_paths_chunk = []
        for i, path1 in enumerate(paths_chunk):
            for j, path2 in enumerate(paths_chunk[i + 1:], i + 1):
                common_edges = set(path1) & set(path2)
                if len(common_edges) > 0:
                    correlated_paths_chunk.append((path1, path2))
        return correlated_paths_chunk
    
    if len(paths) > 0:
        if num_processes == 1:
            correlated_paths = find_common_edges(paths)
        else:
            chunk_size = len(paths) // num_processes
            with Pool(num_processes) as p:
                results = p.map(find_common_edges, [paths[i:i+chunk_size] for i in range(0, len(paths), chunk_size)])
                correlated_paths = [item for sublist in results for item in sublist]
    
    if len(correlated_paths) == 0:
        return None
    else:
        return correlated_paths


def correlation_coefficient(R1, R2):
    intersection = set(R1).intersection(set(R2))
    length = max(len(R1), len(R2))
    rho = len(intersection) / length
    return rho

def correlated_paths_coefficient(G, node1, node2):
    paths = find_correlated_paths(G, node1, node2)
    if paths is None:
        return None
    else:
        result = []
        for path1, path2 in paths:
            rho = correlation_coefficient(path1, path2)
            result.append((path1, path2, rho))
            break
        return rho ,result


rho,result = correlated_paths_coefficient(G, observers[0],observers[1])
print(result)
print(rho)

def expected_minimum(µ,σ,ρ):
    σ1 = σ[0]
    σ2 = σ[1]
    µ1 = µ[0]
    µ2 = µ[1]
    θ = sqrt(σ1**2 + σ2**2 - 2*ρ*σ1*σ2)
    φ = lambda x: 1/(sqrt(2*pi)) * exp(-x**2/2)
    EY = µ1*φ((µ2 - µ1)/θ) + µ2*φ((µ1 - µ2)/θ) - θ*φ((µ1 + µ2)/θ)
    return EY

# ans = expected_minimum(means[0],means[1], stds[0],stds[1], rho)
# print(f"result : {ans}")
# Calculate expected minimum for each pair of observers

min_ans = None
min_result = None
for i in range(len(observers)):
    for j in range(i+1, len(observers)):
        µ = [means[i], means[j]]
        σ = [stds[i], stds[j]]
        rho , result = correlated_paths_coefficient(G,observers[i],observers[j])
        ans = expected_minimum(µ, σ, rho)
        if min_ans is None or ans < min_ans:
            min_ans = ans
            min_result = result


print(f"\nThe minimum expected minimum value is: {min_ans}")
print(f"The corresponding path is: {min_result}")
print(f"The corresponding path is: {min_result[0][0]}")

def shortest_distance(G, source_node, nodes):
    min_distance = float('inf')
    min_node = None    
    for node in nodes:
        if node == nodes[0] or node == nodes[-1]:
            continue       
        distance = nx.shortest_path_length(G,source_node,node)        
        if distance < min_distance:
            min_distance = distance
            min_node = node
    
    return min_distance, min_node

min1,node1 = shortest_distance(G, source_node , min_result[0][0])
min2,node2 = shortest_distance(G,source_node , min_result[0][0])
if min1 < min2:
    print(f"\nsource_node : {node1} ,error distance : {min1}")
else: 
    print(f"\nsource_node : {node2} ,error distance : {min2}")

end_time = time.time()

execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")



