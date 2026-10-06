from typing import Optional
import numpy as np
from omegaconf import DictConfig
from sklearn.metrics.pairwise import cosine_distances as cos_dis
import torch
from .hypergraph_data import HyperGraphData
import networkx as nx
import community as community_louvain  

from .hypergraph_generation import (
    networks_map,
    topologyheterogeneity_adaptive_hyperedge,
    generate_network_labels,
    generate_expansion_hyperedge,
    triple_network_inspired_hyperedge,
)




# HC-IAHI
def hierarchical_hypergraph_construct(corr_matrix):   
   
    # THIA
    sub_hypergraph = topologyheterogeneity_adaptive_hyperedge(corr_matrix)

    # HHE
    num_nodes = corr_matrix.shape[0]
    network_labels = generate_network_labels(networks_map)  
    expansion_hyperedge = generate_expansion_hyperedge(corr_matrix,network_labels)
    # TNHE
    triple_Network_Hyperedge = triple_network_inspired_hyperedge(corr_matrix, network_labels)
    hyper_edge_sum = torch.cat((sub_hypergraph, expansion_hyperedge,triple_Network_Hyperedge), dim=1)

    return hyper_edge_sum

def create_hypergraph_data(cfg: DictConfig,         
                      final_pearson: torch.Tensor,
                      labels: torch.Tensor,
                      site: torch.Tensor):
    
    num_nodes = final_pearson.shape[1]

    hypergraph_data_list = []
    for i in range(final_pearson.shape[0]):

        hyper_edge_index = []
        fc_hyper_edges = hierarchical_hypergraph_construct(final_pearson[i])
        hyper_edge_index.append(fc_hyper_edges)
        hyper_edge_index = torch.cat(hyper_edge_index, dim=1)
        node_feature = final_pearson[i]
        edge_weight = torch.ones(hyper_edge_index[1].max() + 1)

        # Node Feature Enhancement
        num_nodes = max(max(nodes) for nodes in networks_map.values()) + 1  
        network_labels = np.zeros(num_nodes, dtype=int)  
        label = 0
        for nodes in networks_map.values():
                for node in nodes:
                    network_labels[node] = label
                label += 1
        max_label = len(networks_map) - 1  
        normalized_labels = network_labels / max_label 
        x = node_feature
        normalized_labels_tensor = torch.tensor(normalized_labels, dtype=torch.float32, device=x.device).unsqueeze(1)
        new_x = torch.cat([x, normalized_labels_tensor], dim=1)
        x = new_x
        data = HyperGraphData(x=new_x, edge_index=hyper_edge_index,
                    y=labels[i], edge_weight=edge_weight)

        hypergraph_data_list.append(data)
    return hypergraph_data_list, labels

