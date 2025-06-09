import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from typing import List, Dict, Set, Tuple
import networkx as nx
from circuit_tracer.graph import Graph
from collections import defaultdict

class CircuitPatternMiner:
    def __init__(self, graphs: Dict[str, Graph]):
        self.graphs = graphs
        self.patterns = []
        
    def extract_circuit_fingerprint(self, graph: Graph, 
                                  node_threshold: float = 0.8) -> np.ndarray:
        """Extract a feature vector representing the circuit structure."""
        
        # Get pruned adjacency matrix
        from circuit_tracer.graph import prune_graph
        node_mask, edge_mask, _ = prune_graph(graph, node_threshold=node_threshold)
        
        # Extract structural features
        adj_matrix = graph.adjacency_matrix.cpu().numpy()
        pruned_adj = adj_matrix * edge_mask.cpu().numpy()
        
        features = []
        
        # 1. Degree distribution statistics
        in_degrees = (pruned_adj > 0).sum(axis=1)
        out_degrees = (pruned_adj > 0).sum(axis=0)
        features.extend([
            in_degrees.mean(), in_degrees.std(), in_degrees.max(),
            out_degrees.mean(), out_degrees.std(), out_degrees.max()
        ])
        
        # 2. Layer-wise feature activation patterns
        active_features = graph.active_features[graph.selected_features]
        layer_counts = np.bincount(active_features[:, 0].cpu().numpy(), 
                                  minlength=graph.cfg.n_layers)
        features.extend(layer_counts / layer_counts.sum())  # Normalized
        
        # 3. Path statistics
        G = nx.from_numpy_array(pruned_adj, create_using=nx.DiGraph)
        
        # Find paths from tokens to logits
        n_tokens = len(graph.input_tokens)
        n_logits = len(graph.logit_tokens)
        token_nodes = list(range(len(node_mask) - n_tokens - n_logits, 
                               len(node_mask) - n_logits))
        logit_nodes = list(range(len(node_mask) - n_logits, len(node_mask)))
        
        path_lengths = []
        for token in token_nodes:
            for logit in logit_nodes:
                if nx.has_path(G, token, logit):
                    path_lengths.append(nx.shortest_path_length(G, token, logit))
        
        if path_lengths:
            features.extend([np.mean(path_lengths), np.std(path_lengths), 
                           np.min(path_lengths), np.max(path_lengths)])
        else:
            features.extend([0, 0, 0, 0])
        
        return np.array(features)
    
    def find_circuit_clusters(self, category: str = None, 
                            n_components: int = 10) -> Dict[int, List[str]]:
        """Find clusters of similar circuits using PCA + DBSCAN."""
        
        # Filter graphs by category if specified
        if category:
            filtered_graphs = {k: v for k, v in self.graphs.items() 
                             if k.startswith(category)}
        else:
            filtered_graphs = self.graphs
        print(f"category -> {category}")
        # Extract fingerprints
        fingerprints = []
        graph_keys = []
        
        for key, graph in filtered_graphs.items():
            try:
                fp = self.extract_circuit_fingerprint(graph)
                fingerprints.append(fp)
                graph_keys.append(key)
            except:
                continue
        
        if len(fingerprints) < 2:
            return {}
        print(len(fingerprints))
        # Dimensionality reduction
        fingerprints = np.stack(fingerprints)
        pca = PCA(n_components=min(n_components, len(fingerprints)-1))
        reduced = pca.fit_transform(fingerprints)
        
        # Clustering (dynamic vaues for DBSCAN)
        min_samples = max(2, len(fingerprints) // 10)
        clustering = DBSCAN(eps=0.5, min_samples=min_samples).fit(reduced)
        
        # Group results
        clusters = defaultdict(list)
        for idx, label in enumerate(clustering.labels_):
            if label != -1:  # Not noise
                clusters[label].append(graph_keys[idx])
        
        return dict(clusters)
    
    def extract_common_motifs(self, cluster_graphs: List[Graph], 
                            min_support: float = 0.8) -> List[Dict]:
        """Extract common subcircuits (motifs) from a cluster of graphs."""
        
        motifs = []
        
        # Extract common active features
        feature_counts = defaultdict(int)
        total_graphs = len(cluster_graphs)
        
        for graph in cluster_graphs:
            active_features = graph.active_features[graph.selected_features]
            unique_features = set(map(tuple, active_features.cpu().numpy()))
            
            for feat in unique_features:
                feature_counts[feat] += 1
        
        # Find features that appear in most graphs
        common_features = [
            feat for feat, count in feature_counts.items()
            if count / total_graphs >= min_support
        ]
        
        if not common_features:
            return motifs
        
        # For each common feature, find common connection patterns
        for feat in common_features:
            layer, pos, feature_id = feat
            
            # Track incoming and outgoing connections
            incoming_patterns = defaultdict(int)
            outgoing_patterns = defaultdict(int)
            
            for graph in cluster_graphs:
                # Find this feature in the graph
                active_features = graph.active_features[graph.selected_features]
                feat_mask = (
                    (active_features[:, 0] == layer) & 
                    (active_features[:, 1] == pos) & 
                    (active_features[:, 2] == feature_id)
                )
                
                if not feat_mask.any():
                    continue
                
                feat_idx = torch.where(feat_mask)[0][0].item()
                adj = graph.adjacency_matrix
                
                # Incoming connections
                incoming = adj[feat_idx] > 0.1  # Threshold for significant connection
                # Outgoing connections  
                outgoing = adj[:, feat_idx] > 0.1
                
                # Record patterns (simplified - you might want more detail)
                incoming_patterns[incoming.sum().item()] += 1
                outgoing_patterns[outgoing.sum().item()] += 1
            
            motifs.append({
                'feature': (layer, pos, feature_id),
                'support': feature_counts[feat] / total_graphs,
                'avg_incoming': np.mean(list(incoming_patterns.keys())),
                'avg_outgoing': np.mean(list(outgoing_patterns.keys()))
            })
        
        return sorted(motifs, key=lambda x: x['support'], reverse=True)