import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cosine
from typing import List, Dict, Set, Tuple
import networkx as nx
from circuit_tracer.graph import Graph
from collections import defaultdict

class CircuitPatternMiner:
    def __init__(self, graphs: Dict[str, Graph], debug: bool = False):
        self.graphs = graphs
        self.patterns = []
        self.debug = debug
        
    def extract_circuit_fingerprint(self, graph: Graph, node_threshold: float = 0.8) -> np.ndarray:
        """Extract a feature vector representing the circuit structure.
        
        Returns only the layer-wise activation pattern, which captures
        the functional organization of the circuit better than degree statistics.
        """
        try:
            # Get pruned adjacency matrix
            from circuit_tracer.graph import prune_graph
            node_mask, edge_mask, _ = prune_graph(graph, node_threshold=node_threshold)
            
            if self.debug:
                print(f"    Graph pruning: {node_mask.sum().item()}/{len(node_mask)} nodes, "
                      f"{edge_mask.sum().item()}/{edge_mask.numel()} edges")
            
            # Extract layer-wise feature activation patterns
            active_features = graph.active_features[graph.selected_features]
            
            if len(active_features) > 0:
                # Count features per layer
                layer_counts = np.bincount(active_features[:, 0].cpu().numpy(), 
                                         minlength=graph.cfg.n_layers)
                
                # Normalize to get distribution
                if layer_counts.sum() > 0:
                    layer_features = layer_counts / layer_counts.sum()
                else:
                    layer_features = layer_counts.astype(float)
            else:
                layer_features = np.zeros(graph.cfg.n_layers)
            
            if self.debug:
                print(f"    Layer features shape: {layer_features.shape}, "
                      f"sum: {layer_features.sum():.3f}")
                print(f"    Layer distribution: {layer_features}")
            
            return layer_features.astype(np.float32)
            
        except Exception as e:
            print(f"    Error in fingerprint extraction: {e}")
            # Return zero vector with expected size
            n_layers = getattr(graph.cfg, 'n_layers', 26)  # Default to 26 if not available
            return np.zeros(n_layers, dtype=np.float32)
    
    def find_circuit_clusters(self, category: str = None, 
                        distance_threshold: float = 1) -> Dict[int, List[str]]:
        """Find clusters of similar circuits using PCA + DBSCAN."""
        
        # Filter graphs by category if specified
        if category:
            if "safe_contrast" in category:
                filtered_graphs = {k: v for k, v in self.graphs.items() 
                            if k.startswith(category)}
            else:
                filtered_graphs = {k: v for k, v in self.graphs.items() 
                            if k.startswith(category) and "safe_contrast" not in k}
        else:
            filtered_graphs = self.graphs
        
        print(f"Clustering category '{category}': {len(filtered_graphs)} graphs")
        
        # Extract fingerprints
        fingerprints = []
        graph_keys = []
        
        for key, graph in filtered_graphs.items():
            try:
                fp = self.extract_circuit_fingerprint(graph)
                if not np.isnan(fp).any() and not np.isinf(fp).any():
                    fingerprints.append(fp)
                    graph_keys.append(key)
                    print(f"  Valid fingerprint for {key}: shape {fp.shape}, mean {fp.mean():.3f}")
                else:
                    print(f"Skipping graph {key}: invalid fingerprint (nan/inf)")
            except Exception as e:
                print(f"Error extracting fingerprint for {key}: {e}")
                continue
        
        print(f"Valid fingerprints: {len(fingerprints)}")
        print(f"\n{fingerprints}\n")
        if len(fingerprints) < 2:
            print(f"Not enough valid fingerprints for clustering ({len(fingerprints)})")
            return {}
        
        # Stack and analyze fingerprints
        fingerprints = np.stack(fingerprints)
        print(f"Fingerprint matrix shape: {fingerprints.shape}")
        print(f"Fingerprint stats: mean={fingerprints.mean():.3f}, std={fingerprints.std():.3f}")
        print(f"Min values: {fingerprints.min(axis=0)[:5]}")
        print(f"Max values: {fingerprints.max(axis=0)[:5]}")
        
        # Check for constant features (no variance)
        feature_stds = fingerprints.std(axis=0)
        constant_features = np.sum(feature_stds < 1e-6)
        print(f"Constant features (std < 1e-6): {constant_features}/{len(feature_stds)}")
        
        # Remove constant features
        varying_features = feature_stds >= 1e-6
        if varying_features.sum() == 0:
            print("ERROR: All features are constant!")
            return {}
        
        fingerprints_filtered = fingerprints[:, varying_features]
        print(f"After removing constant features: {fingerprints_filtered.shape}")
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        fingerprints_scaled = scaler.fit_transform(fingerprints_filtered)
        print(f"After scaling: mean={fingerprints_scaled.mean():.3f}, std={fingerprints_scaled.std():.3f}")

        # Compute pairwise distances using cosine similarity
        # (1 - cosine_similarity gives cosine distance)
        distances = pdist(fingerprints_scaled, metric='cosine')

        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='average')
        
        clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

        # Group results
        cluster_dict = defaultdict(list)
        for idx, label in enumerate(clusters):
            cluster_dict[label].append(graph_keys[idx])
        
        print(f"Final result: {len(cluster_dict)} clusters")
        for cluster_id, members in cluster_dict.items():
            print(f"  Cluster {cluster_id}: {len(members)} members")
        
        return dict(cluster_dict)
    
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
            
            # Convert numpy types to Python native types for JSON serialization
            motifs.append({
                'feature': (int(layer), int(pos), int(feature_id)),  # Convert to Python int
                'support': float(feature_counts[feat] / total_graphs),  # Convert to Python float
                'avg_incoming': float(np.mean(list(incoming_patterns.keys()))) if incoming_patterns else 0.0,
                'avg_outgoing': float(np.mean(list(outgoing_patterns.keys()))) if outgoing_patterns else 0.0,
                'incoming_pattern_counts': {int(k): int(v) for k, v in incoming_patterns.items()},  # Convert keys and values
                'outgoing_pattern_counts': {int(k): int(v) for k, v in outgoing_patterns.items()}   # Convert keys and values
            })
        
        return sorted(motifs, key=lambda x: x['support'], reverse=True)