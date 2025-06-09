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
        
        try:
            # Get pruned adjacency matrix
            from circuit_tracer.graph import prune_graph
            node_mask, edge_mask, _ = prune_graph(graph, node_threshold=node_threshold)
            
            print(f"    Graph pruning: {node_mask.sum().item()}/{len(node_mask)} nodes, {edge_mask.sum().item()}/{edge_mask.numel()} edges")
            
            # Extract structural features
            adj_matrix = graph.adjacency_matrix.cpu().numpy()
            pruned_adj = adj_matrix * edge_mask.cpu().numpy()
            
            features = []
            
            # 1. Degree distribution statistics
            in_degrees = (pruned_adj > 0).sum(axis=1)
            out_degrees = (pruned_adj > 0).sum(axis=0)
            
            degree_features = [
                in_degrees.mean() if len(in_degrees) > 0 else 0,
                in_degrees.std() if len(in_degrees) > 0 else 0,
                in_degrees.max() if len(in_degrees) > 0 else 0,
                out_degrees.mean() if len(out_degrees) > 0 else 0,
                out_degrees.std() if len(out_degrees) > 0 else 0,
                out_degrees.max() if len(out_degrees) > 0 else 0
            ]
            features.extend(degree_features)
            print(f"    Degree features: {degree_features}")
            
            # 2. Layer-wise feature activation patterns
            active_features = graph.active_features[graph.selected_features]
            if len(active_features) > 0:
                layer_counts = np.bincount(active_features[:, 0].cpu().numpy(), 
                                        minlength=graph.cfg.n_layers)
                # Normalize only if sum > 0
                if layer_counts.sum() > 0:
                    layer_features = layer_counts / layer_counts.sum()
                else:
                    layer_features = layer_counts.astype(float)
            else:
                layer_features = np.zeros(graph.cfg.n_layers)
            
            features.extend(layer_features)
            print(f"    Layer features shape: {layer_features.shape}, sum: {layer_features.sum()}")
            
            # 3. Path statistics (simplified to avoid networkx issues)
            # Just use basic graph connectivity measures
            if pruned_adj.sum() > 0:
                # Graph density
                n_nodes = node_mask.sum().item()
                max_edges = n_nodes * (n_nodes - 1)
                density = edge_mask.sum().item() / max_edges if max_edges > 0 else 0
                
                # Average node degree
                avg_degree = (in_degrees.mean() + out_degrees.mean()) / 2 if len(in_degrees) > 0 else 0
                
                path_features = [density, avg_degree, 0, 0]  # Simplified path stats
            else:
                path_features = [0, 0, 0, 0]
            
            features.extend(path_features)
            print(f"    Path features: {path_features}")
            
            feature_vector = np.array(features, dtype=np.float32)
            print(f"    Final feature vector shape: {feature_vector.shape}, has nan: {np.isnan(feature_vector).any()}, has inf: {np.isinf(feature_vector).any()}")
            
            # Replace any remaining nan/inf with 0
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_vector
            
        except Exception as e:
            print(f"    Error in fingerprint extraction: {e}")
            # Return zero vector as fallback
            return np.zeros(50, dtype=np.float32)  # Fixed size fallback
    
    def find_circuit_clusters(self, category: str = None, 
                        n_components: int = 10) -> Dict[int, List[str]]:
        """Find clusters of similar circuits using PCA + DBSCAN."""
        
        # Filter graphs by category if specified
        if category:
            filtered_graphs = {k: v for k, v in self.graphs.items() 
                            if k.startswith(category)}
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
        
        # Dimensionality reduction
        n_components = min(n_components, len(fingerprints)-1, fingerprints_scaled.shape[1])
        
        if n_components < 1:
            print("Cannot perform PCA: insufficient data")
            return {}
        
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(fingerprints_scaled)
        print(f"PCA shape: {reduced.shape}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_[:min(5, len(pca.explained_variance_ratio_))]}")
        
        # Calculate pairwise distances to help choose eps
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(reduced)
        # Get upper triangle (excluding diagonal)
        upper_tri_indices = np.triu_indices_from(distances, k=1)
        pairwise_dists = distances[upper_tri_indices]
        
        print(f"Pairwise distance stats:")
        print(f"  Mean: {pairwise_dists.mean():.3f}")
        print(f"  Std: {pairwise_dists.std():.3f}")
        print(f"  Min: {pairwise_dists.min():.3f}")
        print(f"  Max: {pairwise_dists.max():.3f}")
        print(f"  25th percentile: {np.percentile(pairwise_dists, 25):.3f}")
        print(f"  50th percentile: {np.percentile(pairwise_dists, 50):.3f}")
        print(f"  75th percentile: {np.percentile(pairwise_dists, 75):.3f}")
        
        # Try multiple eps values
        eps_candidates = [
            np.percentile(pairwise_dists, 10),
            np.percentile(pairwise_dists, 25),
            np.percentile(pairwise_dists, 50),
            pairwise_dists.mean(),
            np.percentile(pairwise_dists, 75)
        ]
        
        min_samples_candidates = [2, 3, max(2, len(fingerprints) // 10)]
        
        best_clustering = None
        best_score = -1
        best_params = None
        
        for eps in eps_candidates:
            for min_samples in min_samples_candidates:
                if min_samples >= len(fingerprints):
                    continue
                    
                print(f"\nTrying DBSCAN with eps={eps:.3f}, min_samples={min_samples}")
                
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(reduced)
                
                n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
                n_noise = list(clustering.labels_).count(-1)
                
                print(f"  Result: {n_clusters} clusters, {n_noise} noise points")
                
                # Score clustering (prefer fewer noise points and reasonable number of clusters)
                if n_clusters > 0:
                    score = n_clusters - (n_noise / len(fingerprints))
                    print(f"  Score: {score:.3f}")
                    
                    if score > best_score:
                        best_clustering = clustering
                        best_score = score
                        best_params = (eps, min_samples)
        
        if best_clustering is None:
            print("No successful clustering found with any parameters")
            return {}
        
        print(f"\nBest clustering: eps={best_params[0]:.3f}, min_samples={best_params[1]}")
        
        # Group results using best clustering
        clusters = defaultdict(list)
        noise_count = 0
        
        for idx, label in enumerate(best_clustering.labels_):
            if label != -1:  # Not noise
                clusters[label].append(graph_keys[idx])
            else:
                noise_count += 1
        
        print(f"Final result: {len(clusters)} clusters, {noise_count} noise points")
        for cluster_id, members in clusters.items():
            print(f"  Cluster {cluster_id}: {len(members)} members")
        
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