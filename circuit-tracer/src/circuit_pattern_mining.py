import torch
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, cosine
from typing import List, Dict, Set, Tuple, Optional
import networkx as nx
from circuit_tracer.graph import Graph, prune_graph
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CircuitPatternMiner:
    def __init__(self, graphs: Dict[str, Graph], debug: bool = False):
        self.graphs = graphs
        self.patterns = []
        self.debug = debug
        self.pca = None
        self.scaler = StandardScaler()
        
    def _compute_layer_feature_importance(self, graph: Graph) -> np.ndarray:
        """Compute layer importance scores based on activation and connectivity.
        
        The function return a vector of importance scores normalized to [0,1] for each layer, with a final shape (n_layers,).
        The steps are the following:
        1. For each feature, compute the incoming and outgoing importance.
        2. Compute the importance score for each feature, multiplying the activation value by the sum of the incoming and outgoing importance.
            A feature with high incoming importance but low outgoing importance might be a "sink" that collects information
            A feature with low incoming importance but high outgoing importance might be a "source" that distributes information
            A feature with both high incoming and outgoing importance might be a crucial "hub" in the circuit
            A feature with low values for both might be less important to the circuit's function
        3. Sum the importance scores for all features in the layer to get the layer importance score.
        4. Normalize the layer importance scores to [0,1].
        """
        try:
            active_features = graph.active_features[graph.selected_features]
            activation_values = graph.activation_values
            
            n_layers = graph.cfg.n_layers
            feature_importance = np.zeros(n_layers)
            
            for feat_idx, (layer, pos, feature_id) in enumerate(active_features):
                activation = activation_values[feat_idx].item()
                adj = graph.adjacency_matrix
                incoming_importance = adj[feat_idx].sum().item()
                outgoing_importance = adj[:, feat_idx].sum().item()
                
                # TODO: Add a weight for incoming and outgoing importance
                importance = activation * (incoming_importance + outgoing_importance)
                feature_importance[layer] += importance
            
            # Normalize to [0,1]
            if feature_importance.max() > 0:
                feature_importance = feature_importance / feature_importance.max()
                
            return feature_importance
            
        except Exception as e:
            logger.error(f"Error computing feature importance: {e}")
            return np.zeros(graph.cfg.n_layers)
    
    def _analyze_information_flow(self, graph: Graph) -> np.ndarray:
        """Analyze information flow patterns through the circuit.
        
        The steps are the following:
        1. Find the active features in the graph of each layer and the next layer using the adjacency matrix.
        2. Compute the flow strength as the mean of the connection weights between the active features of the current layer and the next layer.
        3. Compute the diversity of the flow as the standard deviation of the connection weights between the active features of the current layer and the next layer.
        4. Combine the flow and the diversity to get the flow pattern for the current layer.
        5. Normalize the flow pattern to [0,1].
        """
        try:
            active_features = graph.active_features[graph.selected_features]
            n_layers = graph.cfg.n_layers
            actual_size = len(active_features)
            flow_patterns = np.zeros(n_layers)
            
            # Get the device of the adjacency matrix
            adj = graph.adjacency_matrix
            device = adj.device
            
            for layer in range(n_layers - 1):
                # Create layer masks using actual tensor size and correct device
                layer_mask = torch.zeros(actual_size, dtype=torch.bool, device=device)
                next_layer_mask = torch.zeros(actual_size, dtype=torch.bool, device=device)
                
                layer_mask[active_features[:, 0] == layer] = True
                next_layer_mask[active_features[:, 0] == layer + 1] = True
                
                if layer_mask.any() and next_layer_mask.any():
                    layer_indices = torch.where(layer_mask)[0].to(device)
                    next_layer_indices = torch.where(next_layer_mask)[0].to(device)
                    
                    adj = graph.adjacency_matrix
                    layer_conn = adj[layer_indices][:, next_layer_indices]
                    
                    flow_strength = layer_conn.mean().item()
                    flow_diversity = layer_conn.std().item()
                    
                    flow_patterns[layer] = flow_strength * (1 + flow_diversity)
            
            # Normalize to [0,1]
            if flow_patterns.max() > 0:
                flow_patterns = flow_patterns / flow_patterns.max()
            
            return flow_patterns
            
        except Exception as e:
            logger.error(f"Error analyzing information flow: {e}")
            return np.zeros(n_layers)
    
    def _compute_robustness_metrics(self, graph: Graph) -> np.ndarray:
        """Compute metrics related to circuit robustness and redundancy.
        
        The steps are the following:
        1. Compute the average clustering coefficient. It shows how resilient the circuit is to local failures.
            If neighbors are well-connected, information can find alternative paths.
        2. Compute the network density. Important for understanding the circuit's capacity for information flow.
        3. Compute the average node connectivity. It is a global measure of the circuit's resilience to failures.
        4. Compute the redundancy as the number of alternative paths between all pairs of nodes.
        5. Normalize the robustness metrics to [0,1].
        """
        try:
            G = nx.from_numpy_array(graph.adjacency_matrix.cpu().numpy())
            
            # Calculate robustness metrics
            metrics = []
            
            # 1. Average clustering coefficient
            metrics.append(nx.average_clustering(G))
            
            # 2. Network density
            metrics.append(nx.density(G))
            
            # 3. Average node connectivity
            if nx.is_connected(G):
                metrics.append(nx.average_node_connectivity(G))
            else:
                metrics.append(0)
            
            # 4. Redundancy (number of alternative paths)
            metrics.append(self._compute_redundancy(G))
            
            return np.array(metrics)
            
        except Exception as e:
            logger.error(f"Error computing robustness metrics: {e}")
            return np.zeros(4)
        
    def _compute_redundancy(self, G: nx.Graph) -> float:
        """Compute redundancy more efficiently by using edge connectivity.
        
        Args:
            G: NetworkX graph representing the circuit
            
        Returns:
            float: Redundancy score between 0 and 1
        """
        try:
            if G.number_of_edges() == 0:
                return 0.0
                
            # Get all edges
            edges = list(G.edges())
            redundancy_count = 0
            
            # For each edge, check if there's an alternative path
            for u, v in edges:
                # Remove the edge temporarily
                G.remove_edge(u, v)
                
                # Check if there's still a path between u and v
                # Using BFS which is faster than finding all paths
                try:
                    # Try to find a path of length > 1
                    path = nx.shortest_path(G, u, v)
                    if len(path) > 2:  # Path length > 1 means there's an alternative route
                        redundancy_count += 1
                except nx.NetworkXNoPath:
                    pass
                    
                # Add the edge back
                G.add_edge(u, v)
            
            return redundancy_count / len(edges)
            
        except Exception as e:
            logger.error(f"Error computing redundancy: {e}")
            return 0.0
    
    def extract_circuit_fingerprint(self, graph: Graph, node_threshold: float = 0.8) -> np.ndarray:
        """Extract a comprehensive feature vector representing the circuit structure."""
        try:
            # Get pruned adjacency matrix with dynamic sizing
            node_mask, edge_mask, _ = prune_graph(graph, node_threshold=node_threshold)
            actual_size = node_mask.sum().item()
            
            if self.debug:
                logger.info(f"Graph pruning: {actual_size}/{len(node_mask)} nodes, "
                          f"{edge_mask.sum().item()}/{edge_mask.numel()} edges")
            
            # Extract different aspects of the circuit
            feature_importance = self._compute_layer_feature_importance(graph)
            flow_patterns = self._analyze_information_flow(graph)
            robustness_metrics = self._compute_robustness_metrics(graph)
            
            # Combine all features
            fingerprint = np.concatenate([
                feature_importance,      # Layer-wise feature importance
                flow_patterns,           # Information flow patterns
                robustness_metrics       # Circuit robustness metrics
            ])
            
            return fingerprint.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error in fingerprint extraction: {e}")
            # Return zero vector with expected size
            n_layers = getattr(graph.cfg, 'n_layers', 26)
            return np.zeros(n_layers * 4 + 4, dtype=np.float32)  # Adjusted size for new features
    
    def find_circuit_clusters(self, category: str = None, 
                            distance_threshold: float = 1,
                            n_components: int = 10) -> Dict[int, List[str]]:
        """Find clusters of similar circuits using improved fingerprinting and dimensionality reduction."""
        
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
        
        logger.info(f"Clustering category '{category}': {len(filtered_graphs)} graphs")
        
        # Extract fingerprints
        fingerprints = []
        graph_keys = []
        
        for key, graph in filtered_graphs.items():
            try:
                fp = self.extract_circuit_fingerprint(graph)
                if not np.isnan(fp).any() and not np.isinf(fp).any():
                    fingerprints.append(fp)
                    graph_keys.append(key)
                    logger.info(f"Valid fingerprint for {key}: shape {fp.shape}, mean {fp.mean():.3f}")
                else:
                    logger.warning(f"Skipping graph {key}: invalid fingerprint (nan/inf)")
            except Exception as e:
                logger.error(f"Error extracting fingerprint for {key}: {e}")
                continue
        
        if len(fingerprints) < 2:
            logger.warning(f"Not enough valid fingerprints for clustering ({len(fingerprints)})")
            return {}
        
        # Stack and preprocess fingerprints
        fingerprints = np.stack(fingerprints)
        
        # Remove constant features
        feature_stds = fingerprints.std(axis=0)
        varying_features = feature_stds >= 1e-6
        fingerprints_filtered = fingerprints[:, varying_features]
        
        # Standardize features
        fingerprints_scaled = self.scaler.fit_transform(fingerprints_filtered)
        
        # Apply PCA for dimensionality reduction
        if self.pca is None:
            self.pca = PCA(n_components=min(n_components, len(fingerprints_scaled[0])))
        fingerprints_reduced = self.pca.fit_transform(fingerprints_scaled)
        
        # Compute pairwise distances using cosine similarity
        distances = pdist(fingerprints_reduced, metric='cosine')
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(distances, method='average')
        clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
        
        # Group results
        cluster_dict = defaultdict(list)
        for idx, label in enumerate(clusters):
            cluster_dict[label].append(graph_keys[idx])
        
        logger.info(f"Final result: {len(cluster_dict)} clusters")
        for cluster_id, members in cluster_dict.items():
            logger.info(f"Cluster {cluster_id}: {len(members)} members")
        
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