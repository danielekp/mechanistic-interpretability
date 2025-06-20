import torch
import numpy as np
from typing import List, Dict
from circuit_tracer.graph import Graph
from collections import defaultdict
from tqdm import tqdm

class CircuitPatternMiner:
    
    def analyze_common_feature_connectivity(self, cluster_graphs: List[Graph], 
                            min_support: float = 0.8) -> List[Dict]:
        """Extract common subcircuits (motifs) from a cluster of graphs."""
        
        motifs = []
        
        # Extract common active features
        feature_counts = defaultdict(int)
        total_graphs = len(cluster_graphs)
        
        print(f"Extracting motifs from {total_graphs} graphs...")
        for graph in tqdm(cluster_graphs, desc="Processing graphs for motifs"):
            active_features = graph.active_features[graph.selected_features]
            unique_features = set(map(tuple, active_features.cpu().numpy()))
            
            for feat in unique_features:
                feature_counts[feat] += 1
        
        common_features = [
            feat for feat, count in feature_counts.items()
            if count / total_graphs >= min_support
        ]
        
        if not common_features:
            return motifs
        
        print(f"Analyzing {len(common_features)} common features...")
        for feat in tqdm(common_features, desc="Analyzing common features"):
            layer, pos, feature_id = feat
            
            # Track incoming and outgoing connections
            incoming_patterns = defaultdict(list)  # Changed from int to list
            outgoing_patterns = defaultdict(list)  # Changed from int to list
            
            for graph in tqdm(cluster_graphs, desc=f"Processing graphs for feature {feat[2]}", leave=False):
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
                
                incoming = adj[feat_idx] > 0.1  # Threshold for significant connection
                 
                outgoing = adj[:, feat_idx] > 0.1
                
                # Store average weights instead of just counts
                incoming_patterns[incoming.sum().item()].append(adj[feat_idx][incoming].mean().item() if incoming.any() else 0.0)
                outgoing_patterns[outgoing.sum().item()].append(adj[:, feat_idx][outgoing].mean().item() if outgoing.any() else 0.0)
            
            # Convert to average weights per connection count
            avg_incoming = {k: np.mean(v) for k, v in incoming_patterns.items()}
            avg_outgoing = {k: np.mean(v) for k, v in outgoing_patterns.items()}
            
            # Convert numpy types to Python native types for JSON serialization
            motifs.append({
                'feature': (int(layer), int(pos), int(feature_id)),  # Convert to Python int
                'support': float(feature_counts[feat] / total_graphs),  # Convert to Python float
                'avg_incoming': float(np.mean(list(avg_incoming.keys()))) if avg_incoming else 0.0,
                'avg_outgoing': float(np.mean(list(avg_outgoing.keys()))) if avg_outgoing else 0.0,
                'incoming_pattern_counts': avg_incoming,  # Convert to Python dict
                'outgoing_pattern_counts': avg_outgoing   # Convert to Python dict
            })
        
        return sorted(motifs, key=lambda x: x['support'], reverse=True)