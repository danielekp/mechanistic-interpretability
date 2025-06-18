import torch
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import sys

from circuit_tracer import ReplacementModel, attribute
from circuit_tracer.graph import Graph

sys.path.append(str(Path(__file__).parent.parent))
from data.safety_benchmark import SafetyBenchmark
from data.safety_prompt import SafetyPrompt

class SafetyCircuitAnalyzer:
    def __init__(self, model: ReplacementModel, benchmark: SafetyBenchmark):
        self.model = model
        self.benchmark = benchmark
        self.graphs = {}
        self.feature_stats = defaultdict(lambda: defaultdict(list))
        
    def collect_attributions(self, 
                           output_dir: Path,
                           max_feature_nodes: int = 4096,
                           categories: List[str] = None,
                           force_recompute: bool = False):
        """Run attribution on all benchmark prompts and save graphs.
        
        Args:
            output_dir: Directory to save/load graphs
            max_feature_nodes: Maximum number of feature nodes to consider
            categories: List of categories to process. If None, processes all categories
            force_recompute: If True, recomputes attributions even if graphs exist
        """
        
        categories = categories or list(set(p.category for p in self.benchmark.prompts))
        output_dir.mkdir(exist_ok=True, parents=True)
        
        for category in categories:
            cat_dir = output_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            prompts = self.benchmark.get_by_category(category)
            
            for i, safety_prompt in enumerate(tqdm(prompts, desc=f"Processing {category}")):
                graph_path = cat_dir / f"{category}_{i:03d}.pt"
                
                # Skip if graph exists and we're not forcing recomputation
                if graph_path.exists() and not force_recompute:
                    try:
                        graph = Graph.from_pt(graph_path)
                        self.graphs[f"{category}_{i}"] = graph
                        self._collect_feature_stats(graph, safety_prompt)
                        continue
                    except Exception as e:
                        print(f"Error loading existing graph {graph_path}: {e}")
                        # If loading fails, we'll recompute below
                
                try:
                    # Run attribution
                    graph = attribute(
                        prompt=safety_prompt.prompt,
                        model=self.model,
                        max_n_logits=10,
                        desired_logit_prob=0.95,
                        max_feature_nodes=max_feature_nodes,
                        batch_size=128,
                        verbose=False
                    )
                    
                    # Save graph
                    graph.to_pt(graph_path)
                    
                    # Store in memory for analysis
                    self.graphs[f"{category}_{i}"] = graph
                    
                    # Collect feature statistics
                    self._collect_feature_stats(graph, safety_prompt)
                    
                except Exception as e:
                    print(f"Error processing {safety_prompt.prompt}: {e}")
                    continue
    
    def _collect_feature_stats(self, graph: Graph, safety_prompt: SafetyPrompt):
        """Collect statistics about which features activate for each category."""
        
        # Get active features and their activation values
        active_features = graph.active_features[graph.selected_features]
        activation_values = graph.activation_values
        
        for feat_idx, (layer, pos, feature_id) in enumerate(active_features):
            feature_key = f"L{layer}_F{feature_id}"
            activation = activation_values[feat_idx].item()
            
            self.feature_stats[safety_prompt.category][feature_key].append({
                'activation': activation,
                'position': pos.item(),
                'prompt': safety_prompt.prompt,
                'severity': safety_prompt.severity
            })
    
    def find_category_specific_features(self, min_frequency: float = 0.3, 
                                      min_activation: float = 0.1) -> Dict[str, List[str]]:
        """Find features that consistently activate for specific safety categories."""
        
        category_features = {}
        
        for category in self.feature_stats:
            prompts_in_category = len(self.benchmark.get_by_category(category))
            feature_frequencies = {}
            
            for feature, activations in self.feature_stats[category].items():
                # Calculate frequency of activation
                frequency = len(activations) / prompts_in_category
                # Calculate average activation strength
                avg_activation = np.mean([a['activation'] for a in activations])
                if frequency >= min_frequency and avg_activation >= min_activation:
                    feature_frequencies[feature] = {
                        'frequency': frequency,
                        'avg_activation': avg_activation,
                        'examples': activations[:5]
                    }
            
            # Sort by frequency * avg_activation
            sorted_features = sorted(
                feature_frequencies.items(),
                key=lambda x: x[1]['frequency'] * x[1]['avg_activation'],
                reverse=True
            )
            
            category_features[category] = sorted_features[:20]  # Top 20 features
        
        return category_features
    
    def find_contrasting_features(self, safe_categories: List[str] = ["deception_safe_contrast", "harmful_content_safe_contrast", "power_seeking_safe_contrast", "manipulation_safe_contrast"]) -> Dict[str, Dict]:
        """Find features that differentiate safe vs unsafe responses."""
        
        contrastive_features = {}
        
        unsafe_categories = [c for c in self.feature_stats.keys() if c not in safe_categories]
        
        safe_features = set()
        for safe_category in safe_categories:
            if safe_category in self.feature_stats:
                safe_features.update(self.feature_stats[safe_category].keys())
        for unsafe_cat in unsafe_categories:
            unsafe_features = set(self.feature_stats[unsafe_cat].keys())
            
            # Features unique to unsafe
            unsafe_only = unsafe_features - safe_features
            
            # Features that activate differently
            shared_features = unsafe_features & safe_features

            differential_features = []
            
            for feat in shared_features:
                safe_acts = []
                for safe_cat in safe_categories:
                    if safe_cat in self.feature_stats and feat in self.feature_stats[safe_cat]:
                        safe_acts.extend([a['activation'] for a in self.feature_stats[safe_cat][feat]])
                unsafe_acts = [a['activation'] for a in self.feature_stats[unsafe_cat][feat]]

                if safe_acts and unsafe_acts:
                    # Statistical test (simplified - you might want t-test)
                    safe_mean = np.mean(safe_acts)
                    unsafe_mean = np.mean(unsafe_acts)
                    
                    if abs(unsafe_mean - safe_mean) > 0.2:  # Significant difference

                        differential_features.append({
                            'feature': feat,
                            'safe_activation': safe_mean,
                            'unsafe_activation': unsafe_mean,
                            'difference': unsafe_mean - safe_mean
                        })
            
            contrastive_features[unsafe_cat] = {
                'unique_to_unsafe': list(unsafe_only)[:10],
                'differential': sorted(differential_features, 
                                     key=lambda x: abs(x['difference']), 
                                     reverse=True)[:10]
            }
        return contrastive_features
