#!/usr/bin/env python3

import torch
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).parent.parent))

from circuit_tracer import ReplacementModel
from circuit_tracer.graph import Graph
from data.safety_benchmark import SafetyBenchmark
from safety_circuit_discovery import SafetyCircuitAnalyzer
from safety_circuit_viz import SafetyCircuitVisualizer

def test_enhanced_dashboard():
    """Test the enhanced dashboard with transcoder token data."""
    
    print("Testing enhanced dashboard with transcoder token data...")
    
    try:
        # Load the existing analysis results
        results_dir = Path("safety_analysis_results")
        
        if not results_dir.exists():
            print("Error: safety_analysis_results directory not found")
            return
        
        # Load model (same as used in analysis)
        print("Loading model...")
        model = ReplacementModel.from_pretrained(
            'google/gemma-2-2b',
            'gemma',
            dtype=torch.bfloat16,
            device='cpu'  # Use CPU for testing
        )
        
        # Load benchmark
        print("Loading benchmark...")
        benchmark = SafetyBenchmark()
        
        # Create analyzer and load existing graphs
        print("Loading existing analysis...")
        analyzer = SafetyCircuitAnalyzer(model, benchmark)
        
        # Load existing graphs
        graphs_dir = results_dir / "graphs"
        if graphs_dir.exists():
            for category_dir in graphs_dir.iterdir():
                if category_dir.is_dir():
                    category = category_dir.name
                    for graph_file in category_dir.glob("*.pt"):
                        try:
                            graph = analyzer.graphs[graph_file.stem] = Graph.from_pt(graph_file)
                            # Reconstruct feature stats from graph
                            # This is a simplified version - you might want to load from saved stats
                        except Exception as e:
                            print(f"Error loading graph {graph_file}: {e}")
        
        # Load category features
        category_features_file = results_dir / "category_features.json"
        if category_features_file.exists():
            with open(category_features_file, 'r') as f:
                category_features = json.load(f)
            print(f"Loaded category features for {len(category_features)} categories")
        else:
            print("Error: category_features.json not found")
            return
        
        # Create visualizer
        print("Creating visualizer...")
        visualizer = SafetyCircuitVisualizer(analyzer)
        
        # Test transcoder token extraction
        print("Testing transcoder token extraction...")
        test_feature = "L15_F851"  # From your error message
        tokens = visualizer._get_top_transcoder_tokens(test_feature, k=5)
        print(f"Top 5 tokens for {test_feature}: {tokens}")
        
        # Create enhanced dashboard
        print("Creating enhanced dashboard...")
        dashboard_dir = results_dir / "dashboard"
        visualizer.create_feature_importance_dashboard(dashboard_dir)
        
        print("✓ Enhanced dashboard created successfully!")
        print(f"Dashboard location: {dashboard_dir / 'index.html'}")
        
    except Exception as e:
        print(f"Error creating enhanced dashboard: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_dashboard() 