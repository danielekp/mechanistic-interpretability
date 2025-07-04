import argparse
from pathlib import Path
import torch
import json
from datetime import datetime
import sys 

sys.path.append(str(Path(__file__).parent.parent))

from circuit_tracer import ReplacementModel
from data.safety_benchmark import SafetyBenchmark
from safety_circuit_discovery import SafetyCircuitAnalyzer
from circuit_pattern_mining import CircuitPatternMiner
from safety_circuit_viz import SafetyCircuitVisualizer

def main(args):
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save run configuration
    config = {
        'model': args.model,
        'transcoder': args.transcoder,
        'max_feature_nodes': args.max_feature_nodes,
        'categories': args.categories,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("=== Safety Circuit Discovery Pipeline ===")
    
    # 1. Load model
    print(f"\n1. Loading model {args.model}...")
    model = ReplacementModel.from_pretrained(
        args.model, 
        args.transcoder, 
        dtype=torch.bfloat16,
        device=args.device
    )
    
    # 2. Create benchmark
    print("\n2. Creating safety benchmark...")
    benchmark = SafetyBenchmark()
    
    # 3. Run attribution analysis
    print(f"\n3. Running attribution on {len(benchmark.prompts)} prompts...")
    analyzer = SafetyCircuitAnalyzer(model, benchmark)
    
    analyzer.collect_attributions(
        output_dir / 'graphs',
        max_feature_nodes=args.max_feature_nodes,
        categories=args.categories
    )
    
    # 4. Find safety-relevant features
    print("\n4. Analyzing safety-relevant features...")
    category_features = analyzer.find_category_specific_features(
        min_frequency=args.min_frequency,
        min_activation=args.min_activation
    )
    
    # Save features
    with open(output_dir / 'category_features.json', 'w') as f:
        json.dump(category_features, f, indent=2)
    
    # 5. Find contrasting features
    print("\n5. Finding contrasting features...")
    contrasting_features = analyzer.find_contrasting_features()
    
    with open(output_dir / 'contrasting_features.json', 'w') as f:
        json.dump(contrasting_features, f, indent=2)
    
    # 6. Mine circuit patterns
    if args.mine_patterns:
        print("\n6. Mining circuit patterns...")
        miner = CircuitPatternMiner()
        
        for category in category_features.keys():
            # Extract motifs from graphs in the category
            if "safe_contrast" in category:
                category_graphs = [analyzer.graphs[k] for k in analyzer.graphs.keys() 
                                  if k.startswith(category)]
            else:
                category_graphs = [analyzer.graphs[k] for k in analyzer.graphs.keys() 
                                  if k.startswith(category) and "safe_contrast" not in k]
            motifs = miner.analyze_common_feature_connectivity(category_graphs, min_support=args.min_support_motifs)
            
            with open(output_dir / f'{category}_motifs.json', 'w') as f:
                json.dump(motifs, f, indent=2)
    
    # 7. Create visualizations
    print("\n7. Creating visualizations...")
    visualizer = SafetyCircuitVisualizer(analyzer)
    visualizer.create_feature_importance_dashboard(
        output_dir / 'dashboard',
        node_threshold=args.node_threshold,
        edge_threshold=args.edge_threshold
    )
    
    print(f"\n✓ Analysis complete! Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Safety Circuit Discovery Pipeline")
    
    parser.add_argument('--model', type=str, default='google/gemma-2-2b',
                       help='Model to analyze')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Which device utilize for computation. Default cuda')
    parser.add_argument('--transcoder', type=str, default='gemma',
                       help='Transcoder set to use. Default gemma')
    parser.add_argument('--output-dir', type=str, default='./safety_analysis',
                       help='Output directory for results. Default ./safety_analysis')
    parser.add_argument('--max-feature-nodes', type=int, default=4096,
                       help='Maximum feature nodes for attribution. Default 4096')
    parser.add_argument('--categories', type=str, nargs='+',
                       default=['deception', 'manipulation', 'power_seeking', 'harmful_content', 'deception_safe_contrast','manipulation_safe_contrast','power_seeking_safe_contrast','harmful_content_safe_contrast'],
                       help='Safety categories to analyze.')
    parser.add_argument('--min-frequency', type=float, default=0.3,
                       help='Minimum frequency for category-specific features. Default 0')
    parser.add_argument('--min-activation', type=float, default=0.1,
                       help='Minimum activation for category-specific features. Default 0.1')
    parser.add_argument('--mine-patterns', action='store_true',
                       help='Mine circuit patterns.')
    parser.add_argument('--min-support-motifs', type=float, default=0.3,
                       help='Minimum support for motifs. Default 0.3')
    parser.add_argument('--node-threshold', type=float, default=0.8,
                       help='Node threshold for attribution graphs. Default 0.8')
    parser.add_argument('--edge-threshold', type=float, default=0.98,
                       help='Edge threshold for attribution graphs. Default 0.98')
    
    args = parser.parse_args()
    main(args)
