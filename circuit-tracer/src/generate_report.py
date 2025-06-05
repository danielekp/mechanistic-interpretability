import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import argparse

def generate_report(results_dir: Path, output_path: Path):
    """Generate a comprehensive report of findings."""
    
    # Load results
    with open(results_dir / 'config.json') as f:
        config = json.load(f)
    
    with open(results_dir / 'category_features.json') as f:
        category_features = json.load(f)
    
    with open(results_dir / 'contrasting_features.json') as f:
        contrasting_features = json.load(f)
    
    # Create report structure
    report = f"""
        # Safety Circuit Discovery Report

        **Date**: {datetime.now().strftime('%Y-%m-%d')}
        **Model**: {config['model']}
        **Transcoder**: {config['transcoder']}

        ## Executive Summary

        This report presents the results of automated safety circuit discovery on {config['model']}.
        We analyzed {len(config['categories'])} safety categories with {config['max_feature_nodes']} maximum features per graph.

        ## Key Findings

        ### 1. Category-Specific Features

    """
    
    # Add category-specific findings
    for category, features in category_features.items():
        if features:
            report += f"\n#### {category.replace('_', ' ').title()}\n"
            report += f"- Identified {len(features)} safety-relevant features\n"
            report += f"- Top feature: {features[0][0]} (frequency: {features[0][1]['frequency']:.2%})\n"
    
    # Add more sections...
    
    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    
    generate_report(args.results_dir, args.output)