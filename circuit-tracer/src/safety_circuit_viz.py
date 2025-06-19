from pathlib import Path
import json
from typing import Dict, List
import re

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import re

from safety_circuit_discovery import SafetyCircuitAnalyzer

class SafetyCircuitVisualizer:
    def __init__(self, analyzer: SafetyCircuitAnalyzer):
        self.analyzer = analyzer
        self.model = analyzer.model
        
    def _get_top_transcoder_tokens(self, feature_key: str, k: int = 5):
        """Get the top-k semantic tokens for a feature using the transcoder."""
        match = re.match(r'L(\d+)_F(\d+)', feature_key)
        if not match:
            return []
        layer = int(match.group(1))
        feature_id = int(match.group(2))
        try:
            transcoder = self.model.transcoders[layer]
            logits = transcoder.W_dec[feature_id]  # shape: [vocab_size]
            topk = torch.topk(logits, k)
            token_ids = topk.indices.tolist()
            scores = topk.values.tolist()
            tokens = [self.model.tokenizer.decode([tid]) for tid in token_ids]
            return list(zip(tokens, scores))
        except Exception as e:
            return []

    def create_feature_heatmap(self, category_features: Dict[str, List]) -> go.Figure:
        """Create a heatmap showing feature activation patterns with transcoder token data."""
        # Prepare data
        categories = list(category_features.keys())
        all_features = set()
        for features in category_features.values():
            all_features.update([f[0] for f in features[:10]])  # Top 10 per category
        all_features = sorted(list(all_features))
        # Create matrix
        matrix = np.zeros((len(categories), len(all_features)))
        # Prepare hover text with transcoder token data
        hover_text = []
        for i, cat in enumerate(categories):
            cat_features = {f[0]: f[1]['frequency'] * f[1]['avg_activation'] 
                          for f in category_features[cat]}
            row_hover = []
            for j, feat in enumerate(all_features):
                value = cat_features.get(feat, 0)
                matrix[i, j] = value
                if value > 0:
                    tokens_scores = self._get_top_transcoder_tokens(feat, k=5)
                    if tokens_scores:
                        token_info = [f"'{tok}': {score:.2f}" for tok, score in tokens_scores]
                        hover_text_entry = f"<b>{feat}</b><br>Category: {cat}<br>Importance: {value:.3f}<br>Top tokens: {', '.join(token_info)}"
                    else:
                        hover_text_entry = f"<b>{feat}</b><br>Category: {cat}<br>Importance: {value:.3f}<br>No transcoder data"
                else:
                    hover_text_entry = f"<b>{feat}</b><br>Category: {cat}<br>Importance: 0.000"
                row_hover.append(hover_text_entry)
            hover_text.append(row_hover)
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_features,
            y=categories,
            colorscale='RdYlBu_r',
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Importance Score"),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_text
        ))
        fig.update_layout(
            title="Safety-Relevant Feature Activation Patterns with Transcoder Token Data",
            xaxis_title="Features",
            yaxis_title="Safety Categories",
            height=600,
            width=1200
        )
        return fig
    
    def create_transcoder_token_table(self, category_features: Dict[str, List]) -> go.Figure:
        """Create a detailed table showing transcoder tokens for top features."""
        
        table_data = []
        
        for category, features in category_features.items():
            for feature_key, feature_info in features[:5]:  # Top 5 features per category
                tokens_scores = self._get_top_transcoder_tokens(feature_key, k=5)
                
                if tokens_scores:
                    for i, (token, score) in enumerate(tokens_scores):
                        table_data.append({
                            'Category': category,
                            'Feature': feature_key,
                            'Transcoder Token': token,
                            'Score': f"{score:.3f}",
                            'Rank': i + 1,
                            'Avg Activation': f"{feature_info['avg_activation']:.3f}",
                            'Frequency': f"{feature_info['frequency']*100:.1f}%"
                        })
                else:
                    # Add row for features without transcoder data
                    table_data.append({
                        'Category': category,
                        'Feature': feature_key,
                        'Transcoder Token': 'N/A',
                        'Score': 'N/A',
                        'Rank': 'N/A',
                        'Avg Activation': f"{feature_info['avg_activation']:.3f}",
                        'Frequency': f"{feature_info['frequency']*100:.1f}%"
                    })
        
        if not table_data:
            # Create empty table
            fig = go.Figure(data=go.Table(
                header=dict(values=['Category', 'Feature', 'Transcoder Token', 'Score', 'Rank', 'Avg Activation', 'Frequency']),
                cells=dict(values=[[], [], [], [], [], [], []])
            ))
            fig.update_layout(title="No transcoder token data available")
            return fig
        
        df = pd.DataFrame(table_data)
        
        fig = go.Figure(data=go.Table(
            header=dict(
                values=list(df.columns),
                fill_color='paleturquoise',
                align='left',
                font=dict(size=12)
            ),
            cells=dict(
                values=[df[col] for col in df.columns],
                fill_color='lavender',
                align='left',
                font=dict(size=10),
                height=30
            )
        ))
        
        fig.update_layout(
            title="Transcoder Token Activations for Safety Features",
            height=400,
            width=1200
        )
        
        return fig

    def create_feature_importance_dashboard(self, output_path: Path):
        """Create an interactive HTML dashboard with all visualizations."""
        
        try:
            # Get analysis results
            category_features = self.analyzer.find_category_specific_features()
            contrasting_features = self.analyzer.find_contrasting_features()
            
            # Create output directory
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Create feature heatmap
            try:
                heatmap = self.create_feature_heatmap(category_features)
                heatmap.write_html(output_path / "feature_heatmap.html")
                print("✓ Feature heatmap created")
            except Exception as e:
                print(f"Warning: Could not create feature heatmap: {e}")
            
            # Create transcoder token table
            try:
                transcoder_table = self.create_transcoder_token_table(category_features)
                transcoder_table.write_html(output_path / "transcoder_tokens.html")
                print("✓ Transcoder token table created")
            except Exception as e:
                print(f"Warning: Could not create transcoder token table: {e}")
            
            # Create contrast visualization
            try:
                contrast_data = []
                for category, contrasts in contrasting_features.items():
                    if 'differential' in contrasts:
                        for feat in contrasts['differential'][:5]:
                            contrast_data.append({
                                'category': category,
                                'feature': feat['feature'],
                                'safe_activation': feat['safe_activation'],
                                'unsafe_activation': feat['unsafe_activation'],
                                'difference': feat['difference']
                            })
                
                if contrast_data:
                    contrast_df = pd.DataFrame(contrast_data)
                    contrast_fig = px.scatter(
                        contrast_df,
                        x='safe_activation',
                        y='unsafe_activation',
                        color='category',
                        size=contrast_df['difference'].abs(),
                        hover_data=['feature', 'difference'],
                        title='Contrastive Feature Analysis: Safe vs Unsafe Activations'
                    )
                    contrast_fig.write_html(output_path / "contrast_analysis.html")
                    print("✓ Contrast analysis created")
                else:
                    # Create empty plot
                    empty_fig = go.Figure()
                    empty_fig.update_layout(title="No contrastive features found")
                    empty_fig.write_html(output_path / "contrast_analysis.html")
                    
            except Exception as e:
                print(f"Warning: Could not create contrast analysis: {e}")
            
            # Calculate statistics
            total_features = sum(len(feats) for feats in category_features.values())
            total_prompts = len(self.analyzer.graphs)
            categories = len(category_features)
            
            # Generate findings
            findings = []
            for cat, features in category_features.items():
                if features:
                    top_feat = features[0]
                    findings.append(
                        f"<li><strong>{cat}:</strong> Top feature {top_feat[0]} "
                        f"activates in {top_feat[1]['frequency']*100:.1f}% of prompts "
                        f"with avg activation {top_feat[1]['avg_activation']:.3f}</li>"
                    )
            
            if not findings:
                findings.append("<li>No category-specific features found with current thresholds.</li>")
            
            # Create main dashboard HTML (with properly escaped CSS)
            dashboard_html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Safety Circuit Analysis Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                    iframe {{ width: 100%; height: 600px; border: none; }}
                    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .stat-box {{ text-align: center; padding: 20px; background: #f0f0f0; }}
                    .stat-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
                    .stat-label {{ color: #666; margin-top: 10px; }}
                </style>
            </head>
            <body>
                <h1>Safety Circuit Discovery Dashboard with Transcoder Analysis</h1>
                
                <div class="stats">
                    <div class="stat-box">
                        <div class="stat-value">{total_features}</div>
                        <div class="stat-label">Safety-Relevant Features</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{total_prompts}</div>
                        <div class="stat-label">Analyzed Prompts</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-value">{categories}</div>
                        <div class="stat-label">Safety Categories</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Feature Activation Patterns (Hover for Transcoder Tokens)</h2>
                    <iframe src="feature_heatmap.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Transcoder Token Activations</h2>
                    <iframe src="transcoder_tokens.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Contrastive Analysis</h2>
                    <iframe src="contrast_analysis.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    <ul>
                        {findings}
                    </ul>
                </div>
            </body>
            </html>
            """
            
            # Fill template
            html = dashboard_html.format(
                total_features=total_features,
                total_prompts=total_prompts,
                categories=categories,
                findings='\n'.join(findings)
            )
            
            with open(output_path / "index.html", 'w') as f:
                f.write(html)
            
            print(f"✓ Dashboard created at {output_path / 'index.html'}")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            # Create basic fallback
            self._create_basic_fallback_dashboard(output_path)

    def _create_basic_fallback_dashboard(self, output_path: Path):
        """Create a basic dashboard when full dashboard creation fails."""
        output_path.mkdir(exist_ok=True, parents=True)
        
        basic_html = """
        <!DOCTYPE html>
        <html>
        <head><title>Safety Analysis - Basic Report</title></head>
        <body>
            <h1>Safety Circuit Analysis Report</h1>
            <p>Full dashboard creation failed, but analysis completed successfully.</p>
            <p>Check the following files for detailed results:</p>
            <ul>
                <li>category_features.json</li>
                <li>contrasting_features.json</li>
                <li>*_motifs.json</li>
            </ul>
        </body>
        </html>
        """
        
        with open(output_path / "index.html", 'w') as f:
            f.write(basic_html)
        
        print(f"✓ Basic fallback dashboard created at {output_path / 'index.html'}")