from pathlib import Path
import json
from typing import Dict, List

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from circuit_tracer.graph import Graph

from safety_circuit_discovery import SafetyCircuitAnalyzer
from safety_interventions import InterventionResult

class SafetyCircuitVisualizer:
    def __init__(self, analyzer: SafetyCircuitAnalyzer):
        self.analyzer = analyzer
        
    def create_feature_heatmap(self, category_features: Dict[str, List]) -> go.Figure:
        """Create a heatmap showing feature activation patterns across categories."""
        
        # Prepare data
        categories = list(category_features.keys())
        all_features = set()
        
        for features in category_features.values():
            all_features.update([f[0] for f in features[:10]])  # Top 10 per category
        
        all_features = sorted(list(all_features))
        
        # Create matrix
        matrix = np.zeros((len(categories), len(all_features)))
        
        for i, cat in enumerate(categories):
            cat_features = {f[0]: f[1]['frequency'] * f[1]['avg_activation'] 
                          for f in category_features[cat]}
            for j, feat in enumerate(all_features):
                matrix[i, j] = cat_features.get(feat, 0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=all_features,
            y=categories,
            colorscale='RdYlBu_r',
            text=np.round(matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Importance Score")
        ))
        
        fig.update_layout(
            title="Safety-Relevant Feature Activation Patterns",
            xaxis_title="Features",
            yaxis_title="Safety Categories",
            height=600,
            width=1200
        )
        
        return fig
    
    def create_intervention_results_plot(self, results: List[InterventionResult]) -> go.Figure:
        """Visualize intervention effectiveness."""
        
        df = pd.DataFrame([
            {
                'prompt_idx': i,
                'original_prob': r.original_prob,
                'intervened_prob': r.intervened_prob,
                'safety_improved': r.safety_improved,
                'capability_preserved': r.capability_preserved
            }
            for i, r in enumerate(results)
        ])
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Probability Changes', 'Safety Improvement Rate',
                          'Capability Preservation', 'Success Rate by Category'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Probability changes
        fig.add_trace(
            go.Scatter(x=df['prompt_idx'], y=df['original_prob'],
                      mode='markers', name='Original',
                      marker=dict(color='blue', size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df['prompt_idx'], y=df['intervened_prob'],
                      mode='markers', name='Intervened',
                      marker=dict(color='red', size=8)),
            row=1, col=1
        )
        # Safety improvement rate
        safety_rate = df['safety_improved'].mean() * 100
        fig.add_trace(
            go.Bar(x=['Safety Improved'], y=[safety_rate],
                    text=[f"{safety_rate:.1f}%"],
                    textposition='auto',
                    marker_color='green'),
            row=1, col=2
        )
        
        # Capability preservation
        capability_rate = df['capability_preserved'].mean() * 100
        fig.add_trace(
            go.Bar(x=['Capability Preserved'], y=[capability_rate],
                    text=[f"{capability_rate:.1f}%"],
                    textposition='auto',
                    marker_color='blue'),
            row=2, col=1
        )
        
        # Success rate (both safety and capability)
        success_rate = (df['safety_improved'] & df['capability_preserved']).mean() * 100
        fig.add_trace(
            go.Bar(x=['Overall Success'], y=[success_rate],
                    text=[f"{success_rate:.1f}%"],
                    textposition='auto',
                    marker_color='purple'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=True,
                            title_text="Intervention Effectiveness Analysis")
        
        return fig
    
    def create_circuit_comparison_view(self, 
                                        safe_graph: Graph,
                                        unsafe_graph: Graph,
                                        node_threshold: float = 0.8) -> go.Figure:
        """Create side-by-side comparison of safe vs unsafe circuits."""
        
        from circuit_tracer.graph import prune_graph
        
        # Prune both graphs
        safe_nodes, safe_edges, _ = prune_graph(safe_graph, node_threshold)
        unsafe_nodes, unsafe_edges, _ = prune_graph(unsafe_graph, node_threshold)
        
        # Create network graphs
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Safe Response Circuit', 'Unsafe Response Circuit'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # Helper function to create network visualization
        def add_network(graph, nodes, edges, row, col, title):
            # Extract node positions using force-directed layout
            import networkx as nx
            
            G = nx.DiGraph()
            edge_list = []
            
            adj = graph.adjacency_matrix.cpu().numpy()
            for i in range(len(nodes)):
                if nodes[i]:
                    for j in range(len(nodes)):
                        if nodes[j] and edges[i, j]:
                            edge_list.append((i, j, adj[i, j]))
            
            G.add_weighted_edges_from(edge_list)
            
            if len(G.nodes()) > 0:
                pos = nx.spring_layout(G, k=2, iterations=50)
                
                # Add nodes
                node_x = [pos[node][0] for node in G.nodes()]
                node_y = [pos[node][1] for node in G.nodes()]
                
                fig.add_trace(
                    go.Scatter(x=node_x, y=node_y,
                                mode='markers',
                                marker=dict(size=10, color='lightblue'),
                                text=[f"Node {i}" for i in G.nodes()],
                                hoverinfo='text',
                                showlegend=False),
                    row=row, col=col
                )
                
                # Add edges
                for edge in G.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    fig.add_trace(
                        go.Scatter(x=[x0, x1], y=[y0, y1],
                                    mode='lines',
                                    line=dict(width=1, color='gray'),
                                    showlegend=False),
                        row=row, col=col
                    )
        
        add_network(safe_graph, safe_nodes, safe_edges, 1, 1, "Safe")
        add_network(unsafe_graph, unsafe_nodes, unsafe_edges, 1, 2, "Unsafe")
        
        fig.update_layout(height=600, title_text="Circuit Structure Comparison")
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
                    import pandas as pd
                    import plotly.express as px
                    
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
                    import plotly.graph_objects as go
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
                <h1>Safety Circuit Discovery Dashboard</h1>
                
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
                    <h2>Feature Activation Patterns</h2>
                    <iframe src="feature_heatmap.html"></iframe>
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