from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import re

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import torch
import re
import networkx as nx

from safety_circuit_discovery import SafetyCircuitAnalyzer

class SafetyCircuitVisualizer:
    def __init__(self, analyzer: SafetyCircuitAnalyzer):
        self.analyzer = analyzer
        self.model = analyzer.model

    def create_feature_heatmap(self, category_features: Dict[str, List]) -> go.Figure:
        """Create a heatmap showing feature activation patterns."""
        # Prepare data
        categories = list(category_features.keys())
        all_features = set()
        for features in category_features.values():
            all_features.update([f[0] for f in features[:10]])  # Top 10 per category
        all_features = sorted(list(all_features))
        
        # Create matrix
        matrix = np.zeros((len(categories), len(all_features)))
        
        # Prepare hover text
        hover_text = []
        for i, cat in enumerate(categories):
            cat_features = {f[0]: f[1]['frequency'] * f[1]['avg_activation'] 
                          for f in category_features[cat]}
            row_hover = []
            for j, feat in enumerate(all_features):
                value = cat_features.get(feat, 0)
                matrix[i, j] = value
                
                if value > 0:
                    hover_text_entry = f"<b>{feat}</b><br>Category: {cat}<br>Importance: {value:.3f}"
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
            title="Safety-Relevant Feature Activation Patterns",
            xaxis_title="Features",
            yaxis_title="Safety Categories",
            height=600,
            width=1200
        )
        
        return fig

    def create_graph_visualization(self, graph_key: str, max_nodes: int = 50) -> go.Figure:
        """Create an interactive graph visualization for a specific attribution graph."""
        if graph_key not in self.analyzer.graphs:
            raise ValueError(f"Graph {graph_key} not found")
        
        graph = self.analyzer.graphs[graph_key]
        
        # Get active features and their activations
        active_features = graph.active_features[graph.selected_features]
        activation_values = graph.activation_values
        
        # Limit number of nodes for visualization
        if len(active_features) > max_nodes:
            # Sort by activation value and take top nodes
            sorted_indices = torch.argsort(activation_values, descending=True)[:max_nodes]
            active_features = active_features[sorted_indices]
            activation_values = activation_values[sorted_indices]
        
        # Create networkx graph
        G = nx.DiGraph()
        
        # Add nodes
        node_labels = {}
        node_sizes = []
        node_colors = []
        
        for i, (layer, pos, feature_id) in enumerate(active_features):
            node_id = f"L{layer}_F{feature_id}"
            activation = activation_values[i].item()
            
            G.add_node(node_id)
            node_labels[node_id] = f"{node_id}<br>Act: {activation:.3f}"
            node_sizes.append(max(10, activation * 50))  # Scale node size by activation
            node_colors.append(activation)
        
        # Add edges from adjacency matrix
        adj = graph.adjacency_matrix
        edge_weights = []
        
        for i in range(len(active_features)):
            for j in range(len(active_features)):
                if i != j and adj[i, j] > 0.1:  # Threshold for significant connections
                    node_i = f"L{active_features[i, 0].item()}_F{active_features[i, 2].item()}"
                    node_j = f"L{active_features[j, 0].item()}_F{active_features[j, 2].item()}"
                    
                    weight = adj[i, j].item()
                    G.add_edge(node_i, node_j, weight=weight)
                    edge_weights.append(weight)
        
        # Use spring layout for positioning
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Extract positions
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
        
        # Create edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]['weight']
            
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=weight * 5, color='gray'),
                hoverinfo='none',
                mode='lines',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node_labels[node] for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Activation Value"),
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace])
        
        fig.update_layout(
            title=f"Attribution Graph: {graph_key}<br>Input: {graph.input_string[:100]}...",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            width=800
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
            
            # Create graph visualizations for each graph
            graph_files = {}
            available_graphs = []
            try:
                # Create graphs for all available graphs (not just first 10)
                for graph_key in self.analyzer.graphs.keys():
                    try:
                        fig = self.create_graph_visualization(graph_key)
                        graph_filename = f"graph_{graph_key}.html"
                        fig.write_html(output_path / graph_filename)
                        graph_files[graph_key] = graph_filename
                        available_graphs.append(graph_key)
                        print(f"✓ Created graph visualization: {graph_filename}")
                    except Exception as e:
                        print(f"Warning: Could not create graph for {graph_key}: {e}")
            except Exception as e:
                print(f"Warning: Could not create graph visualizations: {e}")
            
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
            
            # Create main dashboard HTML with embedded graph visualization
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
                    .graph-controls {{ 
                        margin: 20px 0; 
                        padding: 15px; 
                        background: #f8f9fa; 
                        border-radius: 5px; 
                    }}
                    .graph-controls select {{ 
                        padding: 8px 12px; 
                        font-size: 14px; 
                        border: 1px solid #ddd; 
                        border-radius: 4px; 
                        margin-right: 10px; 
                    }}
                    .graph-controls button {{ 
                        padding: 8px 16px; 
                        background: #007bff; 
                        color: white; 
                        border: none; 
                        border-radius: 4px; 
                        cursor: pointer; 
                    }}
                    .graph-controls button:hover {{ background: #0056b3; }}
                    .graph-container {{ 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        padding: 10px; 
                        background: white; 
                    }}
                    .no-graph {{ 
                        text-align: center; 
                        padding: 100px; 
                        color: #666; 
                        font-style: italic; 
                    }}
                    .debug-info {{ 
                        font-size: 12px; 
                        color: #666; 
                        margin-top: 10px; 
                        font-style: italic; 
                    }}
                </style>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
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
                    <p><em>Compare feature activations between harmful and safe content</em></p>
                    <iframe src="contrast_analysis.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Visualize Graphs</h2>
                    <p><em>Explore individual attribution graphs to understand feature connectivity patterns</em></p>
                    
                    <div class="graph-controls">
                        <label for="graph-select">Select a prompt:</label>
                        <select id="graph-select">
                            <option value="">Choose a prompt...</option>
                            {graph_options}
                        </select>
                        <button onclick="loadGraph()">Load Graph</button>
                    </div>
                    
                    <div class="graph-container">
                        <div id="graph-display" class="no-graph">
                            Select a prompt above to view its attribution graph
                        </div>
                    </div>
                    
                    <div class="debug-info">
                        Available graphs: {num_available_graphs} | Total graphs: {total_prompts}
                    </div>
                    
                    <p>Each graph shows:</p>
                    <ul>
                        <li><strong>Nodes:</strong> Individual features (colored by activation strength)</li>
                        <li><strong>Edges:</strong> Connections between features (thickness indicates connection strength)</li>
                        <li><strong>Layout:</strong> Spring-based positioning for optimal visibility</li>
                    </ul>
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    <ul>
                        {findings}
                    </ul>
                </div>
                
                <script>
                    const graphFiles = {graph_files_json};
                    console.log('Available graph files:', graphFiles);
                    
                    function loadGraph() {{
                        const select = document.getElementById('graph-select');
                        const graphKey = select.value;
                        const display = document.getElementById('graph-display');
                        
                        console.log('Selected graph key:', graphKey);
                        console.log('Available files:', Object.keys(graphFiles));
                        
                        if (!graphKey) {{
                            display.innerHTML = '<div class="no-graph">Select a prompt above to view its attribution graph</div>';
                            return;
                        }}
                        
                        if (graphFiles[graphKey]) {{
                            console.log('Loading graph file:', graphFiles[graphKey]);
                            // Load the graph iframe
                            display.innerHTML = `<iframe src="${{graphFiles[graphKey]}}" width="100%" height="600px" frameborder="0"></iframe>`;
                        }} else {{
                            console.log('Graph file not found for key:', graphKey);
                            display.innerHTML = '<div class="no-graph">Graph not available for this prompt</div>';
                        }}
                    }}
                </script>
            </body>
            </html>
            """
            
            # Generate graph options for dropdown (only for available graphs)
            graph_options = []
            for graph_key in sorted(available_graphs):
                graph = self.analyzer.graphs[graph_key]
                # Truncate input string for display
                input_preview = graph.input_string[:80] + "..." if len(graph.input_string) > 80 else graph.input_string
                graph_options.append(f'<option value="{graph_key}">{graph_key}: {input_preview}</option>')
            
            # Fill template
            html = dashboard_html.format(
                total_features=total_features,
                total_prompts=total_prompts,
                categories=categories,
                findings='\n'.join(findings),
                graph_options='\n'.join(graph_options),
                graph_files_json=json.dumps(graph_files),
                num_available_graphs=len(available_graphs)
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