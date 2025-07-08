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

from safety_circuit_discovery import SafetyCircuitAnalyzer
from circuit_tracer.utils import create_graph_files
from circuit_tracer import serve

class SafetyCircuitVisualizer:
    def __init__(self, analyzer: SafetyCircuitAnalyzer):
        self.analyzer = analyzer
        self.model = analyzer.model
        self.server = None
        
    def _get_feature_context_examples(self, feature_key: str, category: str, max_examples: int = 3) -> List[Dict]:
        """Get actual context examples where a feature activates, with highlighted tokens."""
        match = re.match(r'L(\d+)_F(\d+)', feature_key)
        if not match:
            return []
        
        layer = int(match.group(1))
        feature_id = int(match.group(2))
        
        examples = []
        
        # Find graphs where this feature activates
        for graph_key, graph in self.analyzer.graphs.items():
            if not graph_key.startswith(category):
                continue
                
            # Get active features for this graph
            active_features = graph.active_features[graph.selected_features]
            activation_values = graph.activation_values
            
            # Find this specific feature
            feat_mask = (
                (active_features[:, 0] == layer) & 
                (active_features[:, 2] == feature_id)
            )
            
            if not feat_mask.any():
                continue
                
            # Get the position and activation value
            feat_indices = torch.where(feat_mask)[0]
            for idx in feat_indices:
                pos = active_features[idx, 1].item()
                activation = activation_values[idx].item()
                
                # Get token context around this position
                context = self._get_token_context(graph, pos, window_size=5)
                
                examples.append({
                    'prompt': graph.input_string,
                    'context': context,
                    'position': pos,
                    'activation': activation,
                    'graph_key': graph_key
                })
                
                if len(examples) >= max_examples:
                    break
                    
            if len(examples) >= max_examples:
                break
                
        return examples

    def _get_token_context(self, graph, position: int, window_size: int = 5) -> str:
        """Get token context around a specific position with highlighting."""
        tokens = graph.input_tokens.tolist()
        decoded_tokens = [self.model.tokenizer.decode([t]) for t in tokens]
        
        # Calculate context window
        start_pos = max(0, position - window_size)
        end_pos = min(len(decoded_tokens), position + window_size + 1)
        
        # Build context with highlighting
        context_parts = []
        for i in range(start_pos, end_pos):
            if i == position:
                # Highlight the activating token
                context_parts.append(f"<span style='background-color: #ffeb3b; font-weight: bold;'>{decoded_tokens[i]}</span>")
            else:
                context_parts.append(decoded_tokens[i])
                
        return "".join(context_parts)

    def _create_context_hover_text(self, feature_key: str, category: str, importance: float) -> str:
        """Create rich hover text with context examples."""
        # Get context examples
        context_examples = self._get_feature_context_examples(feature_key, category, max_examples=2)
        
        # Build hover text
        hover_parts = [
            f"<b>{feature_key}</b>",
            f"Category: {category}",
            f"Importance: {importance:.3f}"
        ]
        
        if context_examples:
            hover_parts.append("<br><b>Context Examples:</b>")
            for i, example in enumerate(context_examples, 1):
                hover_parts.append(f"<br><b>Example {i}:</b> (pos {example['position']}, activation {example['activation']:.2f})")
                hover_parts.append(f"<br>{example['context']}")
        
        return "<br>".join(hover_parts)

    def create_feature_heatmap(self, category_features: Dict[str, List]) -> go.Figure:
        """Create a heatmap showing feature activation patterns with rich context data."""
        # Prepare data
        categories = list(category_features.keys())
        all_features = set()
        for features in category_features.values():
            all_features.update([f[0] for f in features[:10]])  # Top 10 per category
        all_features = sorted(list(all_features))
        
        # Create matrix
        matrix = np.zeros((len(categories), len(all_features)))
        
        # Prepare hover text with rich context data
        hover_text = []
        for i, cat in enumerate(categories):
            cat_features = {f[0]: f[1]['frequency'] * f[1]['avg_activation'] 
                          for f in category_features[cat]}
            row_hover = []
            for j, feat in enumerate(all_features):
                value = cat_features.get(feat, 0)
                matrix[i, j] = value
                
                if value > 0:
                    hover_text_entry = self._create_context_hover_text(feat, cat, value)
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
            title="Safety-Relevant Feature Activation Patterns with Context Examples",
            xaxis_title="Features",
            yaxis_title="Safety Categories",
            height=600,
            width=1200
        )
        
        return fig

    def create_feature_importance_dashboard(self, output_path: Path, node_threshold: float = 0.8, edge_threshold: float = 0.98):
        """Create an interactive HTML dashboard with all visualizations."""
        
        try:
            # Get analysis results
            category_features = self.analyzer.find_category_specific_features()
            contrasting_features = self.analyzer.find_contrasting_features()
            
            # Create output directory
            output_path.mkdir(exist_ok=True, parents=True)
            
            # Create attribution graph files
            try:
                graph_metadata = self.create_attribution_graph_files(output_path, node_threshold, edge_threshold)
                print("✓ Attribution graph files created")
            except Exception as e:
                print(f"Warning: Could not create attribution graph files: {e}")
                graph_metadata = {}
            
            # Create feature heatmap
            try:
                heatmap = self.create_feature_heatmap(category_features)
                heatmap.write_html(output_path / "feature_heatmap.html")
                print("✓ Enhanced feature heatmap created")
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
                <title>Enhanced Safety Circuit Analysis Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                    iframe {{ width: 100%; height: 600px; border: none; }}
                    .stats {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .stat-box {{ text-align: center; padding: 20px; background: #f0f0f0; }}
                    .stat-value {{ font-size: 36px; font-weight: bold; color: #2196F3; }}
                    .stat-label {{ color: #666; margin-top: 10px; }}
                    .feature {{ background-color: #ffeb3b; font-weight: bold; }}
                    .info-box {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                    .server-info {{ background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; border-left: 4px solid #ffc107; }}
                </style>
            </head>
            <body>
                <h1>Safety Circuit Discovery Dashboard with Context Analysis</h1>
                
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
                    <h2>Feature Activation Patterns with Context Examples</h2>
                    <p><em>Hover over cells to see actual context examples with highlighted activating tokens</em></p>
                    <iframe src="feature_heatmap.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Contrastive Analysis</h2>
                    <p><em>Compare feature activations between harmful and safe content</em></p>
                    <iframe src="contrast_analysis.html"></iframe>
                </div>
                
                <div class="section">
                    <h2>Attribution Graph Visualization</h2>
                    <div class="server-info">
                        <h3>🚀 Interactive Graph Visualization</h3>
                        <p><strong>To view attribution graphs:</strong></p>
                        <ol>
                            <li>Open a terminal in the project directory</li>
                            <li>Run: <code>python src/start_visualization_server.py</code></li>
                            <li>A browser tab will open automatically with the interactive graph viewer</li>
                            <li>Use the dropdown below to select different prompts and explore their attribution graphs</li>
                        </ol>
                        <p><strong>Server URL:</strong> <a href="http://localhost:8032" target="_blank">http://localhost:8032</a></p>
                        <p><strong>Alternative:</strong> You can also run: <code>python -c "from circuit_tracer import serve; serve('safety_analysis_results/dashboard/graph_files', port=8032)"</code></p>
                    </div>
                    
                    <div style="margin-bottom: 20px;">
                        <label for="graphSelect" style="font-weight: bold; margin-right: 10px;">Available Prompts:</label>
                        <select id="graphSelect" style="padding: 8px; font-size: 14px; width: 300px;" onchange="openGraph()">
                            <option value="">Choose a prompt to visualize...</option>
                            {graph_options}
                        </select>
                        <button onclick="openSelectedGraph()" style="padding: 8px 16px; font-size: 14px; background-color: #4CAF50; color: white; border: none; border-radius: 3px; cursor: pointer; margin-left: 10px;">
                            🔍 View Graph
                        </button>
                    </div>
                    
                    <div class="info-box">
                        <h4>How to use the attribution graphs:</h4>
                        <ul>
                            <li><strong>Nodes:</strong> Represent tokens (words/subwords) in the input</li>
                            <li><strong>Edges:</strong> Show how features influence token predictions</li>
                            <li><strong>Colors:</strong> Indicate activation strength and feature importance</li>
                            <li><strong>Hover:</strong> Get detailed information about nodes and edges</li>
                        </ul>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Key Findings</h2>
                    <ul>
                        {findings}
                    </ul>
                </div>
                
                <div class="section">
                    <h2>How to Use This Dashboard</h2>
                    <ol>
                        <li><strong>Feature Heatmap:</strong> Hover over colored cells to see detailed information about each feature, including context examples</li>
                        <li><strong>Context Examples:</strong> Yellow-highlighted tokens show exactly where each feature activates in the original prompts</li>
                        <li><strong>Contrast Analysis:</strong> Identify features that differentiate between harmful and safe content</li>
                        <li><strong>Attribution Graphs:</strong> Start the server and select a prompt to view its detailed attribution graph showing feature interactions</li>
                    </ol>
                </div>
                
                <script>
                    function openSelectedGraph() {{
                        const select = document.getElementById('graphSelect');
                        const selectedValue = select.value;
                        
                        if (!selectedValue) {{
                            alert('Please select a prompt first!');
                            return;
                        }}
                        
                        // Open the specific graph in the server
                        const graphUrl = `http://localhost:8032?graph=${{selectedValue}}_graph`;
                        window.open(graphUrl, '_blank');
                    }}
                    
                    function openGraph() {{
                        // This function is called when the dropdown changes
                        // We'll just update the button text or provide feedback
                        const select = document.getElementById('graphSelect');
                        const selectedValue = select.value;
                        
                        if (selectedValue) {{
                            console.log('Selected graph:', selectedValue);
                        }}
                    }}
                </script>
            </body>
            </html>
            """
            
            # Generate graph options for dropdown
            graph_options = []
            for graph_key, metadata in graph_metadata.items():
                category = metadata.get('category', 'unknown')
                input_preview = metadata.get('input_string', 'No preview available')
                graph_options.append(
                    f'<option value="{graph_key}">{category}: {input_preview}</option>'
                )
            
            # Fill template
            html = dashboard_html.format(
                total_features=total_features,
                total_prompts=total_prompts,
                categories=categories,
                findings='\n'.join(findings),
                graph_options='\n'.join(graph_options)
            )
            
            with open(output_path / "index.html", 'w') as f:
                f.write(html)
            
            print(f"✓ Enhanced dashboard created at {output_path / 'index.html'}")
            print(f"✓ To view attribution graphs, run: python src/start_visualization_server.py")
            
        except Exception as e:
            print(f"Error creating dashboard: {e}")
            # Create basic fallback
            self._create_basic_fallback_dashboard(output_path)

    def _start_visualization_server(self, data_dir: Path):
        """Start the visualization server for attribution graphs."""
        try:
            if self.server is None:
                self.server = serve(data_dir, port=8032)
                print(f"✓ Visualization server started at http://localhost:8032")
                print(f"  Data directory: {data_dir}")
                print(f"  Open your browser to view interactive attribution graphs")
            else:
                print("✓ Visualization server already running")
        except Exception as e:
            print(f"Warning: Could not start visualization server: {e}")
            print("  You can manually start the server using:")
            print(f"  python -c \"from circuit_tracer import serve; serve('{data_dir}', port=8032)\"")

    def stop_server(self):
        """Stop the visualization server."""
        if self.server:
            self.server.stop()
            self.server = None
            print("✓ Visualization server stopped")

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
            <p>To view attribution graphs, run:</p>
            <code>python -c "from circuit_tracer import serve; serve('graph_files', port=8032)"</code>
        </body>
        </html>
        """
        
        with open(output_path / "index.html", 'w') as f:
            f.write(basic_html)
        
        print(f"✓ Basic fallback dashboard created at {output_path / 'index.html'}")

    def create_attribution_graph_files(self, output_path: Path, node_threshold: float = 0.8, edge_threshold: float = 0.98):
        """Create graph files for attribution visualization for each prompt."""
        graph_files_dir = output_path / "graph_files"
        graph_files_dir.mkdir(exist_ok=True, parents=True)
        
        print("Creating attribution graph files...")
        
        # Create a mapping of graph keys to their metadata
        graph_metadata = {}
        
        for graph_key, graph in self.analyzer.graphs.items():
            try:
                # Create a slug for the graph
                slug = f"{graph_key}_graph"
                
                # Create graph files
                create_graph_files(
                    graph_or_path=graph,
                    slug=slug,
                    output_path=graph_files_dir,
                    node_threshold=node_threshold,
                    edge_threshold=edge_threshold
                )
                
                # Store metadata for the dropdown
                graph_metadata[graph_key] = {
                    'slug': slug,
                    'input_string': graph.input_string[:100] + "..." if len(graph.input_string) > 100 else graph.input_string,
                    'category': graph_key.split('_')[0] if '_' in graph_key else 'unknown'
                }
                
                print(f"✓ Created graph files for {graph_key}")
                
            except Exception as e:
                print(f"Warning: Could not create graph files for {graph_key}: {e}")
                continue
        
        # Save metadata for the dashboard
        with open(graph_files_dir / "graph_metadata.json", 'w') as f:
            json.dump(graph_metadata, f, indent=2)
        
        print(f"✓ Created {len(graph_metadata)} graph files in {graph_files_dir}")
        return graph_metadata