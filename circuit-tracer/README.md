# Circuit Tracer: Safety Circuit Discovery

A mechanistic interpretability tool for discovering and analyzing safety-relevant circuits in language models using feature attribution and transcoder analysis.

## Overview

This project implements a comprehensive pipeline for identifying neural circuits responsible for safety behaviors in language models. It combines feature attribution techniques with transcoder analysis to map how specific model features activate in response to safety-critical prompts.

## Dataset

The project includes a manually curated safety benchmark dataset created using Claude 4, containing 164 carefully crafted prompts across multiple safety categories:

### Safety Categories
- **Harmful Content**: 25 prompts covering weapons, self-harm, violence, hate speech, illegal activities
- **Manipulation**: 13 prompts covering psychological manipulation, coercion, exploitation
- **Power Seeking**: 11 prompts covering power acquisition, control, dominance
- **Deception**: 25 prompts covering fraud, misrepresentation, disinformation

### Contrasting Examples
Each safety category includes corresponding "safe contrast" examples that request similar information but in ethical, legal contexts:
- **Harmful Content Safe Contrast**: 25 prompts requesting safety information, mental health resources, legal alternatives
- **Manipulation Safe Contrast**: 13 prompts requesting ethical communication, consent education
- **Power Seeking Safe Contrast**: 11 prompts requesting leadership development, collaborative approaches
- **Deception Safe Contrast**: 25 prompts requesting fraud prevention, academic integrity, media literacy

Each prompt includes metadata such as:
- Expected behavior (refuse/comply)
- Severity level (severe/moderate/none)
- Subcategory classification
- Detailed explanation of the safety concern

## Main Objective

The primary goal is to identify and analyze neural circuits that activate when language models process safety-critical content. This involves:

1. **Circuit Discovery**: Finding specific model features that consistently activate for safety-relevant prompts
2. **Feature Analysis**: Understanding what these features represent using transcoder token analysis
3. **Contrastive Analysis**: Comparing activation patterns between harmful and safe content
4. **Pattern Mining**: Identifying common connectivity patterns in safety circuits

## Analysis Pipeline

### 1. Model Loading and Setup
- Loads a target language model (default: Google Gemma-2-2B)
- Initializes transcoder for feature interpretation
- Sets up attribution parameters

### 2. Attribution Collection
- Runs feature attribution on all benchmark prompts
- Collects activation graphs showing feature-to-output connections
- Saves computational graphs for each prompt

### 3. Feature Analysis
- Identifies category-specific features with high activation frequency
- Calculates feature importance scores (frequency × average activation)
- Filters features by minimum frequency and activation thresholds

### 4. Contrastive Analysis
- Compares feature activations between harmful and safe content
- Identifies features unique to harmful content
- Finds features with differential activation patterns

### 5. Pattern Mining
- Analyzes common connectivity patterns in safety circuits
- Identifies motifs that appear across multiple examples
- Calculates support scores for pattern significance

### 6. Visualization
- Creates interactive dashboards with multiple visualizations
- Generates feature heatmaps with transcoder token information
- Produces contrastive analysis plots

## Visualizations

The analysis generates several interactive HTML visualizations:

### Feature Heatmap
Shows feature activation patterns across safety categories with transcoder token information on hover. Features are ranked by importance score (frequency × average activation).

### Transcoder Token Table
Detailed table showing the top semantic tokens associated with each safety-relevant feature, including activation scores and frequency statistics.

### Contrast Analysis
Scatter plots comparing feature activations between harmful and safe content, highlighting features that differentiate between the two types.

### Dashboard
Comprehensive HTML dashboard combining all visualizations with key findings and statistics.

## Sample Visualizations

### Interactive Dashboard
The analysis produces a comprehensive HTML dashboard that includes:

```html
<!DOCTYPE html>
<html>
<head>
    <title>Safety Circuit Analysis Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
        iframe { width: 100%; height: 600px; border: none; }
        .stats { display: flex; justify-content: space-around; margin: 20px 0; }
        .stat-box { text-align: center; padding: 20px; background: #f0f0f0; }
        .stat-value { font-size: 36px; font-weight: bold; color: #2196F3; }
        .stat-label { color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <h1>Safety Circuit Discovery Dashboard with Transcoder Analysis</h1>
    
    <div class="stats">
        <div class="stat-box">
            <div class="stat-value">120</div>
            <div class="stat-label">Safety-Relevant Features</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">164</div>
            <div class="stat-label">Analyzed Prompts</div>
        </div>
        <div class="stat-box">
            <div class="stat-value">6</div>
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
            <li><strong>harmful_content:</strong> Top feature L0_F10219 activates in 65.9% of prompts with avg activation 25.730</li>
            <li><strong>manipulation:</strong> Top feature L0_F13486 activates in 65.5% of prompts with avg activation 23.654</li>
            <li><strong>power_seeking:</strong> Top feature L0_F1903 activates in 100.0% of prompts with avg activation 15.435</li>
            <li><strong>harmful_content_safe_contrast:</strong> Top feature L0_F1903 activates in 100.0% of prompts with avg activation 17.000</li>
            <li><strong>manipulation_safe_contrast:</strong> Top feature L0_F1903 activates in 100.0% of prompts with avg activation 17.772</li>
            <li><strong>power_seeking_safe_contrast:</strong> Top feature L0_F1903 activates in 100.0% of prompts with avg activation 16.202</li>
        </ul>
    </div>
</body>
</html>
```

### Feature Heatmap
The feature heatmap visualization shows activation patterns across safety categories with interactive hover information:

```html
<plotly>
{
  "data": [{
    "type": "heatmap",
    "z": [[0.659, 0.655, 1.0, 1.0, 1.0, 1.0]],
    "x": ["L0_F10219", "L0_F13486", "L0_F1903", "L0_F1903", "L0_F1903", "L0_F1903"],
    "y": ["harmful_content", "manipulation", "power_seeking", "harmful_content_safe_contrast", "manipulation_safe_contrast", "power_seeking_safe_contrast"],
    "colorscale": "RdYlBu_r",
    "text": [[0.659, 0.655, 1.0, 1.0, 1.0, 1.0]],
    "texttemplate": "%{text}",
    "textfont": {"size": 10},
    "colorbar": {"title": "Importance Score"},
    "hovertemplate": "<b>%{x}</b><br>Category: %{y}<br>Importance: %{z:.3f}<br>Top tokens: 'harmful': 0.85, 'dangerous': 0.72, 'illegal': 0.68<extra></extra>"
  }],
  "layout": {
    "title": "Safety-Relevant Feature Activation Patterns with Transcoder Token Data",
    "xaxis_title": "Features",
    "yaxis_title": "Safety Categories",
    "height": 600,
    "width": 1200
  }
}
</plotly>
```

## Key Findings

Analysis of the Gemma-2-2B model reveals several important patterns:

- **Harmful Content**: Feature L0_F10219 activates in 65.9% of prompts with average activation of 25.730
- **Manipulation**: Feature L0_F13486 activates in 65.5% of prompts with average activation of 23.654  
- **Power Seeking**: Feature L0_F1903 activates in 100% of prompts with average activation of 15.435
- **Safe Contrasts**: Feature L0_F1903 shows consistent activation across all safe contrast categories

## Usage

```bash
python src/main.py --model google/gemma-2-2b --transcoder gemma \
    --output-dir ./safety_analysis_results \
    --max-feature-nodes 8192 \
    --min-frequency 0.25 \
    --min-activation 0.1 \
    --mine-patterns \
    --device cuda \
    --categories harmful_content manipulation power_seeking \
    harmful_content_safe_contrast manipulation_safe_contrast power_seeking_safe_contrast
```

### Parameters
- `--model`: Target language model to analyze
- `--transcoder`: Transcoder set for feature interpretation
- `--output-dir`: Directory for saving results
- `--max-feature-nodes`: Maximum number of feature nodes to consider
- `--min-frequency`: Minimum activation frequency threshold
- `--min-activation`: Minimum average activation threshold
- `--mine-patterns`: Enable circuit pattern mining
- `--categories`: Safety categories to analyze

## Output Structure

```
safety_analysis_results/
├── config.json                 # Analysis configuration
├── category_features.json      # Category-specific features
├── contrasting_features.json   # Contrastive analysis results
├── *_motifs.json              # Circuit pattern motifs
├── graphs/                     # Attribution graphs by category
│   ├── harmful_content/
│   ├── manipulation/
│   ├── power_seeking/
│   └── *_safe_contrast/
└── dashboard/                  # Interactive visualizations
    ├── index.html
    ├── feature_heatmap.html
    ├── transcoder_tokens.html
    └── contrast_analysis.html
```

## Dependencies

- PyTorch
- Circuit Tracer (for attribution)
- Plotly (for visualizations)
- NumPy
- Pandas

## Research Applications

This tool enables researchers to:
- Understand how language models process safety-critical content
- Identify potential vulnerabilities in model safety mechanisms
- Develop more robust safety interventions
- Study the relationship between model architecture and safety behavior
- Compare safety circuits across different model sizes and architectures

## Limitations

- Analysis is computationally intensive and requires significant GPU memory
- Results are specific to the analyzed model and may not generalize
- Feature interpretations depend on transcoder quality
- Manual prompt curation may not capture all safety concerns 