# visualization/charts.py
"""
Chart generation functions for the dashboard
Uses Plotly for interactive visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import numpy as np


def create_profile_radar(profile: Dict) -> go.Figure:
    """
    Create a radar chart for learner profile strengths/weaknesses
    
    Args:
        profile: Student profile dictionary
        
    Returns:
        Plotly figure
    """
    # Extract interests and skills
    interests = profile.get('interests', [])
    strengths = profile.get('strengths', [])
    weaknesses = profile.get('weaknesses', [])
    
    # Create categories
    all_topics = list(set(interests + strengths + weaknesses))
    if not all_topics:
        all_topics = ['Python', 'Data Science', 'Algorithms', 'OOP']
    
    # Score each topic
    scores = []
    for topic in all_topics:
        if topic in strengths:
            scores.append(8)
        elif topic in interests and topic not in weaknesses:
            scores.append(6)
        elif topic in weaknesses:
            scores.append(3)
        else:
            scores.append(5)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=all_topics,
        fill='toself',
        name='Current Level',
        line_color='#1f77b4'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10]
            )),
        showlegend=True,
        title="Knowledge Profile",
        height=400
    )
    
    return fig


def create_path_network(learning_path: Dict) -> go.Figure:
    """
    Create an interactive network graph of the learning path
    
    Args:
        learning_path: Learning path dictionary
        
    Returns:
        Plotly figure
    """
    path_steps = learning_path.get('path', [])
    
    if not path_steps:
        # Empty path
        fig = go.Figure()
        fig.add_annotation(
            text="No learning path available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    # Create nodes and edges
    x_pos = []
    y_pos = []
    node_text = []
    node_colors = []
    
    for i, step in enumerate(path_steps):
        # Position nodes in a flow layout
        row = i // 3
        col = i % 3
        x_pos.append(col)
        y_pos.append(-row)
        
        node_text.append(f"{step.get('title', 'Resource')}<br>({step.get('type', 'N/A')})")
        
        # Color by completion status
        if step.get('completed', False):
            node_colors.append('green')
        else:
            node_colors.append('lightblue')
    
    # Create figure
    fig = go.Figure()
    
    # Add edges (connections)
    for i in range(len(path_steps) - 1):
        fig.add_trace(go.Scatter(
            x=[x_pos[i], x_pos[i+1]],
            y=[y_pos[i], y_pos[i+1]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add nodes
    fig.add_trace(go.Scatter(
        x=x_pos,
        y=y_pos,
        mode='markers+text',
        marker=dict(
            size=20,
            color=node_colors,
            line=dict(color='darkblue', width=2)
        ),
        text=[str(i+1) for i in range(len(path_steps))],
        textposition="middle center",
        textfont=dict(color='white', size=12),
        hovertext=node_text,
        hoverinfo='text',
        showlegend=False
    ))
    
    fig.update_layout(
        title="Learning Path Graph",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=500,
        plot_bgcolor='white'
    )
    
    return fig


def create_shap_waterfall(xai_data: Dict) -> go.Figure:
    """
    Create a SHAP-like waterfall chart for feature importance
    
    Args:
        xai_data: XAI explanations dictionary
        
    Returns:
        Plotly figure
    """
    feature_importance = xai_data.get('profile_explanation', {}).get('feature_importance', {})
    
    if not feature_importance:
        # Mock data for demonstration
        feature_importance = {
            'quiz_performance': 0.35,
            'interaction_frequency': 0.25,
            'time_spent': 0.20,
            'resource_diversity': 0.15,
            'completion_rate': 0.05
        }
    
    features = list(feature_importance.keys())
    values = list(feature_importance.values())
    
    # Sort by absolute value
    sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
    features, values = zip(*sorted_pairs)
    
    colors = ['green' if v > 0 else 'red' for v in values]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{v:.3f}" for v in values],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Feature Importance (SHAP-like)",
        xaxis_title="Contribution to Level Classification",
        yaxis_title="Feature",
        height=400,
        showlegend=False
    )
    
    return fig


def create_recommendation_bars(recommendations: Dict) -> go.Figure:
    """
    Create a bar chart of recommendation priorities
    
    Args:
        recommendations: Recommendations dictionary
        
    Returns:
        Plotly figure
    """
    recs = recommendations.get('recommendations', [])
    
    if not recs:
        fig = go.Figure()
        fig.add_annotation(
            text="No recommendations available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    titles = [rec.get('title', 'Unknown')[:30] + '...' if len(rec.get('title', '')) > 30 
              else rec.get('title', 'Unknown') for rec in recs]
    scores = [rec.get('priority_score', 0.5) for rec in recs]
    types = [rec.get('type', 'unknown') for rec in recs]
    
    # Color by type
    color_map = {'course': 'blue', 'exercise': 'green', 'quiz': 'orange', 
                 'video': 'purple', 'article': 'teal'}
    colors = [color_map.get(t, 'gray') for t in types]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=titles,
        orientation='h',
        marker=dict(color=colors),
        text=[f"{s:.2f}" for s in scores],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="Top Recommendations by Priority",
        xaxis_title="Priority Score",
        yaxis_title="Resource",
        height=400,
        showlegend=False
    )
    
    return fig
