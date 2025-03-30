import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple

def plot_similarity_distribution(
    scores: List[float], 
    labels: List[str] = None, 
    title: str = "Similarity Distribution"
) -> None:
    """
    Plot distribution of similarity scores.
    
    Args:
        scores: List of similarity scores
        labels: Optional labels for each score
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    if labels:
        plt.bar(labels, scores, color="skyblue")
        plt.xticks(rotation=45, ha="right")
    else:
        plt.bar(range(len(scores)), scores, color="skyblue")
    
    plt.xlabel("Documents")
    plt.ylabel("Similarity Score")
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_retrieval_comparison(
    question: str,
    results: List[Dict[str, Any]],
    metric: str = "count",
    title: str = "Retrieval Parameter Comparison"
) -> None:
    """
    Plot comparison of different retrieval parameters.
    
    Args:
        question: The query question
        results: List of result dictionaries with parameters and metrics
        metric: Metric to compare ('count', 'time', 'relevance')
        title: Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Extract parameters and metrics
    params = [f"k={r.get('k', 'N/A')}, {r.get('search_type', 'similarity')}" for r in results]
    values = [r.get(metric, 0) for r in results]
    
    plt.bar(params, values, color="lightgreen")
    plt.xlabel("Retrieval Parameters")
    plt.ylabel(metric.capitalize())
    plt.title(f"{title} for: '{question}'")
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def create_heatmap(
    similarity_matrix: List[List[float]],
    x_labels: List[str],
    y_labels: List[str],
    title: str = "Similarity Heatmap"
) -> None:
    """
    Create a heatmap visualization of similarity between documents.
    
    Args:
        similarity_matrix: 2D matrix of similarity values
        x_labels: Labels for x-axis
        y_labels: Labels for y-axis
        title: Plot title
    """
    plt.figure(figsize=(12, 10))
    
    plt.imshow(similarity_matrix, cmap="viridis")
    plt.colorbar(label="Similarity")
    
    # Add labels
    plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
    plt.yticks(np.arange(len(y_labels)), y_labels)
    
    plt.title(title)
    plt.tight_layout()
    plt.show()
