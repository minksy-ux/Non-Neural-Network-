import matplotlib.pyplot as plt
import numpy as np


def plot_spectral_embedding(model, title: str = "Spectral Embedding") -> None:
    """Plot first two spectral components, colored by class labels."""
    if getattr(model, "embedding", None) is None:
        raise ValueError("Model has no embedding. Fit a spectral model first.")

    embedding = model.embedding
    if embedding.shape[1] < 2:
        raise ValueError("Need at least 2 spectral components to plot embedding.")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=model.y_train, cmap="viridis", s=45, alpha=0.85)
    plt.colorbar(scatter, label="Class")
    plt.title(title)
    plt.xlabel("Spectral Component 1")
    plt.ylabel("Spectral Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_pruning_effect(model) -> None:
    """Show edge count before and after pruning for SGPC models."""
    if getattr(model, "X_train", None) is None or getattr(model, "pruned_graph", None) is None:
        raise ValueError("Model must be fitted before plotting pruning effect.")

    original = model._build_knn_graph(np.asarray(model.X_train))
    original_edges = original.nnz // 2
    pruned_edges = model.pruned_graph.nnz // 2

    plt.figure(figsize=(7, 5))
    bars = plt.bar(["Original Graph", "Pruned Graph"], [original_edges, pruned_edges], color=["#8ecae6", "#1d3557"])
    plt.title("Spectral Graph Pruning Effect")
    plt.ylabel("Number of Edges")
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 5, f"{int(h)}", ha="center")
    plt.tight_layout()
    plt.show()


def plot_graph(model, max_nodes: int = 100) -> None:
    """Plot a 2D view of the pruned graph using spectral coordinates."""
    if getattr(model, "pruned_graph", None) is None or getattr(model, "embedding", None) is None:
        raise ValueError("Model must be fitted before plotting graph.")

    n = min(len(model.X_train), max_nodes)
    adj = model.pruned_graph[:n, :n].toarray()
    if model.embedding.shape[1] >= 2:
        pos = model.embedding[:n, :2]
    else:
        pos = np.random.rand(n, 2)

    plt.figure(figsize=(9, 7))
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] > 0:
                plt.plot(
                    [pos[i, 0], pos[j, 0]],
                    [pos[i, 1], pos[j, 1]],
                    color="gray",
                    alpha=min(0.6, 0.2 + adj[i, j]),
                    linewidth=0.7,
                )

    plt.scatter(pos[:, 0], pos[:, 1], c=model.y_train[:n], cmap="viridis", s=45, edgecolors="k")
    plt.title(f"Pruned Spectral Graph ({n} nodes)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
