import numpy as np
from sklearn.model_selection import train_test_split

from non_neuralx import (
    DecisionTree,
    HistogramGradientBoosting,
    KNearestNeighbors,
    SpectralGraphPruningClassifier,
    accuracy_score,
    create_synthetic_data,
)


def benchmark() -> dict:
    X, y = create_synthetic_data(n_samples=800, n_features=6, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "DecisionTree": DecisionTree(max_depth=8),
        "KNN": KNearestNeighbors(k=7),
        "SGPC": SpectralGraphPruningClassifier(n_neighbors=10, n_components=6, prune_threshold=0.25),
        "HistogramGB": HistogramGradientBoosting(n_estimators=120, max_depth=5),
    }

    print("=== NonNeuralX Benchmark ===")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        results[name] = score
        print(f"{name:16s} accuracy = {score:.4f}")

    winner = max(results, key=results.get)
    print(f"\nWinner: {winner} ({results[winner]:.4f})")
    return results


if __name__ == "__main__":
    benchmark()
