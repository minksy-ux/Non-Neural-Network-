import unittest

import numpy as np

from non_neural_toolkit import (
    DecisionTree,
    HistogramGradientBoosting,
    KNearestNeighbors,
    LinearRegression,
    LinearSVM,
    SpectralGraphPruningClassifier,
    accuracy_score,
    benchmark,
    create_regression_data,
    create_synthetic_data,
    mean_squared_error,
    train_test_split,
)


class TestNonNeuralToolkit(unittest.TestCase):
    def setUp(self):
        self.X_bin, self.y_bin = create_synthetic_data(
            n_samples=240,
            n_features=5,
            n_classes=2,
            random_state=123,
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_bin,
            self.y_bin,
            test_size=0.25,
            random_state=123,
        )

    def test_decision_tree_smoke(self):
        model = DecisionTree(max_depth=6).fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])
        self.assertGreaterEqual(model.score(self.X_test, self.y_test), 0.5)

    def test_knn_smoke(self):
        model = KNearestNeighbors(k=5).fit(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        self.assertEqual(pred.shape[0], self.X_test.shape[0])
        self.assertGreaterEqual(model.score(self.X_test, self.y_test), 0.5)

    def test_linear_svm_binary(self):
        model = LinearSVM(n_iterations=300, learning_rate=0.01).fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_histogram_gradient_boosting_binary(self):
        model = HistogramGradientBoosting(n_estimators=40, max_depth=3).fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        self.assertGreaterEqual(score, 0.5)

    def test_spectral_graph_pruning_classifier(self):
        model = SpectralGraphPruningClassifier(
            n_neighbors=8,
            n_components=4,
            prune_threshold=0.25,
            n_iterations=15,
            random_state=7,
        ).fit(self.X_train, self.y_train)
        probs = model.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (self.X_test.shape[0], 2))
        np.testing.assert_allclose(
            probs.sum(axis=1),
            np.ones(self.X_test.shape[0]),
            rtol=1e-6,
            atol=1e-6,
        )

    def test_linear_regression_and_metrics(self):
        X, y = create_regression_data(n_samples=260, n_features=3, random_state=222)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=222)
        model = LinearRegression().fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        self.assertGreaterEqual(mse, 0.0)
        self.assertLess(mse, 2.0)

        r2 = model.score(X_test, y_test)
        self.assertGreater(r2, 0.7)

    def test_accuracy_score(self):
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        self.assertAlmostEqual(accuracy_score(y_true, y_pred), 0.75)

    def test_benchmark_contract(self):
        results = benchmark()
        expected = {
            "DecisionTree",
            "KNearestNeighbors",
            "LinearSVM",
            "SpectralGraphPruningClassifier",
            "HistogramGradientBoosting",
        }
        self.assertTrue(expected.issubset(results.keys()))
        for value in results.values():
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


if __name__ == "__main__":
    unittest.main()
