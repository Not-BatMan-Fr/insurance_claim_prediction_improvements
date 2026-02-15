import unittest
import tempfile
import os
import numpy as np
import pandas as pd
from src.visualizer import MatplotlibVisualizer

class StubInner:
    pass

class StubModel:
    def __init__(self):
        self.model = StubInner()
    def predict(self, X):
        return np.zeros(len(X), dtype=int)

class HappyPathModel:
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    def predict_proba(self, X):
        # Mock probabilities for ROC
        return np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])
    @property
    def feature_importances_(self):
        # Mock importances
        return np.array([0.5, 0.3, 0.2])

class TestMatplotlibVisualizer(unittest.TestCase):
    def test_generate_all_plots_handles_models_without_proba(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MatplotlibVisualizer(output_dir=tmpdir)
            results = pd.DataFrame({'Accuracy': [0.5]}, index=['stub']).T
            models = {'StubModel': StubModel()}
            
            paths = viz.generate_all_plots(
                results=results, models_dict=models,
                X_test=np.zeros((3, 1)), y_test=pd.Series([0, 1, 0]),
                y_train_before=pd.Series([0, 1]), y_train_after=pd.Series([0, 1, 1]),
                feature_names=['f1']
            )

            self.assertIn('performance', paths)
            self.assertNotIn('roc_curves', paths)

    def test_generate_all_plots_happy_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = MatplotlibVisualizer(output_dir=tmpdir)
            results = pd.DataFrame({'Accuracy': [0.8]}, index=['HappyModel']).T
            models = {'HappyModel': HappyPathModel()}
            
            paths = viz.generate_all_plots(
                results=results, models_dict=models,
                X_test=np.zeros((3, 3)), y_test=pd.Series([1, 0, 1]),
                y_train_before=pd.Series([0, 0, 1]), y_train_after=pd.Series([0, 0, 1, 1]),
                feature_names=['f1', 'f2', 'f3']
            )

            # Check that ROC and feature importances were successfully triggered
            self.assertIn('roc_curves', paths)
            self.assertIn('feature_importance_HappyModel', paths)
            
            # Verify the files were actually created on disk
            self.assertTrue(os.path.exists(paths['roc_curves']))
            self.assertTrue(os.path.exists(paths['feature_importance_HappyModel']))