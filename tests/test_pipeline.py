import unittest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

from src.config import AppConfig, PipelineConfig
from src.pipeline import PipelineOrchestrator
from src.model_factory import ModelFactory
from src.preprocessor import Preprocessor

class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.raw_data = pd.DataFrame({
            'policy_id': ['ID1', 'ID2', 'ID3', 'ID4'],
            'is_parking_camera': ['Yes', 'No', 'Yes', 'No'],
            'length': ['4000', '4200', '4100', '4300'],
            'is_claim': [1, 0, 1, 0],
            'transmission_type': ['Manual', 'Automatic', 'Manual', 'Manual']
        })
        
        pipeline_config = PipelineConfig(
            scaler_type="standard",
            oversampling_method="none" 
        )
        
        self.config = AppConfig(
            test_size=0.5, 
            random_state=42,
            pipeline=pipeline_config
        )
        
        self.preprocessor = Preprocessor(self.config)
        self.mock_loader = MagicMock()
        self.mock_loader.load.return_value = self.raw_data

    def test_full_orchestration_flow(self):
        """
        Tests the integration of Data Loader (mocked), Preprocessor, 
        Scaler, Model, and Evaluator via the PipelineOrchestrator.
        """
        orchestrator = PipelineOrchestrator(
            self.config, 
            self.mock_loader, 
            self.preprocessor
        )
        
        model = ModelFactory.create_model("logistic_regression", {}, random_state=self.config.random_state)
        orchestrator.add_model("Logistic Regression", model)
        
        results = orchestrator.run()
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertIn("Logistic Regression", results.index)
        
        expected_metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC', 'FNR']
        for metric in expected_metrics:
            self.assertIn(metric, results.columns)
            
        accuracy = results.loc["Logistic Regression", "Accuracy"]
        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)