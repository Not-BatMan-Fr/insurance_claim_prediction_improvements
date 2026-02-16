import unittest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from src.model_factory import ModelFactory
from src.models import SklearnModelAdapter

class TestModelFactory(unittest.TestCase):
    def test_create_model_success(self):
        model_adapter = ModelFactory.create_model(
            "logistic_regression", 
            {"C": 0.5}, 
            random_state=42
        )
        
        self.assertIsInstance(model_adapter, SklearnModelAdapter)
        self.assertIsInstance(model_adapter.model, LogisticRegression)
        
        self.assertEqual(model_adapter.model.C, 0.5)
        self.assertEqual(model_adapter.model.random_state, 42)

    def test_create_model_unknown_raises_error(self):
        with self.assertRaises(ValueError):
            ModelFactory.create_model("unknown_model", {})

    def test_create_models_from_config(self):
        mock_config = MagicMock()
        mock_config.models_to_train = ["logistic_regression", "decision_tree"]
        mock_config.logistic_regression = {"max_iter": 100}
        mock_config.decision_tree = {"max_depth": 5}

        models = ModelFactory.create_models_from_config(mock_config, random_state=123)
        
        self.assertIn("Logistic Regression", models)
        self.assertIn("Decision Tree", models)
        
        lr_model = models["Logistic Regression"].model
        self.assertEqual(lr_model.max_iter, 100)

    def test_register_and_list_models(self):
        class DummyModel:
            pass
        
        ModelFactory.register_model("dummy", DummyModel, "Dummy Display Name")
        
        self.assertIn("dummy", ModelFactory.list_available_models())
        self.assertEqual(ModelFactory.get_display_name("dummy"), "Dummy Display Name")