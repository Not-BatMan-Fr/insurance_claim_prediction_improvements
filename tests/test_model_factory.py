import unittest
from unittest.mock import MagicMock
from sklearn.linear_model import LogisticRegression
from src.model_factory import ModelFactory
from src.models import SklearnModelAdapter

class TestModelFactory(unittest.TestCase):
    def test_create_model_success(self):
        # Action: Create a model with specific parameters and a random state
        model_adapter = ModelFactory.create_model(
            "logistic_regression", 
            {"C": 0.5}, 
            random_state=42
        )
        
        # Assert: Check that it returns our custom Adapter wrapping the correct sklearn model
        self.assertIsInstance(model_adapter, SklearnModelAdapter)
        self.assertIsInstance(model_adapter.model, LogisticRegression)
        
        # Assert: Check that the hyperparameters and random state were injected correctly
        self.assertEqual(model_adapter.model.C, 0.5)
        self.assertEqual(model_adapter.model.random_state, 42)

    def test_create_model_unknown_raises_error(self):
        # Assert: Asking for an unregistered model should raise a ValueError
        with self.assertRaises(ValueError):
            ModelFactory.create_model("unknown_model", {})

    def test_create_models_from_config(self):
        # Setup: Create a mock configuration object to simulate src.config.ModelConfig
        mock_config = MagicMock()
        mock_config.models_to_train = ["logistic_regression", "decision_tree"]
        mock_config.logistic_regression = {"max_iter": 100}
        mock_config.decision_tree = {"max_depth": 5}

        # Action: Create models from the mock config
        models = ModelFactory.create_models_from_config(mock_config, random_state=123)
        
        # Assert: Check that the dictionary keys are the human-readable display names
        self.assertIn("Logistic Regression", models)
        self.assertIn("Decision Tree", models)
        
        # Assert: Check that the configurations were applied
        lr_model = models["Logistic Regression"].model
        self.assertEqual(lr_model.max_iter, 100)

    def test_register_and_list_models(self):
        # Setup: Define a dummy model class
        class DummyModel:
            pass
        
        # Action: Register the dummy model
        ModelFactory.register_model("dummy", DummyModel, "Dummy Display Name")
        
        # Assert: Verify the new model is in the available list and fetches the correct name
        self.assertIn("dummy", ModelFactory.list_available_models())
        self.assertEqual(ModelFactory.get_display_name("dummy"), "Dummy Display Name")