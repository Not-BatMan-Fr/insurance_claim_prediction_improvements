"""
Model Factory for creating ML models from configuration.

This module implements the Factory Pattern to decouple model creation
from model usage. Models are instantiated based on configuration rather
than hard-coded in main.py.
"""

from typing import Dict, Any, Callable
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from .models import SklearnModelAdapter
from .config import ModelConfig


class ModelFactory:
    """
    Factory for creating sklearn models from configuration.
    
    Uses a registry pattern to map model names to their constructors.
    New models can be added to the registry without modifying core logic.
    """
    
    # Registry mapping model names to sklearn classes
    _model_registry: Dict[str, Callable] = {
        "logistic_regression": LogisticRegression,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
    }
    
    # Human-friendly names for display
    _display_names: Dict[str, str] = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    
    @classmethod
    def create_model(cls, model_type: str, params: Dict[str, Any], 
                    random_state: int = None) -> SklearnModelAdapter:
        """
        Create a model instance from type and parameters.
        
        Args:
            model_type: Name of model (e.g., "logistic_regression")
            params: Dictionary of hyperparameters for the model
            random_state: Random seed (injected if model supports it)
            
        Returns:
            SklearnModelAdapter wrapping the configured sklearn model
            
        Raises:
            ValueError: if model_type not found in registry
        """
        if model_type not in cls._model_registry:
            available = " ".join(cls._model_registry.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )
        
        # Get the model class from registry
        model_class = cls._model_registry[model_type]
        
        # Make a copy of params to avoid mutating the config
        model_params = params.copy()
        
        # Inject random_state if the model supports it and it's not already set
        if random_state is not None and 'random_state' not in model_params:
            # Check if the model class accepts random_state parameter
            if 'random_state' in model_class.__init__.__code__.co_varnames:
                model_params['random_state'] = random_state
        
        # Instantiate the sklearn model with configured parameters
        sklearn_model = model_class(**model_params)
        
        # Wrap in adapter and return
        return SklearnModelAdapter(sklearn_model)
    
    @classmethod
    def create_models_from_config(cls, model_config: ModelConfig, 
                                  random_state: int = None) -> Dict[str, SklearnModelAdapter]:
        """
        Create all enabled models from ModelConfig.
        
        Args:
            model_config: Configuration object containing model settings
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary mapping display names to model instances
        """
        models = {}
        
        for model_type in model_config.models_to_train:
            # Get hyperparameters for this model type
            params = getattr(model_config, model_type, {})
            
            # Create the model
            model = cls.create_model(model_type, params, random_state)
            
            # Use human-friendly name as key
            display_name = cls._display_names.get(model_type, model_type)
            models[display_name] = model
        
        return models
    
    @classmethod
    def get_display_name(cls, model_type: str) -> str:
        """
        Get human-readable name for a model type.
        
        Args:
            model_type: Internal model name (e.g., "logistic_regression")
            
        Returns:
            Display name (e.g., "Logistic Regression")
        """
        return cls._display_names.get(model_type, model_type)
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Callable, 
                      display_name: str = None):
        """
        Register a new model type (for extensibility).
        
        This allows users to add custom models without modifying this file.
        
        Args:
            model_type: Internal identifier (e.g., "xgboost")
            model_class: The sklearn-compatible class
            display_name: Human-readable name (defaults to model_type)
            
        Example:
            from xgboost import XGBClassifier
            ModelFactory.register_model("xgboost", XGBClassifier, "XGBoost")
        """
        cls._model_registry[model_type] = model_class
        cls._display_names[model_type] = display_name or model_type
    
    @classmethod
    def list_available_models(cls) -> list:
        """
        Get list of all registered model types.
        
        Returns:
            List of model type strings
        """
        return list(cls._model_registry.keys())