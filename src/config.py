from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class FeatureConfig:
    """
    Configuration for feature preprocessing.
    
    Defines which columns receive which type of preprocessing treatment.
    """
    
    # Columns to drop (identifiers that shouldn't be features)
    id_columns: List[str] = field(default_factory=lambda: ["policy_id"])
    
    # Boolean columns: "Yes"/"No" → True/False
    boolean_columns: List[str] = field(default_factory=lambda: [
        "is_parking_camera",
        "is_tpms",
        "is_adjustable_steering",
        "is_esc",
        "is_parking_sensors",
        "is_front_fog_lights",
        "is_rear_window_wiper",
        "is_rear_window_washer",
        "is_rear_window_defogger",
        "is_brake_assist",
        "is_power_door_locks",
        "is_power_steering",
        "is_central_locking",
        "is_driver_seat_height_adjustable",
        "is_day_night_rear_view_mirror",
        "is_ecw",
        "is_speed_alert"
    ])
    
    # Numeric columns: convert to float
    float_columns: List[str] = field(default_factory=lambda: [
        "length",
        "width",
        "height",
        "gross_weight",
        "airbags",
        "population_density"
    ])
    
    # Ordinal columns: ordered categories (column_name -> list of categories in order)
    ordinal_columns: Dict[str, List[str]] = field(default_factory=lambda: {
        "ncap_rating": ["0", "1", "2", "3", "4", "5"]
    })
    
    # Categorical columns: one-hot encode these
    categorical_columns: List[str] = field(default_factory=lambda: [
        "transmission_type",
        "cylinder",
        "gear_box",
        "rear_brakes_type",
        "steering_type",
        "fuel_type",
        "make",
        "segment",
        "model",
        "engine_type",
        "max_torque",
        "max_power",
        "area_cluster",
        "displacement"
    ])

    # target_column: str = "is_claim"

@dataclass
class ModelConfig:
    """
    Configuration for individual model hyperparameters.
    
    Each model type has its own set of parameters that will be passed
    to the sklearn constructor.
    """
    
    # Logistic Regression hyperparameters
    logistic_regression: Dict[str, any] = field(default_factory=lambda: {
        "max_iter": 1000,
        "solver": "lbfgs",
        "class_weight": None,
        "C": 1.0
    })
    
    # Decision Tree hyperparameters
    decision_tree: Dict[str, any] = field(default_factory=lambda: {
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini"
    })
    
    # Random Forest hyperparameters
    random_forest: Dict[str, any] = field(default_factory=lambda: {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "criterion": "gini",
        "max_features": "sqrt"
    })
    
    # Which models to train (can enable/disable models here)
    models_to_train: List[str] = field(default_factory=lambda: [
        "logistic_regression",
        "decision_tree",
        "random_forest"
    ])


@dataclass
class AppConfig:
    """Main application configuration."""
    
    # Data paths
    data_path: str = "data/train_data.csv"
    target_column: str = "is_claim"
    
    # Train/test split
    test_size: float = 0.2
    random_state: int = 11
    
    # Feature preprocessing configuration
    features: FeatureConfig = field(default_factory=FeatureConfig)
    # Model configuration
    models: ModelConfig = field(default_factory=ModelConfig)