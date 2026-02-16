import unittest
from unittest.mock import patch
import numpy as np
from src.sampling_strategy import (
    SamplingStrategyFactory, 
    NoSamplingStrategy, 
    RandomOversamplingStrategy
)

class TestSamplingStrategyFactory(unittest.TestCase):
    def test_create_strategy_none(self):
        strategy = SamplingStrategyFactory.create_strategy("none")
        
        self.assertIsInstance(strategy, NoSamplingStrategy)
        
    def test_create_strategy_random(self):
        strategy = SamplingStrategyFactory.create_strategy(
            "random", 
            sampling_strategy="all", 
            random_state=42
        )
        
        self.assertIsInstance(strategy, RandomOversamplingStrategy)
        self.assertEqual(strategy.sampling_strategy, "all")
        self.assertEqual(strategy.random_state, 42)

    def test_create_strategy_unknown_raises_error(self):
        with self.assertRaises(ValueError):
            SamplingStrategyFactory.create_strategy("unknown_strategy")

    def test_no_sampling_strategy_resample(self):
        strategy = NoSamplingStrategy()
        X = np.array([[1], [2]])
        y = np.array([0, 1])
        
        X_res, y_res = strategy.resample(X, y)
        
        np.testing.assert_array_equal(X_res, X)
        np.testing.assert_array_equal(y_res, y)

    @patch('src.sampling_strategy.RandomOverSampler')
    def test_random_oversampling_resample(self, mock_ros_class):
        mock_instance = mock_ros_class.return_value
        mock_instance.fit_resample.return_value = ("mock_X", "mock_y")
        
        strategy = RandomOversamplingStrategy(sampling_strategy="minority", random_state=42)
        X, y = "raw_X", "raw_y"
        
        X_res, y_res = strategy.resample(X, y)
        
        mock_ros_class.assert_called_once_with(sampling_strategy="minority", random_state=42)
        mock_instance.fit_resample.assert_called_once_with(X, y)
        self.assertEqual(X_res, "mock_X")
        self.assertEqual(y_res, "mock_y")