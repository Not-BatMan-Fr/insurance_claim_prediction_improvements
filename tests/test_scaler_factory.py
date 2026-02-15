import unittest
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.scaler_factory import ScalerFactory, NoScaler

class TestScalerFactory(unittest.TestCase):
    def test_create_scaler_standard(self):
        # Action
        scaler = ScalerFactory.create_scaler("standard")
        
        # Assert
        self.assertIsInstance(scaler, StandardScaler)
            
    def test_create_scaler_unknown_raises_error(self):
        # Assert
        with self.assertRaises(ValueError):
            ScalerFactory.create_scaler("unknown_scaler")
                
    def test_no_scaler_behavior(self):
        # Setup: Ensure NoScaler behaves like a standard sklearn transformer without changing data
        scaler = NoScaler()
        X = np.array([[1, 2], [3, 4]])
        
        # Assert: fit should return itself
        self.assertEqual(scaler.fit(X), scaler)
        
        # Assert: transform and fit_transform should return X completely unchanged
        np.testing.assert_array_equal(scaler.transform(X), X)
        np.testing.assert_array_equal(scaler.fit_transform(X), X)

    def test_register_and_list_scalers(self):
        # Setup
        class CustomScaler:
            pass
        
        # Action
        ScalerFactory.register_scaler("custom", CustomScaler, "Custom Description")
        
        # Assert
        self.assertIn("custom", ScalerFactory.list_available_scalers())
        self.assertEqual(ScalerFactory.get_description("custom"), "Custom Description")