import unittest
import pandas as pd
import tempfile
import os
from src.data_loader import CSVLoader

class TestCSVLoader(unittest.TestCase):
    def test_load_success(self):
        # Setup: Create a real temporary CSV file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.csv")
            df_original = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            df_original.to_csv(file_path, index=False)
            
            # Action
            loader = CSVLoader(file_path)
            df_loaded = loader.load()
            
            # Assert: Dataframe loaded from disk matches original exactly
            pd.testing.assert_frame_equal(df_loaded, df_original)

    def test_load_file_not_found(self):
        # Action & Assert
        loader = CSVLoader('invalid_path_that_does_not_exist.csv')
        with self.assertRaises(FileNotFoundError):
            loader.load()