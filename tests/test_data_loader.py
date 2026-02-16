import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_loader import CSVLoader

class TestCSVLoader(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_load_success(self, mock_read_csv):
        # dummy csv setup
        mock_df = pd.DataFrame({'col1': [1, 2]})
        mock_read_csv.return_value = mock_df
        loader = CSVLoader('dummy_path.csv')

        # Action dummy csv
        df = loader.load()
        
        # Assert dummy csv
        pd.testing.assert_frame_equal(df, mock_df)
        mock_read_csv.assert_called_once_with('dummy_path.csv')

    def test_load_success(self):
        # Setup: a real temporary CSV file
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_data.csv")
            df_original = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
            df_original.to_csv(file_path, index=False)
            
            # Action
            loader = CSVLoader(file_path)
            df_loaded = loader.load()
            
            # Assert: Dataframe loaded from disk should match original exactly
            pd.testing.assert_frame_equal(df_loaded, df_original)

    @patch('pandas.read_csv')
    def test_load_file_not_found(self, mock_read_csv):
        # Setup
        mock_read_csv.side_effect = FileNotFoundError
        loader = CSVLoader('invalid_path.csv')

        # Assert
        with self.assertRaises(FileNotFoundError):
            loader.load()
