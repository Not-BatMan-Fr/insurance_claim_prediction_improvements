import unittest
from unittest.mock import patch
import tempfile
import os
import sys
import importlib

from src.visualizer import MatplotlibVisualizer
original_viz_init = MatplotlibVisualizer.__init__

class TestEndToEnd(unittest.TestCase):
    def test_full_application_execution(self):
        """
        E2E Sanity Check: Runs the entire main application workflow using 
        the truncated dataset and a temporary visualizations directory.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            
            def mocked_viz_init(self, output_dir=None):
                original_viz_init(self, output_dir=tmpdir)

            with patch('src.config.AppConfig.data_path', 'data/truncated_train_data.csv'), \
                 patch.object(MatplotlibVisualizer, '__init__', mocked_viz_init), \
                 patch('sys.argv', ['main.py']):
                
                if 'src.main' in sys.modules:
                    importlib.reload(sys.modules['src.main'])
                else:
                    import src.main
                    
                if hasattr(sys.modules['src.main'], 'main'):
                    sys.modules['src.main'].main()
                    
            expected_files = [
                'performance_comparison.png',
                'confusion_matrices.png',
                'class_distribution.png'
            ]
            
            for file_name in expected_files:
                file_path = os.path.join(tmpdir, file_name)
                self.assertTrue(
                    os.path.exists(file_path), 
                    f"E2E Test Failed: Expected output file '{file_name}' was not created."
                )