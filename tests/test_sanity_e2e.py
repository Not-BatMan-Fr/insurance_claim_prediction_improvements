import unittest
from unittest.mock import patch
import tempfile
import os
import sys
import importlib

# We need to store the original init to call it safely inside our mock
from src.visualizer import MatplotlibVisualizer
original_viz_init = MatplotlibVisualizer.__init__

class TestEndToEnd(unittest.TestCase):
    def test_full_application_execution(self):
        """
        E2E Sanity Check: Runs the entire main application workflow using 
        the truncated dataset and a temporary visualizations directory.
        """
        # 1. Create a temporary directory for the plots
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # 2. Define a wrapper to force the visualizer to use our tmpdir
            def mocked_viz_init(self, output_dir=None):
                original_viz_init(self, output_dir=tmpdir)

            # 3. Patch the Config and the Visualizer
            with patch('src.config.AppConfig.data_path', 'data/truncated_train_data.csv'), \
                 patch.object(MatplotlibVisualizer, '__init__', mocked_viz_init), \
                 patch('sys.argv', ['main.py']):
                
                # 4. Execute the main script
                # We use importlib.reload in case it was already imported by another test
                if 'src.main' in sys.modules:
                    importlib.reload(sys.modules['src.main'])
                else:
                    import src.main
                    
                # If your main.py is wrapped in a main() function, call it explicitly:
                if hasattr(sys.modules['src.main'], 'main'):
                    sys.modules['src.main'].main()
                    
            # 5. Assertions: Verify the E2E run actually generated the files!
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