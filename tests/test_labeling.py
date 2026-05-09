import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

from src.labeling.labeling import Model, LabelingController, LLM

class TestLabeling(unittest.TestCase):

    def setUp(self):
        self.llm = LLM.GPT4o_MINI
        self.model = Model(llm=self.llm)
        self.controller = LabelingController()

    def test_validate_response_positive(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "positive"}}]
        }
        result = self.model.validate_response(mock_response)
        self.assertEqual(result, "positive")

    def test_validate_response_error(self):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "unknown"}}]
        }
        result = self.model.validate_response(mock_response)
        self.assertEqual(result, "ERROR")

    @patch('src.labeling.labeling.requests.post')
    def test_model_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "neutral"}}]
        }
        mock_post.return_value = mock_response

        result = self.model.call("This is a test text")
        self.assertEqual(result, "neutral")
        mock_post.assert_called_once()

    @patch('src.labeling.labeling.Model.call')
    @patch('src.labeling.labeling.time.sleep') # Mock sleep to speed up tests
    def test_run_pipeline(self, mock_sleep, mock_call):
        mock_call.side_effect = ["positive", "negative", "neutral"]
        
        df = pd.DataFrame({
            'processed_text': ["I love this", "I hate this", "This is a book"]
        })
        
        result_df = self.controller.run_pipeline(df, self.llm)
        
        self.assertIn('label', result_df.columns)
        self.assertEqual(result_df['label'].tolist(), ["positive", "negative", "neutral"])
        self.assertEqual(len(result_df), 3)

if __name__ == '__main__':
    unittest.main()
