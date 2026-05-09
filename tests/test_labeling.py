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
    @patch('src.labeling.labeling.time.sleep')
    def test_run_pipeline(self, mock_sleep, mock_call):
        # 3 texts * 3 models = 9 calls
        # Text 1: Majority "positive" (2 pos, 1 neg)
        # Text 2: Majority "negative" (3 neg)
        # Text 3: Tie (all different), fallback to GPT4o_MINI (pos)
        mock_call.side_effect = [
            "positive", "positive", "negative", # Text 1
            "negative", "negative", "negative", # Text 2
            "positive", "negative", "neutral"   # Text 3
        ]
        
        df = pd.DataFrame({
            'processed_text': ["Text 1", "Text 2", "Text 3"]
        })
        
        result_df = self.controller.run_pipeline(df)
        
        self.assertIn('label', result_df.columns)
        self.assertIn('GPT4o_MINI', result_df.columns)
        self.assertIn('GEMINI_FLASH', result_df.columns)
        self.assertIn('CLAUDE_HAIKU', result_df.columns)
        
        self.assertEqual(result_df['label'].tolist(), ["positive", "negative", "positive"])
        self.assertEqual(len(result_df), 3)
        self.assertTrue('fleiss_kappa' in result_df.attrs)

    def test_calculate_fleiss_kappa(self):
        # Perfect agreement
        labels_perfect = [["positive", "positive", "positive"], ["negative", "negative", "negative"]]
        kappa_perfect = self.controller.calculate_fleiss_kappa(labels_perfect)
        self.assertEqual(kappa_perfect, 1.0)

        # No agreement at all (all different on all items)
        labels_none = [["positive", "negative", "neutral"], ["positive", "negative", "neutral"]]
        kappa_none = self.controller.calculate_fleiss_kappa(labels_none)
        self.assertLess(kappa_none, 0.1) # Should be low or negative

if __name__ == '__main__':
    unittest.main()
