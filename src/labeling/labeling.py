import os, time
import requests
import pandas as pd
import logging
from enum import Enum
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(name=__name__)

class LLM(Enum):
    
    GPT4o_MINI = "openai/gpt-4o-mini"
    GEMINI_FLASH = "google/gemini-2.5-flash"
    CLAUDE_HAIKU = "anthropic/claude-3.5-haiku"

class Model:
    def __init__(self, llm: LLM):
        self.llm = llm
        pass
    
    def validate_response(self, response) -> str:

        data = response.json()
        raw_label = data["choices"][0]["message"]["content"].strip().lower()
        if raw_label not in ["positive", "negative", "neutral"]:
            return "ERROR"
        return raw_label
        
    def call(self, text: str):

        openrouter_url = os.getenv("OPENROUTER_URL")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

        url = f"{openrouter_url}/chat/completions"

        payload = {
            "model": self.llm.value,
            "max_tokens": 10,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": """You are a precise text classifier. Your task is to label text snippets
                according to their overall sentiment.

                Labels:
                positive  – The text expresses a favorable, optimistic, supportive, or hopeful tone.
                negative  – The text expresses a critical, hostile, pessimistic, angry, or fearful tone.
                neutral   – The text is factual, balanced, or does not express a clear sentiment.

                Rules:
                • Respond with ONLY one of the three labels: positive, negative, or neutral.
                • Do NOT include any explanation, punctuation, or extra words.
                • If the text is ambiguous, prefer neutral."""},
                {"role": "user", "content": f'Classify the overall sentiment of the following text:\n\n"""\n{text}\n"""'}
            ]
        }

        headers = {

        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type":  "application/json",
        "X-Title":        "Sentiment Ground Truth Annotator",
        }

        for attempt in range(1, 4):
            try:
                response = requests.post(
                    url=url,
                    json=payload,
                    headers=headers,
                    timeout=30
                )
                if response.status_code == 429:
                    wait = 3 * attempt
                    logger.warning(
                        "Rate limited (429). Waiting %ds before retry %d/3 ...",
                        wait, attempt, 
                    )
                    time.sleep(wait)
                    continue

                return self.validate_response(response)
            
            except Exception as e:
                logger.warning(e)

        
class LabelingController:
    def __init__(self):
        self.model = None

    def set_model(self, model: Model):
        
        self.model = model
        
    def run_pipeline(self, df: pd.DataFrame, llm: LLM) -> pd.DataFrame:
        model = Model(llm=llm)
        self.set_model(model=model)
        labels = []
        for idx in range(0, len(df), 5):
            end = min(idx + 5, len(df))
            batch = df.iloc[idx : end]
            batch_labels = [model.call(text=text) for text in batch['processed_text']]

            labels.extend(batch_labels)

            time.sleep(10)
        df['label'] = labels
        return df