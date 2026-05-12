import os, time
import requests
import pandas as pd
import logging
from enum import Enum
from collections import Counter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(name=__name__)

class LLM(Enum):
    
    GPT4o_MINI = "openai/gpt-4o-mini"
    GEMINI_FLASH = "google/gemini-3.1-flash-lite"
    GROK = "x-ai/grok-4.3"

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
                logger.warning(f"Error calling model {self.llm.name}: {e}")
                if "choices" in str(e):
                    try:
                        logger.warning(f"Response content: {response.text}")
                    except:
                        pass

        
class LabelingController:
    def __init__(self):
        self.models = [Model(llm=llm) for llm in LLM]

    def run_pipeline(self, df: pd.DataFrame, checkpoint_path: str = None) -> pd.DataFrame:
        
        all_results = []
        agreement_matrix = []

        start_row = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                checkpoint_df = pd.read_csv(checkpoint_path)
                if not checkpoint_df.empty:
                    start_row = len(checkpoint_df)
                    logger.info(f"Checkpoint found. Resuming from row {start_row}")
                    
                    model_names = [m.llm.name for m in self.models]
                    if all(col in checkpoint_df.columns for col in model_names + ['label']):
                        all_results = checkpoint_df[model_names + ['label']].to_dict('records')
                        agreement_matrix = checkpoint_df[model_names].values.tolist()
                    else:
                        logger.warning("Checkpoint columns mismatch. Restarting.")
                        start_row = 0
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch.")
                start_row = 0

        logger.info("Starting multi-model labeling pipeline...")

        for idx in range(start_row, len(df), 5):
            end = min(idx + 5, len(df))
            batch = df.iloc[idx : end]
            
            for _, row in batch.iterrows():
                text = row['processed_text']
                model_labels = {}
                
                # Call all 3 models synchronously
                for model in self.models:
                    label = model.call(text=text)
                    model_labels[model.llm.name] = label
                
                labels_list = list(model_labels.values())
                agreement_matrix.append(labels_list)
                
                # Voting system: Settle on one label
                counts = Counter(labels_list)
                most_common_label, count = counts.most_common(1)[0]
                
                if count >= 2:
                    # Majority reached
                    final_label = most_common_label
                else:
                    # No majority (3-way tie): Settle using GPT4o_MINI as primary or default to first
                    final_label = model_labels.get(LLM.GPT4o_MINI.name, labels_list[0])
                
                model_labels['label'] = final_label
                all_results.append(model_labels)

            if checkpoint_path:
                temp_results_df = pd.DataFrame(all_results)
                progress_df = pd.concat([df.iloc[:len(all_results)].reset_index(drop=True), temp_results_df], axis=1)
                progress_df.to_csv(checkpoint_path, index=False)
                logger.info(f"Checkpoint saved to {checkpoint_path}")

            logger.info(f"Processed batch {idx//5 + 1}/{(len(df)-1)//5 + 1}")
            time.sleep(12) # Delay to respect rate limits

        
        results_df = pd.DataFrame(all_results)
        output_df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
        

        
        return output_df