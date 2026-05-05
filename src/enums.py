from enum import Enum

class RepresentationMethod(Enum):

    BOW = "bag_of_words"
    GLOVE = "glove"
    TF_IDF = "tf_idf"
    WORD2VEC = "word2vec"

class LLM(Enum):
    
    GPT4o_MINI = "openai/gpt-4o-mini"
    GEMINI_FLASH = "google/gemini-2.5-flash"
    CLAUDE_HAIKU = "anthropic/claude-3.5-haiku"