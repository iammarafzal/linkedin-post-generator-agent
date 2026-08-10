import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from .schemas import LinkedInEvaluation, LinkedInPostSchema

# Available Gemini Models
gemini_3_5_flash_lite: str = "gemini-3.5-flash-lite"
gemini_3_1_flash_lite: str = "gemini-3.1-flash-lite"
gemini_2_5_flash_lite: str = "gemini-2.5-flash-lite"
gemini_3_6_flash: str = "gemini-3.6-flash"
gemini_3_5_flash: str = "gemini-3.5-flash"

# 1. Generator LLM: High-speed creative post generation
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", gemini_3_5_flash_lite)
generator_llm = ChatGoogleGenerativeAI(model=GENERATOR_MODEL, temperature=0.7)
structured_generator_llm = generator_llm.with_structured_output(
    LinkedInPostSchema
)

# 2. Evaluator LLM: Fast structured rule evaluation
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", gemini_3_5_flash_lite)
evaluator_llm = ChatGoogleGenerativeAI(model=EVALUATOR_MODEL, temperature=0.1)
structured_evaluator_llm = evaluator_llm.with_structured_output(
    LinkedInEvaluation
)

# 3. Optimizer LLM: Instant feedback integration & refactoring
OPTIMIZER_MODEL = os.getenv("OPTIMIZER_MODEL", gemini_3_1_flash_lite)
optimizer_llm = ChatGoogleGenerativeAI(model=OPTIMIZER_MODEL, temperature=0.4)