from src.pipeline import TravelAssistantPipeline
from utils.logger import app_logger
from config.config import TravelAssistantConfig as Config

if __name__ == '__main__':
    try:
        pipeline = TravelAssistantPipeline()

        pipeline.run_or_load(force_train=Config.FORCE_RE_TRAIN)

        # Smoke Test (Sanity Check)
        instruction_to_test = "I'd like information about my checked baggage allowance, how can I find it?"
        
        model_response = pipeline.generate_response(instruction_to_test)
        
        app_logger.info("-" * 50)
        app_logger.info("✅ CCLASS INFERENCE CYCLE COMPLETED ✅")
        app_logger.info(f"Instruction: {instruction_to_test}")
        app_logger.info(f"Generated Response:\n{model_response}")
        app_logger.info("-" * 50)
    except Exception as e:
        app_logger.error(f"A critical error occurred during pipeline execution: {e}")