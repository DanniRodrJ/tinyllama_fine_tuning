from src.pipeline import TravelAssistantPipeline
from utils.logger import app_logger
from config.config import TravelAssistantConfig as Config

if __name__ == '__main__':
    try:
        pipeline = TravelAssistantPipeline()

        pipeline.run_or_load(force_train=Config.FORCE_RE_TRAIN)

        # Prueba de Humo (Sanity Check)
        instruction_to_test = "I'd like information about my checked baggage allowance, how can I find it?"
        
        model_response = pipeline.generate_response(instruction_to_test)
        
        app_logger.info("-" * 50)
        app_logger.info("✅ CICLO DE INFERENCIA DE CLASE COMPLETADO ✅")
        app_logger.info(f"Instrucción: {instruction_to_test}")
        app_logger.info(f"Respuesta Generada:\n{model_response}")
        app_logger.info("-" * 50)
    except Exception as e:
        app_logger.error(f"Un error crítico ocurrió durante la ejecución del pipeline: {e}")