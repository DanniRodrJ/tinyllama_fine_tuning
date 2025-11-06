from src.pipeline import TravelAssistantPipeline
from config.config import TravelAssistantConfig as Config

if __name__ == '__main__':
    pipeline = TravelAssistantPipeline()

    pipeline.run_or_load(force_train=Config.FORCE_RE_TRAIN)

    instruction_to_test = "I'd like information about my checked baggage allowance, how can I find it?"
    
    model_response = pipeline.generate_response(instruction_to_test)
    
    # Imprimir el resultado final
    print("\n" + "="*50)
    print("✅ CICLO DE INFERENCIA DE CLASE COMPLETADO ✅")
    print(f"Instrucción (Query): {instruction_to_test}")
    print("-"*50)
    print(f"Respuesta Generada: \n{model_response}")
    print("="*50)