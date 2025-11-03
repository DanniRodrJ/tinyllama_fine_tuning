# main_run.py

from src.pipeline import TravelAssistantPipeline
from src.config import TravelAssistantConfig as Config
import os

if __name__ == '__main__':
    pipeline = TravelAssistantPipeline()
    
    # 1. Ejecutar Entrenamiento (Descomentar para entrenar)
    # pipeline.train() 

    # 2. Cargar y Fusionar Modelo para Inferencia
    # Si ya entrenaste, asegúrate de que el directorio del checkpoint exista
    if os.path.exists(Config.ADAPTER_OUTPUT_DIR):
        pipeline.load_for_inference()
    else:
        print("❌ ERROR: No se encontraron checkpoints de LoRA. Descomenta y ejecuta pipeline.train() primero.")
        exit()

    # 3. Prueba Final
    instruction_to_test = "I'd like information about my checked baggage allowance, how can I find it?"
    
    model_response = pipeline.generate_response(instruction_to_test)
    
    # Imprimir el resultado final
    print("\n" + "="*50)
    print("✅ CICLO DE INFERENCIA DE CLASE COMPLETADO ✅")
    print(f"Instrucción (Query): {instruction_to_test}")
    print("-"*50)
    print(f"Respuesta Generada: \n{model_response}")
    print("="*50)