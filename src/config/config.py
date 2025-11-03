from peft import LoraConfig
from trl import SFTConfig

class TravelAssistantConfig:
    # --- Rutas y Nombres de Modelo ---
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.1"
    DATASET_NAME = 'bitext/Bitext-travel-llm-chatbot-training-dataset'
    DATASET_SPLIT = "train"
    TRAIN_RECORDS_LIMIT = 50
    ADAPTER_OUTPUT_DIR = "./model/tinyllama_travel_adapter"
    
    # --- Configuración de LoRA (PEFT) ---
    LORA_CONFIG = LoraConfig(
        r=12,
        lora_alpha=8,
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        target_modules=['q_proj', 'v_proj']
    )

    # --- Configuración de Entrenamiento (SFTConfig) ---
    SFT_CONFIG = SFTConfig(
        # Parámetros de TrainingArguments:
        learning_rate=2e-3, 
        warmup_ratio=0.03,
        num_train_epochs=3,
        output_dir='./tmp_trainer_logs',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        save_steps=10,
        logging_steps=2,
        lr_scheduler_type='constant',
        report_to='none',
        
        # Parámetros específicos del SFT:
        dataset_text_field='conversation', 
        max_seq_length=250,
    )