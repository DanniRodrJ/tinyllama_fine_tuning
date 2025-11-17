from peft import LoraConfig
from trl import SFTConfig

class TravelAssistantConfig:
    
    MODEL_SELECTION = "tinyllama"  # "tinyllama" o "dialogpt"
    
    _MODELS = {
        "dialogpt": {
            "name": "microsoft/DialoGPT-medium",
            "adapter_dir": "./models/dialogpt_travel_adapter",
            "target_modules": ["c_attn", "c_proj"]  # GPT-2 architecture
        },
        "tinyllama": {
            "name": "TinyLlama/TinyLlama-1.1B-Chat-v0.1",
            "adapter_dir": "./models/tinyllama_travel_adapter",
            "target_modules": ["q_proj", "v_proj"]   # Llama architecture
        }
    }
    
    MODEL_NAME = _MODELS[MODEL_SELECTION]["name"]
    ADAPTER_OUTPUT_DIR = _MODELS[MODEL_SELECTION]["adapter_dir"]
    LORA_TARGET_MODULES = _MODELS[MODEL_SELECTION]["target_modules"]
    
    FORCE_RE_TRAIN = False  # If True, ignore checkpoints and train from scratch
    DATASET_NAME = 'bitext/Bitext-travel-llm-chatbot-training-dataset'
    DATASET_SPLIT = "train"
    TRAIN_RECORDS_LIMIT = 1000
    PREPROCESSED_DATA_DIR = "./data/prepocessed_dataset"
    SHOW_LOGGING = False
    LOGGIN_TXT = False
    
    # LoRA Configuration (PEFT)
    LORA_CONFIG = LoraConfig( 
        r=12,
        lora_alpha=8,
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGET_MODULES
    )

    # Training Configuration (SFTConfig)
    SFT_CONFIG = SFTConfig(
        # TrainingArguments parameters:
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
        
        # Specific parameters of the SFT:
        dataset_text_field='conversation', 
        max_seq_length=250,
    )
