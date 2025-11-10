import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import PeftModel
from config.config import TravelAssistantConfig as Config
from src.data_preparation import load_and_prepare_dataset
from src.utils.logger import app_logger

class TravelAssistantPipeline:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.merged_model = None
        self.train_dataset = load_and_prepare_dataset()
        
        if not os.path.exists(Config.PREPROCESSED_DATA_DIR):
            self.train_dataset.save_to_disk(Config.PREPROCESSED_DATA_DIR)
    
    def _load_base_components(self):
        """Loads the base model and tokenizer"""
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)

    def train(self):
        """Executes fine-tuning and saves the adapters"""
        
        self._load_base_components()
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=Config.SFT_CONFIG, 
            peft_config=Config.LORA_CONFIG,
        )

        app_logger.info("Starting fine-tuning...")
        trainer.train()
        app_logger.info("Fine-tuning completed")

        # Save LoRA adapters and tokenizer
        trainer.model.save_pretrained(Config.ADAPTER_OUTPUT_DIR)
        self.tokenizer.save_pretrained(Config.ADAPTER_OUTPUT_DIR)
        app_logger.info(f"LoRA adapters saved to: {Config.ADAPTER_OUTPUT_DIR}")

    def load_for_inference(self):
        """Loads the base model and merges the adapters for inference"""
        
        # Define the offload directory (if you do not have a GPU)
        OFFLOAD_FOLDER = os.path.join(os.getcwd(), "offload_temp")
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
        
        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.ADAPTER_OUTPUT_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the base model
        base_model_for_inference = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16, 
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER  # Comment if you have a GPU
        )

        # Load and merge adapters
        model_with_peft = PeftModel.from_pretrained(
            base_model_for_inference, 
            Config.ADAPTER_OUTPUT_DIR
        )

        self.merged_model = model_with_peft.merge_and_unload()
        self.merged_model.eval() 
        app_logger.info("Base model and LoRA adapters merged")
        
        
    def run_or_load(self, force_train=False):
        """Conditionally runs training or loads the checkpoint"""
        adapter_exists = os.path.exists(Config.ADAPTER_OUTPUT_DIR)
        
        if adapter_exists and not force_train:
            app_logger.info("\n" + "="*50)
            app_logger.info(f"✅ Checkpoint found: Loading adapters from {Config.ADAPTER_OUTPUT_DIR}")
            app_logger.info("Skipping training")
            app_logger.info("="*50 + "\n")
            
            self.load_for_inference()
        else:
            if adapter_exists and force_train:
                app_logger.warning("⚠️ Checkpoint found, but re-training was forced")
            elif not adapter_exists:
                app_logger.error("❌ Checkpoint not found. Starting training from scratch")
            
            self.train()
            
            self.load_for_inference()


    def generate_response(self, instruction, max_new_tokens=100):
        """Generate a response using the merged model"""
        
        if self.merged_model is None:
            raise ValueError("El modelo de inferencia no ha sido cargado. Ejecuta .load_for_inference() primero")

        prompt = f"Query: {instruction}\nResponse:"
        
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=Config.SFT_CONFIG.max_seq_length)
        device = self.merged_model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_tokens = self.merged_model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )

        full_response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        
        try:
            # Clear the response prompt
            model_response = full_response.split("Response:")[1].strip()
        except IndexError:
            model_response = "Error decoding the response"

        return model_response