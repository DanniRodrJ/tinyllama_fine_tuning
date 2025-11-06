import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from peft import PeftModel
import os
from config.config import TravelAssistantConfig as Config
from src.data_preparation import load_and_prepare_dataset


class TravelAssistantPipeline:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.merged_model = None
        self.train_dataset = load_and_prepare_dataset()
        
        if not os.path.exists(Config.PREPROCESSED_DATA_DIR):
            self.train_dataset.save_to_disk(Config.PREPROCESSED_DATA_DIR)
    
    def _load_base_components(self):
        """Carga el modelo base y el tokenizer."""
        
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model is None:
            self.model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)

    def train(self):
        """Ejecuta el fine-tuning y guarda los adaptadores."""
        
        self._load_base_components()
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=self.train_dataset,
            args=Config.SFT_CONFIG, 
            peft_config=Config.LORA_CONFIG,
        )

        print("Iniciando el fine-tuning...")
        trainer.train()
        print("Fine-tuning completado.")

        # Guardar adaptadores LoRA y tokenizer
        trainer.model.save_pretrained(Config.ADAPTER_OUTPUT_DIR)
        self.tokenizer.save_pretrained(Config.ADAPTER_OUTPUT_DIR)
        print(f"Adaptadores LoRA guardados en: {Config.ADAPTER_OUTPUT_DIR}")

    def load_for_inference(self):
        """Carga el modelo base y fusiona los adaptadores para inferencia."""
        
        # Definir el directorio de offload (Si no cuentas con GPU)
        OFFLOAD_FOLDER = os.path.join(os.getcwd(), "offload_temp")
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
        
        # Cargar Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.ADAPTER_OUTPUT_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Cargar el modelo base
        base_model_for_inference = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME,
            torch_dtype=torch.float16, 
            device_map="auto",
            offload_folder=OFFLOAD_FOLDER
        )

        # Cargar y fusionar adaptadores
        model_with_peft = PeftModel.from_pretrained(
            base_model_for_inference, 
            Config.ADAPTER_OUTPUT_DIR
        )

        self.merged_model = model_with_peft.merge_and_unload()
        self.merged_model.eval() 
        print("Modelo base y adaptadores LoRA fusionados.")
        
        
    def run_or_load(self, force_train=False):
        """Condiciona la ejecución: si existen checkpoints, salta el entrenamiento."""
        adapter_exists = os.path.exists(Config.ADAPTER_OUTPUT_DIR)
        
        if adapter_exists and not force_train:
            print("\n" + "="*50)
            print(f"✅ Checkpoint encontrado: Cargando adaptadores desde {Config.ADAPTER_OUTPUT_DIR}")
            print("Saltando el entrenamiento.")
            print("="*50 + "\n")
            
            self.load_for_inference()
        else:
            if adapter_exists and force_train:
                print("⚠️ Se encontró el checkpoint, pero se forzó el re-entrenamiento.")
            elif not adapter_exists:
                print("❌ No se encontró el checkpoint. Iniciando entrenamiento de cero.")
            
            self.train()
            
            self.load_for_inference()


    def generate_response(self, instruction, max_new_tokens=100):
        """Genera una respuesta usando el modelo fusionado."""
        
        if self.merged_model is None:
            raise ValueError("El modelo de inferencia no ha sido cargado. Ejecuta .load_for_inference() primero.")

        prompt = f"Query: {instruction}\nResponse:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
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
            # Limpiar el prompt de la respuesta
            model_response = full_response.split("Response:")[1].strip()
        except IndexError:
            model_response = "Error al decodificar la respuesta."

        return model_response