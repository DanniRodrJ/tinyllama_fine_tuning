from datasets import Dataset, load_dataset
from collections import defaultdict
import random
from config.config import TravelAssistantConfig as Config
from src.utils.logger import app_logger

def clean_response(response: str) -> str:
    """Reemplaza los placeholders del dataset por texto real"""
    replacements = {
        "{{WEBSITE_URL}}": "the airline's website",
        "{{APP_NAME}}": "the mobile app",
        "{{BOOKINGS_OPTION}}": "Bookings",
        "{{APP}}": "the mobile app",
        "{{CUSTOMER_SERVICE}}": "customer service"
    }
    for placeholder, value in replacements.items():
        response = response.replace(placeholder, value)
    return response

def merger_conversation(row):
    """Converts 'instruction' and 'response' columns to conversational format"""
    
    row['conversation'] = f"Query: {row['instruction']}\nResponse: {row['response']}"
    return row

def load_and_prepare_dataset():
    """Loads, samples, and formats the dataset for fine-tuning"""
    
    app_logger.info(f"Loading base dataset from Hugging Face: {Config.DATASET_NAME}")
    ds = load_dataset(Config.DATASET_NAME, split=Config.DATASET_SPLIT)
    
    random.seed(42)
    intent_groups = defaultdict(list)
    for record in ds:
        intent_groups[record["intent"]].append(record)

    total_intents = len(intent_groups)
    samples_per_intent = 100 // total_intents 

    balanced_subset = []
    for intent, examples in intent_groups.items():
        sampled = random.sample(examples, min(samples_per_intent, len(examples)))
        balanced_subset.extend(sampled)

    # Limit to the total number of desired records
    travel_chat_ds = Dataset.from_list(balanced_subset[:Config.TRAIN_RECORDS_LIMIT])
    
    app_logger.info(f"Applying balanced sampling: {len(travel_chat_ds)} final records")
    travel_chat_ds_ = travel_chat_ds.map(merger_conversation)
    
    return travel_chat_ds_

if __name__ == '__main__':
    prepared_ds = load_and_prepare_dataset()
    app_logger.info(prepared_ds)