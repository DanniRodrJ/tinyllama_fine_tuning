from datasets import Dataset, load_dataset
from collections import defaultdict
import random
from src.config import DATASET_NAME, DATASET_SPLIT, TRAIN_RECORDS_LIMIT

def merger_conversation(row):
    """Convierte las columnas 'instruction' y 'response' a un formato conversacional"""
    
    row['conversation'] = f"Query: {row['instruction']}\nResponse: {row['response']}"
    return row

def load_and_prepare_dataset():
    """Carga, muestrea y formatea el dataset para fine-tuning"""
    
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    
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

    # Limitar al número total de registros deseados
    travel_chat_ds = Dataset.from_list(balanced_subset[:TRAIN_RECORDS_LIMIT])
    
    travel_chat_ds_ = travel_chat_ds.map(merger_conversation)
    
    return travel_chat_ds_

if __name__ == '__main__':
    prepared_ds = load_and_prepare_dataset()
    print(prepared_ds)