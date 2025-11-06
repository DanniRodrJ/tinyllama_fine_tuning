# PROJECT

## Estructura del Proyecto

```bash
tinyllama_fine_tuning/         
│
├── config/                   
│   └── config.py
│
├── data/
│   ├── prepocessed_dataset/                 
│   │   ├── data-00000-of-00001.arrow
│   │   ├── dataset_info.json
│   │   └── state.json     
│
├── models/
│   ├── tinyllama_travel_adapter/                 
│   │   ├── adapter_config.json
│   │   ├── adapter_model.safetensors
│   │   ├── added_tokens.json
│   │   ├── README.md
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   └── tokenizer.model
│
├── notebooks/              
│   └── development.ipynb
│
├── src/                 
│   ├── utils/           
│   │   ├── logger.py
│   ├── data_preparation.py   
│   └── pipeline.py
│
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── main.py
├── README.md
└── requirements.txt    
```
