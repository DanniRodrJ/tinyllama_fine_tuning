import logging
import time
from typing import Callable, Any
import functools
from config.config import TravelAssistantConfig as Config

def setup_logger(name: str, log_file: str = 'app.log', level: int = logging.INFO, enable: bool = True) -> logging.Logger:

    logger = logging.getLogger(name)
    logger.setLevel(level if enable else logging.CRITICAL)  # Desactiva si enable es False
    
    if logger.handlers:
        return logger
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def timer_decorator(logger: logging.Logger = None):

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            
            result = func(*args, **kwargs)
            
            elapsed_time = time.time() - start_time
            minutes, seconds = divmod(elapsed_time, 60)
            
            message = f"Función '{func.__name__}' ejecutada en: {int(minutes)}m {seconds:.2f}s"
            
            if logger:
                logger.info(message)
            else:
                print(message)
            
            return result
        return wrapper
    return decorator

LOG_FILE = 'logfile.txt' if Config.SHOW_LOGGING and Config.LOGGIN_TXT else 'logs/app.log'
app_logger = setup_logger('fine_tuning_llama', LOG_FILE, logging.INFO, enable=Config.SHOW_LOGGING)

