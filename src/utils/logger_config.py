"""
Конфигурация логирования для системы
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = None, level: int = logging.INFO,
                log_file: str = None, console: bool = True) -> logging.Logger:
    """
    Настройка логгера для системы
    
    Args:
        name: Имя логгера (по умолчанию корневой)
        level: Уровень логирования
        log_file: Путь к файлу логов (опционально)
        console: Выводить логи в консоль
        
    Returns:
        Настроенный логгер
    """
    # Создание папки для логов
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Получение или создание логгера
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Очистка существующих обработчиков
    logger.handlers.clear()
    
    # Формат сообщений
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Консольный вывод
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Файловый вывод
    if log_file is None:
        log_file = logs_dir / f"voice_system_{datetime.now().strftime('%Y%m%d')}.log"
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # В файл пишем все
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Предотвращение дублирования сообщений
    logger.propagate = False
    
    return logger


def setup_colored_logger(name: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка цветного логгера (если доступен colorama)
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        
    Returns:
        Настроенный логгер с цветным выводом
    """
    try:
        from colorama import init, Fore, Style
        init(autoreset=True)
        
        class ColoredFormatter(logging.Formatter):
            """Форматтер с цветным выводом"""
            
            COLORS = {
                'DEBUG': Fore.CYAN,
                'INFO': Fore.GREEN,
                'WARNING': Fore.YELLOW,
                'ERROR': Fore.RED,
                'CRITICAL': Fore.MAGENTA + Style.BRIGHT
            }
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, '')
                record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
                return super().format(record)
        
        logger = setup_logger(name, level, console=False)  # Без стандартного консольного вывода
        
        # Цветной консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        colored_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(colored_formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    except ImportError:
        # Fallback к обычному логгеру
        return setup_logger(name, level)


# Настройка логирования для внешних библиотек
def configure_external_loggers():
    """Настройка логирования для внешних библиотек"""
    
    # Уменьшение детализации логов библиотек
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    # Отключение отладочных сообщений PyTorch
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    # Отключение предупреждений librosa
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="librosa")


# Автоматическая настройка при импорте
configure_external_loggers()
