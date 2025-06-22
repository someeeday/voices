"""
🚀 Простой Intel Ultra 9 Оптимизатор
Оптимизация для многоядерного CPU без сложных зависимостей
"""

import os
import sys
import logging
import psutil
from typing import Dict, Any, Optional, Tuple
import torch
import numpy as np
from pathlib import Path
import time

# Простые оптимизации без сложных зависимостей
try:
    import mkl
    MKL_AVAILABLE = True
except ImportError:
    MKL_AVAILABLE = False


class SimpleIntelOptimizer:
    """
    Простой оптимизатор для Intel Ultra 9 архитектуры
    
    Поддерживает:
    - Многоядерный CPU с оптимизациями
    - Intel MKL если доступен
    - PyTorch встроенные оптимизации
    - Автоматическое распределение нагрузки
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Инициализация оптимизатора
        
        Args:
            logger: Логгер для вывода информации
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Системная информация
        self.cpu_count = psutil.cpu_count(logical=True)
        self.cpu_count_physical = psutil.cpu_count(logical=False)
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Конфигурация
        self.config = {
            "use_optimized_cpu": True,
            "num_threads": None,
            "mixed_precision": False,  # Отключено для стабильности
            "jit_compile": True
        }
        
        self._optimize_environment()
    
    def _optimize_environment(self):
        """Оптимизация окружения для Intel Ultra 9"""
        
        # Автоматическое определение оптимального количества потоков
        optimal_threads = min(self.cpu_count_physical, 16)  # Не более 16 для стабильности
        
        # PyTorch оптимизации
        torch.set_num_threads(optimal_threads)
        self.config["num_threads"] = optimal_threads
        
        # Переменные окружения для оптимизации
        os.environ["OMP_NUM_THREADS"] = str(optimal_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(optimal_threads)
        
        # Intel MKL оптимизации если доступны
        if MKL_AVAILABLE:
            try:
                mkl.set_num_threads(optimal_threads)
                os.environ["MKL_NUM_THREADS"] = str(optimal_threads)
                self.logger.info(f"🔧 Intel MKL настроен: {optimal_threads} потоков")
            except Exception as e:
                self.logger.debug(f"Ошибка настройки MKL: {e}")
        
        # Включение оптимизаций для Intel CPU
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        # OpenMP оптимизации
        os.environ["KMP_BLOCKTIME"] = "1"
        os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
        
        self.logger.info(f"⚙️ Окружение оптимизировано для Intel Ultra 9: {optimal_threads} потоков")
    
    def get_optimal_torch_device(self) -> torch.device:
        """
        Получение оптимального PyTorch устройства
        
        Returns:
            CPU устройство с оптимизациями
        """
        return torch.device("cpu")
    
    def optimize_model(self, model: torch.nn.Module, 
                      sample_input: Optional[torch.Tensor] = None) -> torch.nn.Module:
        """
        Оптимизация PyTorch модели для Intel Ultra 9
        
        Args:
            model: PyTorch модель
            sample_input: Пример входных данных для оптимизации
            
        Returns:
            Оптимизированная модель
        """
        device = self.get_optimal_torch_device()
        
        # Перенос модели на CPU
        model = model.to(device)
        
        # JIT компиляция для дополнительной скорости
        if self.config["jit_compile"] and sample_input is not None:
            try:
                sample_input = sample_input.to(device)
                model.eval()
                with torch.no_grad():
                    # Создаем traced модель
                    traced_model = torch.jit.trace(model, sample_input)
                    # Оптимизируем для inference
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    self.logger.info("⚡ JIT компиляция и оптимизация включены")
                    return traced_model
            except Exception as e:
                self.logger.debug(f"JIT компиляция недоступна: {e}")
        
        return model
    
    def optimize_dataloader(self, dataloader_kwargs: Dict) -> Dict:
        """
        Оптимизация DataLoader для многоядерной загрузки
        
        Args:
            dataloader_kwargs: Параметры DataLoader
            
        Returns:
            Оптимизированные параметры
        """
        # Оптимальное количество worker'ов для загрузки данных
        optimal_workers = min(self.cpu_count_physical, 8)  # Не более 8 для стабильности
        
        dataloader_kwargs.update({
            "num_workers": optimal_workers,
            "pin_memory": False,  # Для CPU не нужно
            "persistent_workers": True if optimal_workers > 0 else False,
            "prefetch_factor": 2 if optimal_workers > 0 else None
        })
        
        self.logger.info(f"📊 DataLoader оптимизирован: {optimal_workers} worker'ов")
        return dataloader_kwargs
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Получение оптимальной конфигурации обучения
        
        Returns:
            Конфигурация для обучения
        """
        # Базовая конфигурация
        config = {
            "device": str(self.get_optimal_torch_device()),
            "mixed_precision": self.config["mixed_precision"],
            "compile_model": self.config["jit_compile"],
            "pin_memory": False,
            "non_blocking": False
        }
        
        # Размер батча в зависимости от доступной памяти и ядер
        if self.memory_gb >= 32:
            config["batch_size"] = 128
        elif self.memory_gb >= 16:
            config["batch_size"] = 64
        else:
            config["batch_size"] = 32
        
        # Адаптируем размер батча под количество ядер
        if self.cpu_count_physical >= 8:
            config["batch_size"] = min(config["batch_size"], 96)
        elif self.cpu_count_physical >= 4:
            config["batch_size"] = min(config["batch_size"], 64)
        
        # Количество worker'ов для загрузки данных
        config["num_workers"] = min(self.cpu_count_physical, 8)
        
        return config
    
    def benchmark_cpu_performance(self, model: torch.nn.Module, 
                                 sample_input: torch.Tensor, 
                                 num_iterations: int = 100) -> Dict[str, float]:
        """
        Бенчмарк производительности CPU
        
        Args:
            model: Модель для тестирования
            sample_input: Пример входных данных
            num_iterations: Количество итераций для теста
            
        Returns:
            Результаты бенчмарка (время в мс)
        """
        results = {}
        
        try:
            device = torch.device("cpu")
            test_model = model.to(device)
            test_input = sample_input.to(device)
            
            # Тест обычной модели
            test_model.eval()
            # Прогрев
            with torch.no_grad():
                for _ in range(10):
                    _ = test_model(test_input)
            
            # Измерение времени
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    _ = test_model(test_input)
            end_time = time.time()
            
            avg_time_ms = (end_time - start_time) * 1000 / num_iterations
            results["cpu_normal"] = avg_time_ms
            
            # Тест оптимизированной модели
            try:
                optimized_model = self.optimize_model(test_model, test_input)
                
                # Прогрев оптимизированной модели
                with torch.no_grad():
                    for _ in range(10):
                        _ = optimized_model(test_input)
                
                # Измерение времени оптимизированной модели
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(num_iterations):
                        _ = optimized_model(test_input)
                end_time = time.time()
                
                avg_time_optimized_ms = (end_time - start_time) * 1000 / num_iterations
                results["cpu_optimized"] = avg_time_optimized_ms
                
                # Вычисляем ускорение
                speedup = avg_time_ms / avg_time_optimized_ms
                results["speedup"] = speedup
                
            except Exception as e:
                self.logger.debug(f"Ошибка тестирования оптимизированной модели: {e}")
            
            self.logger.info(f"⏱️ CPU обычная модель: {results.get('cpu_normal', 0):.2f} мс/итерация")
            if "cpu_optimized" in results:
                self.logger.info(f"⚡ CPU оптимизированная: {results['cpu_optimized']:.2f} мс/итерация")
                self.logger.info(f"🚀 Ускорение: {results.get('speedup', 1):.2f}x")
                
        except Exception as e:
            self.logger.error(f"Ошибка бенчмарка: {e}")
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Получение информации о системе
        
        Returns:
            Информация о системе и доступных оптимизациях
        """
        return {
            "cpu": {
                "physical_cores": self.cpu_count_physical,
                "logical_cores": self.cpu_count,
                "optimal_threads": self.config.get("num_threads", self.cpu_count)
            },
            "memory": {
                "total_gb": round(self.memory_gb, 1),
                "available_gb": round(psutil.virtual_memory().available / (1024**3), 1)
            },
            "optimizations": {
                "mkl_available": MKL_AVAILABLE,
                "torch_threads": torch.get_num_threads(),
                "mkldnn_enabled": getattr(torch.backends.mkldnn, 'enabled', False)
            },
            "optimal_device": "Optimized CPU",
            "config": self.config
        }


# Глобальный экземпляр оптимизатора
_simple_intel_optimizer = None

def get_simple_intel_optimizer(logger: Optional[logging.Logger] = None) -> SimpleIntelOptimizer:
    """Получение глобального экземпляра простого Intel оптимизатора"""
    global _simple_intel_optimizer
    if _simple_intel_optimizer is None:
        _simple_intel_optimizer = SimpleIntelOptimizer(logger)
    return _simple_intel_optimizer
