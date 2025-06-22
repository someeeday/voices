import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
import logging
import os
import json
from typing import Dict, List, Optional, Union, Tuple, Any
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import torch.serialization
from tqdm import tqdm

# Импорт оптимизаторов процессора
try:
    from src.utils.simple_cpu_optimizer import SimpleCPUOptimizer
    SIMPLE_CPU_OPTIMIZER_AVAILABLE = True
except Exception:
    SIMPLE_CPU_OPTIMIZER_AVAILABLE = False

try:
    from src.utils.simple_intel_optimizer import SimpleIntelOptimizer
    SIMPLE_INTEL_OPTIMIZER_AVAILABLE = True
except Exception:
    SIMPLE_INTEL_OPTIMIZER_AVAILABLE = False

# Добавление безопасных глобальных объектов для сериализации
try:
    import numpy.core.multiarray
    torch.serialization.add_safe_globals([
        StandardScaler,
        numpy.core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype
    ])
except Exception as e:
    # Обработаем ошибку в методе load_model
    pass

# Проверка на доступность AMP (Automatic Mixed Precision)
AMP_AVAILABLE = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()


class ResidualBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.3):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.BatchNorm1d(in_dim)
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.block(x)
        out += identity
        return self.relu(out)


class SpeakerEmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int = 256,
                 hidden_layers: Optional[List[int]] = None, dropout: float = 0.3,
                 use_residual: bool = True):
        super(SpeakerEmbeddingNet, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = [512, 512, 384]
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Initial batch normalization for input features
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            if use_residual and i > 0 and prev_dim == hidden_dim:
                layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, dropout))
                
            prev_dim = hidden_dim
        
        # Final embedding layer with additional batch norm
        self.feature_layers = nn.Sequential(*layers)
        self.embedding_layer = nn.Linear(prev_dim, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.feature_layers(x)
        x = self.embedding_layer(x)
        x = self.embedding_bn(x)
        
        # L2-нормализация для получения эмбеддингов единичной длины
        # что важно для метрик на основе косинусного расстояния
        x = F.normalize(x, p=2, dim=1)
        return x


class SpeakerClassifier(nn.Module):
    def __init__(self, embedding_dim: int, num_speakers: int, 
                 use_arcface: bool = True, scale: float = 30.0, margin: float = 0.3,
                 dropout: float = 0.5):
        super(SpeakerClassifier, self).__init__()
        self.use_arcface = use_arcface
        self.scale = scale
        self.margin = margin
        self.dropout = nn.Dropout(dropout)
        
        if use_arcface:
            # Инициализируем веса в ArcFace стиле
            self.weight = nn.Parameter(torch.FloatTensor(num_speakers, embedding_dim))
            nn.init.xavier_normal_(self.weight)
        else:
            # Обычный полносвязный слой
            self.fc = nn.Linear(embedding_dim, num_speakers)
            nn.init.xavier_normal_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.dropout(x)
        
        if self.use_arcface and labels is not None:
            # Нормализация весов
            w_norm = F.normalize(self.weight, p=2, dim=1)
            
            # Косинусное сходство
            cos_theta = F.linear(x, w_norm)
            cos_theta = cos_theta.clamp(-1, 1)
            
            # Применение штрафа ArcFace для классов
            phi = torch.cos(torch.acos(cos_theta) + self.margin)
            
            # Применение штрафа только к соответствующим классам
            one_hot = torch.zeros_like(cos_theta)
            one_hot.scatter_(1, labels.view(-1, 1), 1)
            output = torch.where(one_hot == 1, phi, cos_theta)
            
            # Масштабирование для лучшей сходимости
            output = output * self.scale
            return output
        elif self.use_arcface:
            # При выводе без меток (инференс)
            w_norm = F.normalize(self.weight, p=2, dim=1)
            cos_theta = F.linear(x, w_norm)
            return cos_theta * self.scale
        else:
            # Обычная линейная классификация
            return self.fc(x)


class SpeakerRecognitionModel:
    def __init__(self, config: Optional[Dict[str, Any]] = None, for_inference: bool = False):
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.embedding_net = None
        self.classifier = None
        
        # Сохраняем исходные размерности для работы с JIT
        self._original_input_dim = None
        self._original_embedding_dim = None
        
        # Оптимизатор CPU
        self.cpu_optimizer = None
        self.model_save_path = None
        self.scaler = StandardScaler()
        self.speaker_mapping = {}
        
        # Настройка устройства
        self.device = self._setup_device()
        
        # Добавляем mixed precision для GPU
        self.mixed_precision = False
        self.scaler_amp = None
        if AMP_AVAILABLE and not for_inference:
            self.mixed_precision = True
            self.scaler_amp = torch.cuda.amp.GradScaler()
            self.logger.info("Mixed precision enabled for training")
            
        # Инициализация CPU оптимизаций
        self._init_cpu_optimizers()
        
        # Настройка модели (если конфигурация предоставлена)
        if config:
            self.setup_model(config)

    def _setup_device(self) -> torch.device:
        """Определяет оптимальное устройство для модели."""
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"CUDA доступен. Используется GPU: {device_name}")
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.logger.info("Apple MPS доступен. Используется GPU на Mac")
            return torch.device("mps")
        else:
            self.logger.info("GPU недоступен. Используется CPU")
            return torch.device("cpu")
    
    def _init_cpu_optimizers(self):
        """Инициализирует оптимизаторы CPU, если доступны."""
        # Сначала проверяем Intel оптимизатор
        if SIMPLE_INTEL_OPTIMIZER_AVAILABLE:
            try:
                from src.utils.simple_intel_optimizer import SimpleIntelOptimizer
                self.cpu_optimizer = SimpleIntelOptimizer(self.logger)
                self.logger.info("Intel CPU оптимизации инициализированы")
                return
            except Exception as e:
                self.logger.debug(f"Не удалось инициализировать Intel оптимизации: {e}")
        
        # Затем проверяем стандартный CPU оптимизатор
        if SIMPLE_CPU_OPTIMIZER_AVAILABLE:
            try:
                from src.utils.simple_cpu_optimizer import SimpleCPUOptimizer
                self.cpu_optimizer = SimpleCPUOptimizer(self.logger)
                self.logger.info("CPU оптимизации инициализированы")
            except Exception as e:
                self.logger.debug(f"Не удалось инициализировать CPU оптимизации: {e}")
                
    def setup_model(self, config: Dict[str, Any]) -> None:
        """Настройка модели с заданной конфигурацией."""
        # Извлечение параметров конфигурации
        input_dim = config.get("input_dim", 40)
        embedding_dim = config.get("embedding_dim", 256)
        hidden_layers = config.get("hidden_layers", [512, 512, 384])
        dropout = config.get("dropout", 0.3)
        num_speakers = config.get("num_speakers", 0)
        use_arcface = config.get("use_arcface", True)
        use_residual = config.get("use_residual", True)
        
        # Сохраняем исходные размерности (важно для JIT)
        self._original_input_dim = input_dim
        self._original_embedding_dim = embedding_dim
        self._original_use_arcface = use_arcface  # Сохраняем для доступа после JIT
        
        # Создание слоя эмбеддингов
        self.embedding_net = SpeakerEmbeddingNet(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_layers=hidden_layers,
            dropout=dropout,
            use_residual=use_residual
        )
        
        # Создание классификатора (если нужно)
        if num_speakers > 0:
            self.classifier = SpeakerClassifier(
                embedding_dim=embedding_dim,
                num_speakers=num_speakers,
                use_arcface=use_arcface,
                dropout=config.get("classifier_dropout", 0.5)
            )
        
        # Перемещение модели на соответствующее устройство
        self.embedding_net.to(self.device)
        if self.classifier is not None:
            self.classifier.to(self.device)
        
        # Применение CPU оптимизаций, если доступны и нужны
        # ВРЕМЕННО ОТКЛЮЧЕНО: JIT-компиляция может нарушать работу градиентов
        # TODO: Исправить совместимость JIT с обучением
        # if self.device.type == "cpu" and self.cpu_optimizer:
        #     # Создание пробного ввода для JIT-компиляции
        #     sample_input = torch.randn(1, input_dim).to(self.device)
        #     self.embedding_net = self.cpu_optimizer.optimize_model(
        #         self.embedding_net, sample_input)
        #     if self.classifier is not None and not use_arcface:
        #         # ArcFace с его динамическим штрафом не подходит для JIT-компиляции
        #         sample_embedding = self.embedding_net(sample_input)
        #         self.classifier = self.cpu_optimizer.optimize_model(
        #             self.classifier, sample_embedding)
            
    def get_batch_size(self) -> int:
        """Определяет оптимальный размер батча в зависимости от устройства."""
        if self.device.type == "cuda":
            mem_info = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            if mem_info >= 8:
                return 128
            elif mem_info >= 4:
                return 64
            else:
                return 32
        elif hasattr(torch.backends, 'mps') and self.device.type == "mps":
            return 64  # Среднее значение для Apple Silicon
        elif self.cpu_optimizer:
            return self.cpu_optimizer.get_training_config().get("batch_size", 32)
        else:
            return 32

    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.1,
              epochs: int = 30, 
              batch_size: Optional[int] = None,
              learning_rate: float = 0.001,
              weight_decay: float = 1e-4,
              early_stopping_patience: int = 5,
              output_path: Optional[str] = None,
              speakers_mapping: Optional[Dict[int, str]] = None) -> Dict[str, List[float]]:
        """
        Обучает модель распознавания говорящего.
        
        Args:
            X: Признаки
            y: Метки классов
            validation_split: Доля данных для валидации
            epochs: Количество эпох обучения
            batch_size: Размер батча (если None, определяется автоматически)
            learning_rate: Скорость обучения
            weight_decay: Коэффициент регуляризации весов
            early_stopping_patience: Терпение для раннего останова
            output_path: Путь для сохранения модели
            speakers_mapping: Маппинг ID спикеров в имена
        
        Returns:
            Dict с историей обучения
        """
        if batch_size is None:
            batch_size = self.get_batch_size()
            self.logger.info(f"Автоматически определен размер батча: {batch_size}")
            
        # Предобрабатываем данные (стандартизация)
        X_scaled = self.scaler.fit_transform(X)
        
        # Проверка и настройка модели, если необходимо
        num_classes = len(set(y))
        input_dim = X.shape[1]
        
        # Убеждаемся, что модель правильно инициализирована
        config = {
            "input_dim": input_dim,
            "embedding_dim": 256,  # Default value
            "num_speakers": num_classes,
            "use_arcface": False,  # Отключаем ArcFace для более простого обучения
            "use_residual": True,
            "hidden_layers": [512, 512, 384],
            "dropout": 0.3,
            "classifier_dropout": 0.5
        }
        self.setup_model(config)
        
        # Убеждаемся, что все параметры требуют градиенты
        for param in self.embedding_net.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        
        # Преобразуем метки в последовательные числа от 0 до num_classes-1
        unique_labels = sorted(set(y))
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        y_mapped = np.array([label_mapping[label] for label in y])
        
        # Сохранение маппинга спикеров
        self.speaker_mapping = {idx: f"speaker_{label}" for label, idx in label_mapping.items()}
        if speakers_mapping:
            # Обновляем с предоставленным маппингом
            for orig_label, speaker_name in speakers_mapping.items():
                if orig_label in label_mapping:
                    idx = label_mapping[orig_label]
                    self.speaker_mapping[idx] = speaker_name
        
        # Разделение на обучающую и валидационную выборки
        # Проверяем, достаточно ли данных для разделения
        min_samples_for_split = max(2, num_classes * 2)  # Минимум 2 сэмпла на класс
        
        if len(X_scaled) < min_samples_for_split:
            # Слишком мало данных для разделения - используем все данные для обучения
            self.logger.warning(f"Too few samples ({len(X_scaled)}) for train/val split. Using all data for training.")
            X_train, X_val = X_scaled, X_scaled
            y_train, y_val = y_mapped, y_mapped
            validation_split = 0.0
        else:
            # Адаптируем validation_split если данных мало
            if len(X_scaled) < 50:
                validation_split = min(validation_split, 0.2)  # Максимум 20% для валидации при малых данных
            
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_mapped, test_size=validation_split, random_state=42, stratify=y_mapped)
            except ValueError as e:
                self.logger.warning(f"Stratified split failed: {e}. Using random split.")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_mapped, test_size=validation_split, random_state=42)
        
        # Преобразование в тензоры PyTorch
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.LongTensor(y_val)
        
        # Создание даталоадеров
        dataloader_kwargs = {"batch_size": batch_size, "shuffle": True}
        if self.cpu_optimizer and self.device.type == "cpu":
            dataloader_kwargs = self.cpu_optimizer.optimize_dataloader(dataloader_kwargs)
        
        train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
        
        val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=dataloader_kwargs.get("num_workers", 0))
        
        # Настройка оптимизатора и планировщика
        optimizer = optim.AdamW([
            {'params': self.embedding_net.parameters()},
            {'params': self.classifier.parameters()}
        ], lr=learning_rate, weight_decay=weight_decay)
        
        # CosineAnnealingWarmRestarts с перезапуском каждые 5 эпох
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=1, eta_min=learning_rate/10)
        
        # Для сравнения логитов (используем сохраненное значение)
        use_arcface = getattr(self, '_original_use_arcface', getattr(self.classifier, 'use_arcface', False))
        criterion = nn.CrossEntropyLoss()
        
        # История обучения
        history = {
            "train_loss": [], 
            "val_loss": [], 
            "train_acc": [], 
            "val_acc": []
        }
        
        # Для раннего останова
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_weights = None
        
        # Обучение модели
        for epoch in range(epochs):
            # ========== ОБУЧЕНИЕ ==========
            self.embedding_net.train()
            self.classifier.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Обнуление градиентов
                optimizer.zero_grad()
                
                # Forward pass с mixed precision, если доступно
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        embeddings = self.embedding_net(inputs)
                        if use_arcface:
                            outputs = self.classifier(embeddings, labels)
                        else:
                            outputs = self.classifier(embeddings)
                        loss = criterion(outputs, labels)
                    
                    # Backward с масштабированием
                    self.scaler_amp.scale(loss).backward()
                    self.scaler_amp.step(optimizer)
                    self.scaler_amp.update()
                else:
                    # Обычный forward pass
                    embeddings = self.embedding_net(inputs)
                    if use_arcface:
                        outputs = self.classifier(embeddings, labels)
                    else:
                        outputs = self.classifier(embeddings)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                
                # Сбор статистики
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Отладочная информация для первого батча первой эпохи
                if epoch == 0 and train_total <= batch_size:
                    self.logger.info(f"First batch debug info:")
                    self.logger.info(f"  - Batch size: {inputs.size(0)}")
                    self.logger.info(f"  - Input shape: {inputs.shape}")
                    self.logger.info(f"  - Labels shape: {labels.shape}")
                    self.logger.info(f"  - Labels range: {labels.min().item()} to {labels.max().item()}")
                    self.logger.info(f"  - Outputs shape: {outputs.shape}")
                    self.logger.info(f"  - Unique labels in batch: {torch.unique(labels).numel()}")
                    self.logger.info(f"  - Use ArcFace: {self.classifier.use_arcface}")
                    self.logger.info(f"  - Loss: {loss.item():.4f}")
                    self.logger.info(f"  - Predictions: {predicted[:10].tolist()}")
                    self.logger.info(f"  - True labels: {labels[:10].tolist()}")
                    self.logger.debug(f"  outputs shape: {outputs.shape}")
                    self.logger.debug(f"  labels shape: {labels.shape}")
                    self.logger.debug(f"  labels unique: {torch.unique(labels)}")
                    self.logger.debug(f"  predicted unique: {torch.unique(predicted)}")
                    self.logger.debug(f"  num_classes in config: {num_classes}")
                    self.logger.debug(f"  outputs min/max: {outputs.min().item():.4f}/{outputs.max().item():.4f}")
                
                # Обновление прогресс-бара
                progress_bar.set_postfix({
                    'loss': loss.item(),
                    'acc': train_correct / train_total
                })
            
            # Обновление планировщика
            scheduler.step()
            
            # ========== ВАЛИДАЦИЯ ==========
            self.embedding_net.eval()
            self.classifier.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    embeddings = self.embedding_net(inputs)
                    if self.classifier.use_arcface:
                        outputs = self.classifier(embeddings, labels)
                    else:
                        outputs = self.classifier(embeddings)
                    loss = criterion(outputs, labels)
                    
                    # Сбор статистики
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            # Расчет метрик за эпоху
            epoch_train_loss = train_loss / len(X_train)
            epoch_val_loss = val_loss / len(X_val)
            epoch_train_acc = train_correct / train_total
            epoch_val_acc = val_correct / val_total
            
            # Сохранение истории
            history["train_loss"].append(epoch_train_loss)
            history["val_loss"].append(epoch_val_loss)
            history["train_acc"].append(epoch_train_acc)
            history["val_acc"].append(epoch_val_acc)
            
            # Логирование
            self.logger.info(
                f"Эпоха {epoch+1}/{epochs} - "
                f"loss: {epoch_train_loss:.4f}, "
                f"val_loss: {epoch_val_loss:.4f}, "
                f"acc: {epoch_train_acc:.4f}, "
                f"val_acc: {epoch_val_acc:.4f}"
            )
            
            # Проверка для раннего останова
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_weights = {
                    'embedding_net': self.embedding_net.state_dict(),
                    'classifier': self.classifier.state_dict()
                }
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    self.logger.info(f"Ранний останов на эпохе {epoch+1}")
                    break
        
        # Восстановление лучших весов модели
        if best_model_weights:
            self.embedding_net.load_state_dict(best_model_weights['embedding_net'])
            self.classifier.load_state_dict(best_model_weights['classifier'])
        
        # Модель теперь обучена
        self.is_trained = True
        
        # Сохранение модели, если указан путь
        if output_path:
            self.save_model(output_path)
        
        return history
    
    def extract_embedding(self, features: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Извлекает эмбеддинг из входных признаков.
        
        Args:
            features: признаки входного аудио
        
        Returns:
            np.ndarray: эмбеддинг говорящего
        """
        if self.embedding_net is None:
            raise ValueError("Модель не инициализирована или не обучена")
        
        # Преобразование в формат тензора, если необходимо
        if isinstance(features, np.ndarray):
            features = self.scaler.transform(features.reshape(1, -1))
            features = torch.FloatTensor(features).to(self.device)
        elif isinstance(features, torch.Tensor) and features.device != self.device:
            features = features.to(self.device)
        
        # Извлечение эмбеддинга
        self.embedding_net.eval()
        with torch.no_grad():
            embeddings = self.embedding_net(features)
        
        # Возврат в виде numpy array
        return embeddings.cpu().numpy()

    def predict(self, features: Union[np.ndarray, torch.Tensor]) -> Tuple[str, float]:
        """
        Предсказывает говорящего по входным признакам.
        
        Args:
            features: признаки входного аудио
        
        Returns:
            Tuple[str, float]: (идентификатор говорящего, вероятность)
        """
        if self.embedding_net is None or self.classifier is None:
            raise ValueError("Модель не инициализирована или не обучена")
        
        # Преобразование в формат тензора, если необходимо
        if isinstance(features, np.ndarray):
            features = self.scaler.transform(features.reshape(1, -1))
            features = torch.FloatTensor(features).to(self.device)
        elif isinstance(features, torch.Tensor) and features.device != self.device:
            features = features.to(self.device)
        
        # Предсказание класса
        self.embedding_net.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            embeddings = self.embedding_net(features)
            outputs = self.classifier(embeddings)
            probabilities = F.softmax(outputs, dim=1)
            
            # Получение наиболее вероятного класса и его вероятности
            prob, pred_idx = torch.max(probabilities, dim=1)
            speaker_id = pred_idx.item()
            probability = prob.item()
        
        # Получение имени говорящего
        speaker_name = self.speaker_mapping.get(speaker_id, f"speaker_{speaker_id}")
        
        return speaker_name, probability

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Оценивает производительность модели на тестовой выборке.
        
        Args:
            X: признаки
            y: истинные метки
        
        Returns:
            Dict с метриками оценки
        """
        if self.embedding_net is None or self.classifier is None:
            raise ValueError("Модель не инициализирована или не обучена")
        
        # Предобработка данных
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Создание даталоадера
        batch_size = self.get_batch_size()
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        
        # Оценка модели
        self.embedding_net.eval()
        self.classifier.eval()
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(dataloader, desc="Evaluating"):
                embeddings = self.embedding_net(inputs)
                outputs = self.classifier(embeddings)
                probabilities = F.softmax(outputs, dim=1)
                
                _, preds = torch.max(outputs, 1)
                
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probabilities.cpu().numpy())
        
        # Объединяем результаты
        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_labels)
        probs = np.concatenate(all_probs)
        
        # Расчет метрик
        accuracy = accuracy_score(y_true, y_pred)
        
        # Преобразование числовых меток в названия спикеров для отчета
        label_names = [self.speaker_mapping.get(i, f"speaker_{i}") for i in range(len(set(y_true)))]
        report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
        
        # Матрица ошибок
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        return {
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": conf_matrix
        }

    def save_model(self, output_path: str) -> None:
        """
        Сохраняет модель и конфигурацию.
        
        Args:
            output_path: путь для сохранения модели
        """
        if self.embedding_net is None:
            raise ValueError("Модель не инициализирована")
        
        # Создаем директорию, если не существует
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Сохраняем параметры модели
        model_state = {
            "embedding_net": self.embedding_net.state_dict(),
            "scaler": self.scaler,
            "is_trained": self.is_trained
        }
        
        if self.classifier is not None:
            model_state["classifier"] = self.classifier.state_dict()
        
        torch.save(model_state, output_path)
        
        # Сохраняем конфигурацию
        config_path = os.path.join(os.path.dirname(output_path), "config.json")
        
        # Используем сохраненные исходные размерности или получаем из модели
        input_dim = self._original_input_dim or getattr(self.embedding_net, 'input_dim', 51)
        embedding_dim = self._original_embedding_dim or getattr(self.embedding_net, 'embedding_dim', 256)
        
        config = {
            "input_dim": input_dim,
            "embedding_dim": embedding_dim,
            "use_arcface": getattr(self.classifier, 'use_arcface', True) if self.classifier else True,
            "num_speakers": len(self.speaker_mapping),
            "version": "1.1.0",  # Версия модели
            "timestamp": time.time()
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f)
        
        # Сохраняем маппинг спикеров
        speakers_path = os.path.join(os.path.dirname(output_path), "speakers.json")
        with open(speakers_path, "w") as f:
            json.dump(self.speaker_mapping, f)
        
        self.logger.info(f"Модель успешно сохранена в {output_path}")
        self.model_save_path = output_path

    def load_model(self, model_path: str, config_path: Optional[str] = None) -> None:
        """
        Загружает модель и конфигурацию.
        
        Args:
            model_path: путь к файлу модели
            config_path: путь к файлу конфигурации (опционально)
        """
        try:
            # Загружаем состояние модели
            model_state = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Если конфигурация не указана, ищем в той же директории
            if config_path is None:
                config_path = os.path.join(os.path.dirname(model_path), "config.json")
            
            # Загружаем конфигурацию
            with open(config_path, "r") as f:
                config = json.load(f)
            
            # Настраиваем модель
            self.setup_model(config)
            
            # Загружаем параметры
            self.embedding_net.load_state_dict(model_state["embedding_net"])
            if "classifier" in model_state and self.classifier is not None:
                self.classifier.load_state_dict(model_state["classifier"])
            
            # Загружаем скейлер
            self.scaler = model_state.get("scaler", StandardScaler())
            self.is_trained = model_state.get("is_trained", True)
            
            # Загружаем маппинг спикеров, если есть
            speakers_path = os.path.join(os.path.dirname(model_path), "speakers.json")
            if os.path.exists(speakers_path):
                with open(speakers_path, "r") as f:
                    self.speaker_mapping = json.load(f)
            
            self.logger.info(f"Модель успешно загружена из {model_path}")
            self.model_save_path = model_path
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            raise ValueError(f"Не удалось загрузить модель: {e}")
