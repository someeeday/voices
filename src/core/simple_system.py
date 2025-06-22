"""
Main speaker recognition system
Simple file-based system for academic projects
"""

import json
import logging
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import uuid
import torch

from src.models.speaker_recognition_model import SpeakerRecognitionModel
from src.audio.audio_processor import AudioProcessor
from src.data.user_processor import UserProcessor
from src.utils.file_utils import ensure_dir_exists
from src.evaluation.metrics import PerformanceMetrics


class SimpleSpeakerSystem:
    """
    Simple speaker identification system for academic projects
    
    Features:
    - File-based JSON storage
    - 20-30k samples, 200-500 speakers
    - Identification in 100-500ms
    - Compact model for desktop
    """
    
    def __init__(self, data_dir: str = "data", model_dir: str = "data/models"):
        """
        System initialization
        
        Args:
            data_dir: Root data folder
            model_dir: Folder for saving models
        """
        self.logger = logging.getLogger(__name__)
        
        # Настройка путей
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.database_dir = self.data_dir / "user_database"
        
        # Создание необходимых папок        self._setup_directories()
        
        # Инициализация компонентов
        self.audio_processor = AudioProcessor()
        self.user_database = UserProcessor(str(self.database_dir))
        self.model = SpeakerRecognitionModel()
        self.metrics = PerformanceMetrics()
        
        # Load model if exists
        self._load_model_if_exists()

    def _setup_directories(self):
        """Create required directories"""
        directories = [
            self.data_dir,
            self.model_dir,
            self.database_dir,
            self.database_dir / "embeddings",
            self.database_dir / "metadata",
            self.data_dir / "datasets",
            self.data_dir / "processed",
            "test_data",
            "logs"
        ]
        
        for directory in directories:
            ensure_dir_exists(directory)

    def _load_model_if_exists(self):
        """Load trained model if it exists"""
        model_path = self.model_dir / "speaker_model.pth"
        config_path = self.model_dir / "config.json"
        
        if model_path.exists() and config_path.exists():
            try:
                self.model.load_model(str(model_path), str(config_path))
                self.logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load model: {e}")
        else:
            self.logger.info("Model not found, training required")

    def register_user(self, name: str, audio_files: List[str]) -> str:
        """
        Register a new user in the system
        
        Args:
            name: User name
            audio_files: List of audio file paths
            
        Returns:
            UUID of the new user
        """
        if not self.model.is_trained:
            raise ValueError("Model is not trained! Please train the model first.")
        
        self.logger.info(f"Registering user: {name}")
        
        # Проверка на дублирование по имени (без учёта регистра и пробелов)
        existing_users = self.user_database.get_all_users_list()
        norm_name = name.strip().lower()
        for u in existing_users:
            if u['name'].strip().lower() == norm_name:
                self.logger.info(f"User {name} already exists, skipping registration.")
                return u['uuid']
        
        # Обработка аудиофайлов
        embeddings = []
        valid_files = []
        
        for audio_file in audio_files:
            try:
                features = self.audio_processor.extract_features(audio_file)
                embedding = self.model.extract_embedding(features)
                embeddings.append(embedding)
                valid_files.append(audio_file)
                self.logger.debug(f"Processed file: {audio_file}")
            except Exception as e:
                self.logger.warning(f"Error processing {audio_file}: {e}")
        
        if not embeddings:
            raise ValueError("No audio files could be processed")
        
        # Создание усредненного эмбеддинга
        final_embedding = np.mean(embeddings, axis=0)
        
        # Анализ голосовых характеристик
        voice_characteristics = self._analyze_voice_characteristics(valid_files)
        
        # Сохранение в базу данных
        user_id = self.user_database.add_user(
            name=name,
            embedding=final_embedding
        )
        # Сохраняем список обработанных файлов в метаданные пользователя
        meta = self.user_database.get_user_metadata(user_id) or {}
        meta['processed_files'] = valid_files
        self.user_database.update_user_metadata(user_id, meta)
        self.logger.info(f"User {name} registered with ID: {user_id}")
        return user_id

    def identify_voice(self, audio_file: str, threshold: float = 0.7) -> Dict[str, Any]:
        """
        Identify speaker by audio file
        
        Args:
            audio_file: Path to audio file
            threshold: Confidence threshold for identification
            
        Returns:
            Identification result
        """
        start_time = time.time()
        
        if not self.model.is_trained:
            raise ValueError("Model is not trained! Please train the model first.")
        
        self.logger.info(f"Identifying voice: {audio_file}")
        
        try:
            # Извлечение признаков
            features = self.audio_processor.extract_features(audio_file)
            
            # Проверяем, есть ли пользователи в базе данных
            if len(self.user_database.users_data["users"]) > 0:
                # Используем поиск по эмбеддингам в базе пользователей
                test_embedding = self.model.extract_embedding(features)
                best_match = self.user_database.find_closest_user(test_embedding)
                
                processing_time = (time.time() - start_time) * 1000
                
                if best_match and best_match.get("similarity", 0) >= threshold and best_match.get("user_id"):
                    result = {
                        "success": True,
                        "user_id": best_match["user_id"],
                        "name": best_match["name"],
                        "confidence": best_match["similarity"],
                        "processing_time_ms": processing_time,
                        "audio_file": audio_file
                    }
                    self.logger.info(f"Identified: {best_match['name']} (confidence: {best_match['similarity']:.2f})")
                else:
                    # Если есть best_candidate, выводим на кого больше всего похоже
                    candidate = best_match.get('best_candidate') if best_match else None
                    if candidate and candidate.get('name') and candidate.get('similarity', 0) > 0:
                        msg = f"Неизвестный голос, но больше всего похож на: {candidate['name']} (уверенность: {candidate['similarity']:.2%})"
                        self.logger.info(msg)
                        result = {
                            "success": False,
                            "user_id": None,
                            "name": "Unknown user",
                            "confidence": best_match.get("similarity", 0.0),
                            "processing_time_ms": processing_time,
                            "audio_file": audio_file,
                            "best_candidate": candidate
                        }
                    else:
                        result = {
                            "success": False,
                            "user_id": None,
                            "name": "Unknown user",
                            "confidence": best_match["similarity"] if best_match else 0.0,
                            "processing_time_ms": processing_time,
                            "audio_file": audio_file
                        }
                    if candidate and candidate.get('name'):
                        self.logger.info(f"User not identified. Most similar: {candidate['name']} (confidence: {candidate['similarity']:.2f})")
                    else:
                        self.logger.info(f"User not identified (confidence: {result['confidence']:.2f})")
                return result
            else:
                # Используем прямую классификацию через обученную модель
                prediction = self._predict_with_trained_model(features)
                
                processing_time = (time.time() - start_time) * 1000
                
                if prediction["confidence"] >= threshold:
                    result = {
                        "success": True,
                        "user_id": f"speaker_{prediction['speaker_id']:03d}",
                        "name": prediction["speaker_name"],
                        "confidence": prediction["confidence"],
                        "processing_time_ms": processing_time,
                        "audio_file": audio_file
                    }
                    self.logger.info(f"Identified: {prediction['speaker_name']} (confidence: {prediction['confidence']:.2f})")
                else:
                    result = {
                        "success": False,
                        "user_id": None,
                        "name": "Unknown speaker",
                        "confidence": prediction["confidence"],
                        "processing_time_ms": processing_time,
                        "audio_file": audio_file
                    }
                    self.logger.info(f"Speaker not identified (confidence: {prediction['confidence']:.2f})")
                
                return result
            
        except Exception as e:
            self.logger.error(f"Identification error: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_file": audio_file,
                "processing_time_ms": (time.time() - start_time) * 1000
            }

    def train_model(self, epochs: int = 20, batch_size: int = 64, 
                   embedding_dim: int = 256, max_samples: int = 25000,
                   max_speakers: int = 500, learning_rate: float = 0.001):
        """
        Train speaker recognition model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            embedding_dim: Embedding dimension
            max_samples: Maximum number of samples
            max_speakers: Maximum number of speakers
            learning_rate: Learning rate
        """
        self.logger.info("Starting model training...")
        
        # Поиск обучающих данных
        processed_data_path = self.data_dir / "processed" / "training_data.npz"
        
        if not processed_data_path.exists():
            self.logger.error("Training data not found!")
            self.logger.info("Run: python src/main.py --prepare-datasets")
            return
        
        # Загрузка данных
        data = np.load(processed_data_path, allow_pickle=True)
        features = data["features"]
        labels = data["labels"]
        speaker_info = json.loads(data["speaker_info"].item())
        
        self.logger.info(f"Loaded: {len(features)} samples, "
                        f"{len(speaker_info)} speakers")
        
        # Ограничение данных если нужно
        if len(features) > max_samples:
            indices = np.random.choice(len(features), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
            self.logger.info(f"Limited to {max_samples} samples")
        
        # Настройка и обучение модели
        config = {
            "input_dim": features.shape[1],
            "embedding_dim": embedding_dim,
            "num_speakers": len(speaker_info),
            "hidden_layers": [512, 256],
            "dropout": 0.3
        }        # ВАЖНО: для обучения всегда for_inference=False
        self.model.setup_model(config)
        
        # Обучение
        history = self.model.train(
            features, labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            learning_rate=learning_rate
        )
          # Сохранение модели
        model_path = self.model_dir / "speaker_model.pth"
        config_path = self.model_dir / "config.json"
        
        self.model.save_model(str(model_path))
        
        # Сохранение информации о говорящих
        speakers_path = self.model_dir / "speakers.json"
        with open(speakers_path, 'w', encoding='utf-8') as f:
            json.dump(speaker_info, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Model trained and saved to {model_path}")
        if isinstance(history, dict) and 'val_accuracy' in history:
            final_acc = history['val_accuracy'][-1]
            self.logger.info(f"Final validation accuracy: {final_acc:.2%}")
            print(f"\n==============================\nФинальная точность модели на валидации: {final_acc:.2%}\n==============================\n")
            # Сохраняем точность в файл
            try:
                with open(self.model_dir / "last_accuracy.txt", "w", encoding="utf-8") as f:
                    f.write(str(final_acc))
            except Exception as e:
                self.logger.warning(f"Не удалось сохранить точность: {e}")
        else:
            self.logger.info("Обучение завершено. Валидационная точность не возвращается.")
        
        # После обучения модели пересоздаём эмбеддинги всех пользователей
        self.logger.info("Пересоздание эмбеддингов всех пользователей по новой модели...")
        self.recompute_all_user_embeddings()
        self.logger.info("Эмбеддинги пользователей успешно обновлены.")
        
        return True

    def test_folder(self, folder_path: str) -> Dict[str, Any]:
        """
        Test all audio files in a folder
        
        Args:
            folder_path: Path to folder with test files
            
        Returns:
            Test results
        """
        self.logger.info(f"Testing folder: {folder_path}")
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder does not exist: {folder_path}")
        
        # Поиск аудиофайлов
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a"}
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(folder.glob(f"*{ext}"))
            audio_files.extend(folder.glob(f"**/*{ext}"))
        
        if not audio_files:
            self.logger.warning("No audio files found")
            return {"total": 0, "results": []}
        
        # Тестирование файлов
        results = []
        correct = 0
        unknown = 0
        
        for audio_file in audio_files:
            try:
                result = self.identify_voice(str(audio_file))
                results.append(result)
                
                if result["success"]:
                    correct += 1
                else:
                    unknown += 1
                    
            except Exception as e:
                self.logger.error(f"Error processing {audio_file}: {e}")
                results.append({
                    "success": False,
                    "error": str(e),
                    "audio_file": str(audio_file)
                })
        
        # Подсчет статистики
        total = len(results)
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        summary = {
            "total": total,
            "correct": correct,
            "unknown": unknown,
            "errors": total - correct - unknown,
            "accuracy": accuracy,
            "results": results
        }
        
        self.logger.info(f"Test results:")
        self.logger.info(f"   Total files: {total}")
        self.logger.info(f"   Correct: {correct} ({accuracy:.1f}%)")
        self.logger.info(f"   Unknown: {unknown}")
        
        return summary

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.user_database.get_statistics()
        
        # Добавление информации о модели
        stats.update({
            "model_trained": self.model.is_trained,
            "model_path": str(self.model_dir / "speaker_model.pth"),
            "system_version": "1.0.0"
        })
        
        return stats

    def add_voice_samples(self, user_id: str, audio_files: List[str]):
        """
        Add new voice samples for an existing user
        
        Args:
            user_id: User ID
            audio_files: List of new audio files
        """
        if not self.model.is_trained:
            raise ValueError("Model is not trained!")
        
        user_info = self.user_database.get_user(user_id)
        if not user_info:
            raise ValueError(f"User {user_id} not found")
        
        self.logger.info(f"Adding samples for user: {user_info['name']}")
        
        # Обработка новых файлов
        new_embeddings = []
        for audio_file in audio_files:
            try:
                features = self.audio_processor.extract_features(audio_file)
                embedding = self.model.extract_embedding(features)
                new_embeddings.append(embedding)
            except Exception as e:
                self.logger.warning(f"Error processing {audio_file}: {e}")
        
        if new_embeddings:
            # Обновление эмбеддинга пользователя
            final_embedding = np.mean(new_embeddings, axis=0)
            self.user_database.update_user_embedding(user_id, final_embedding)
            self.logger.info(f"Added {len(new_embeddings)} new samples")
        else:
            self.logger.warning("No new files could be processed")

    def _analyze_voice_characteristics(self, audio_files: List[str]) -> Dict[str, Any]:
        """Analyze voice characteristics"""
        try:
            characteristics = self.audio_processor.analyze_voice_characteristics(audio_files)
            return characteristics
        except Exception as e:
            self.logger.warning(f"Voice characteristics analysis error: {e}")
            return {}
    
    def run_demo(self):
        """Запуск демонстрации системы"""
        self.logger.info("=== DEMONSTRATION OF THE VOICE RECOGNITION SYSTEM ===")
        
        # Проверка готовности системы
        if not self.model.is_trained:
            self.logger.info("Model is not trained. Starting quick training...")
            from src.utils.demo_generator import DemoGenerator
            demo_gen = DemoGenerator()
            demo_gen.create_demo_dataset()
            self.train_model(epochs=5, max_samples=1000)
        
        # Демонстрация функций
        stats = self.get_statistics()
        self.print_statistics(stats)
        
        # Тест идентификации если есть тестовые данные
        test_folder = Path("test_data")
        if test_folder.exists():
            self.logger.info("Testing on demo data...")
            results = self.test_folder(str(test_folder))
            self.print_test_results(results)
        
        self.logger.info("Demonstration completed")
    
    def print_identification_result(self, result: Dict[str, Any]):
        """Красивый вывод результата идентификации"""
        print("\n" + "="*50)
        print("IDENTIFICATION RESULT")
        print("="*50)
        print(f"File: {result.get('audio_file', 'N/A')}")
        
        if result.get("success"):
            print(f"Speaker: {result['name']}")
            print(f"ID: {result['user_id']}")
            print(f"Confidence: {result['confidence']:.2%}")
        else:
            print("Speaker not identified")
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Max confidence: {result.get('confidence', 0):.2%}")
        
        print(f"Processing time: {result.get('processing_time_ms', 0):.0f}ms")
        print("="*50)
    
    def print_test_results(self, results: Dict[str, Any]):
        """Красивый вывод результатов тестирования"""
        print("\n" + "="*50)
        print("TESTING RESULTS")
        print("="*50)
        print(f"Total files: {results['total']}")
        print(f"Correctly identified: {results['correct']}")
        print(f"Unknown: {results['unknown']}")
        print(f"Errors: {results['errors']}")
        print(f"Accuracy: {results['accuracy']:.1f}%")
        print("="*50)
    
    def print_statistics(self, stats: Dict[str, Any]):
        """Красивый вывод статистики системы"""
        print("\n" + "="*50)
        print("SYSTEM STATISTICS")
        print("="*50)
        print(f"Registered users: {stats.get('total_users', 0)}")
        print(f"Total voice samples: {stats.get('total_samples', 0)}")
        print(f"Model trained: {'Yes' if stats.get('model_trained') else 'No'}")
        print(f"Last updated: {stats.get('last_updated', 'N/A')}")
        print(f"System version: {stats.get('system_version', '1.0.0')}")
        print("="*50)
    
    def analyze_model_performance(self):
        """Анализ производительности модели"""
        self.logger.info("Analyzing model performance...")
        
        if not self.model.is_trained:
            self.logger.error("Model is not trained!")
            return
        
        # Здесь можно добавить детальный анализ модели
        stats = self.get_statistics()
        self.print_statistics(stats)
        
        # Анализ времени обработки
        test_audio = Path("test_data")
        if test_audio.exists():
            audio_files = list(test_audio.glob("*.wav"))[:10]  # Первые 10 файлов
            times = []
            
            for audio_file in audio_files:
                start = time.time()
                try:
                    self.identify_voice(str(audio_file))
                    times.append((time.time() - start) * 1000)
                except:
                    pass
            
            if times:
                avg_time = np.mean(times)
                self.logger.info(f"Average identification time: {avg_time:.0f}ms")
    
    def export_results_for_thesis(self):
        """Экспорт результатов для курсовой работы"""
        self.logger.info("Exporting results for thesis...")
        
        # Создание отчета
        report = {
            "system_info": self.get_statistics(),
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": {}
        }
        
        # Тестирование если возможно
        test_folder = Path("test_data")
        if test_folder.exists():
            test_results = self.test_folder(str(test_folder))
            report["test_results"] = test_results
        
        # Сохранение отчета
        report_path = Path("thesis_report.json")
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Report saved: {report_path}")
        
        # Создание краткого текстового отчета
        text_report_path = Path("thesis_summary.txt")
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("REPORT ON THE VOICE RECOGNITION SYSTEM\n")
            f.write("="*50 + "\n\n")
            
            stats = report["system_info"]
            f.write(f"Users in the system: {stats.get('total_users', 0)}\n")
            f.write(f"Voice samples: {stats.get('total_samples', 0)}\n")
            f.write(f"Model trained: {stats.get('model_trained')}\n\n")
            
            if "test_results" in report:
                test = report["test_results"]
                f.write("TESTING RESULTS:\n")
                f.write(f"Overall accuracy: {test.get('accuracy', 0):.1f}%\n")
                f.write(f"Files tested: {test.get('total', 0)}\n")
        
        self.logger.info(f"Summary report saved: {text_report_path}")
    
    def _predict_with_trained_model(self, features: np.ndarray) -> Dict[str, Any]:
        """
        Предсказание говорящего через обученную модель
        
        Args:
            features: Аудиопризнаки
            
        Returns:
            Результат предсказания
        """
        # Получение предсказания от модели
        embedding_net = self.model.embedding_net
        classifier = self.model.classifier
        scaler = self.model.scaler
        device = self.model.device
        
        # Нормализация признаков
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Получение предсказания
        embedding_net.eval()
        classifier.eval()
        
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features_scaled).to(device)
            embedding = embedding_net(features_tensor)
            logits = classifier(embedding)
            probabilities = torch.softmax(logits, dim=1)
            
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()
        
        # Загрузка информации о говорящих
        speakers_info = self._load_speakers_info()
        
        if str(predicted_class) in speakers_info:
            speaker_name = speakers_info[str(predicted_class)]["original_id"]
        else:
            speaker_name = f"Speaker {predicted_class + 1}"
        
        return {
            "speaker_id": predicted_class,
            "speaker_name": speaker_name,
            "confidence": confidence
        }
    
    def _load_speakers_info(self) -> Dict[str, Any]:
        """Загрузка информации о говорящих из файла"""
        speakers_file = Path("data/models/speakers.json")
        if speakers_file.exists():
            try:
                with open(speakers_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Error loading speakers info: {e}")
        return {}
    
    def recompute_all_user_embeddings(self):
        """
        Пересоздаёт эмбеддинги всех пользователей с помощью актуальной модели
        """
        if not self.model.is_trained:
            self.logger.warning("Модель не обучена, эмбеддинги не будут обновлены.")
            return
        users = self.user_database.get_all_users()
        for user_id, user_data in users.items():
            # Попытка найти все исходные аудиофайлы пользователя
            meta = self.user_database.get_user_metadata(user_id)
            audio_files = meta.get('processed_files', []) if meta else []
            if not audio_files:
                self.logger.warning(f"Нет аудиофайлов для пользователя {user_data['name']} ({user_id}) — эмбеддинг не обновлён.")
                continue
            embeddings = []
            for audio_file in audio_files:
                try:
                    features = self.audio_processor.extract_features(audio_file)
                    embedding = self.model.extract_embedding(features)
                    embeddings.append(embedding)
                except Exception as e:
                    self.logger.warning(f"Ошибка обработки {audio_file} для {user_data['name']}: {e}")
            if embeddings:
                final_embedding = np.mean(embeddings, axis=0)
                self.user_database.update_user_embedding(user_id, final_embedding)
                self.logger.info(f"Эмбеддинг пользователя {user_data['name']} ({user_id}) обновлён.")
            else:
                self.logger.warning(f"Не удалось создать эмбеддинг для пользователя {user_data['name']} ({user_id})")
