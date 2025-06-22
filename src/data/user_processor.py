import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import logging


class UserProcessor:
    """Простая база данных пользователей с файловой системой"""
    
    def __init__(self, database_dir: str):
        self.database_dir = Path(database_dir)
        self.users_file = self.database_dir / "users.json"
        self.embeddings_dir = self.database_dir / "embeddings"
        self.metadata_dir = self.database_dir / "metadata"
        
        self.logger = logging.getLogger(__name__)
        
        # Создаем директории если не существуют
        self.database_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем или создаем файл пользователей
        self._load_users()
    
    def _load_users(self):
        """Загружает файл пользователей или создает пустой"""
        if self.users_file.exists():
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Ошибка загрузки users.json: {e}. Создаю новый файл.")
                self.users = {}
        else:
            self.users = {}
    
    def _save_users(self):
        """Сохраняет файл пользователей"""
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(self.users, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения users.json: {e}")
    
    def create_user(self, name: str) -> str:
        """Создает нового пользователя"""
        user_id = str(uuid.uuid4())
        self.users[user_id] = {
            'name': name,
            'created_at': str(np.datetime64('now')),
            'embedding_file': f"{user_id}.npy"
        }
        self._save_users()
        
        # Создаем файл метаданных
        metadata = {
            'name': name,
            'user_id': user_id,
            'created_at': str(np.datetime64('now')),
            'processed_files': [],
            'last_updated': str(np.datetime64('now'))
        }
        self.update_user_metadata(user_id, metadata)
        
        self.logger.info(f"Создан пользователь {name} с ID {user_id}")
        return user_id
    
    def search_users(self, name: str) -> List[Dict[str, Any]]:
        """Ищет пользователей по имени"""
        results = []
        for user_id, user_data in self.users.items():
            if user_data['name'].lower() == name.lower():
                results.append({
                    'uuid': user_id,
                    'name': user_data['name'],
                    'created_at': user_data['created_at']
                })
        return results
    
    def get_user_metadata(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получает метаданные пользователя"""
        metadata_file = self.metadata_dir / f"{user_id}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Ошибка чтения метаданных {user_id}: {e}")
                return None
        return None
    
    def update_user_metadata(self, user_id: str, metadata: Dict[str, Any]):
        """Обновляет метаданные пользователя"""
        metadata_file = self.metadata_dir / f"{user_id}.json"
        try:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения метаданных {user_id}: {e}")
    
    def save_user_embedding(self, user_id: str, embedding: np.ndarray):
        """Сохраняет эмбеддинг пользователя"""
        embedding_file = self.embeddings_dir / f"{user_id}.npy"
        try:
            np.save(embedding_file, embedding)
        except Exception as e:
            self.logger.error(f"Ошибка сохранения эмбеддинга {user_id}: {e}")
    
    def load_user_embedding(self, user_id: str) -> Optional[np.ndarray]:
        """Загружает эмбеддинг пользователя"""
        embedding_file = self.embeddings_dir / f"{user_id}.npy"
        if embedding_file.exists():
            try:
                return np.load(embedding_file)
            except Exception as e:
                self.logger.error(f"Ошибка загрузки эмбеддинга {user_id}: {e}")
                return None
        return None
    
    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        """Возвращает всех пользователей"""
        return self.users.copy()
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Загружает все эмбеддинги пользователей"""
        embeddings = {}
        for user_id in self.users.keys():
            embedding = self.load_user_embedding(user_id)
            if embedding is not None:
                embeddings[user_id] = embedding
        return embeddings
    
    def get_all_users_list(self) -> List[Dict[str, Any]]:
        """Возвращает всех пользователей в виде списка с нужными полями"""
        users_list = []
        for user_id, user_data in self.users.items():
            users_list.append({
                'uuid': user_id,
                'name': user_data['name'],
                'created_at': user_data.get('created_at', '')
            })
        return users_list
    
    def add_user(self, name: str, embedding: Optional[np.ndarray] = None) -> str:
        """Псевдоним для create_user для совместимости"""
        user_id = self.create_user(name)
        if embedding is not None:
            self.save_user_embedding(user_id, embedding)
        return user_id
    
    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получает информацию о пользователе"""
        if user_id in self.users:
            user_data = self.users[user_id].copy()
            user_data['uuid'] = user_id
            return user_data
        return None
    
    def update_user_embedding(self, user_id: str, embedding: np.ndarray):
        """Обновляет эмбеддинг пользователя"""
        self.save_user_embedding(user_id, embedding)
    
    def find_closest_user(self, test_embedding: np.ndarray, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """Находит ближайшего пользователя по эмбеддингу"""
        all_embeddings = self.get_all_embeddings()
        if not all_embeddings:
            return {'name': 'Не определен', 'similarity': 0.0}
        # Приведение к одномерному виду
        test_embedding = np.asarray(test_embedding).flatten()
        best_similarity = -1
        best_user_id = None
        for user_id, user_embedding in all_embeddings.items():
            user_embedding = np.asarray(user_embedding).flatten()
            # Косинусная схожесть
            similarity = np.dot(test_embedding, user_embedding) / (
                np.linalg.norm(test_embedding) * np.linalg.norm(user_embedding)
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_user_id = user_id
        best_candidate = None
        if best_user_id:
            user_info = self.get_user(best_user_id)
            if user_info:
                best_candidate = {
                    'user_id': best_user_id,
                    'name': user_info.get('name', 'Не определен'),
                    'similarity': float(best_similarity)
                }
        if best_similarity >= threshold and best_candidate:
            return best_candidate
        # Возвращаем best_candidate даже если не превышен порог
        return {'name': 'Не определен', 'similarity': float(best_similarity), 'best_candidate': best_candidate}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Возвращает статистику базы данных"""
        total_users = len(self.users)
        users_with_embeddings = len(self.get_all_embeddings())
        
        return {
            'total_users': total_users,
            'users_with_embeddings': users_with_embeddings,
            'database_dir': str(self.database_dir)
        }
    
    @property
    def users_data(self) -> Dict[str, Any]:
        """Свойство для совместимости с UserDatabase"""
        return {
            'users': self.get_all_users_list()
        }
