"""
Утилиты для работы с файлами и папками
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import pickle
import logging


def ensure_dir_exists(directory: Union[str, Path]) -> Path:
    """
    Создание папки если она не существует
    
    Args:
        directory: Путь к папке
        
    Returns:
        Path объект папки
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_file_hash(file_path: Union[str, Path], algorithm: str = 'md5') -> str:
    """
    Получение хеша файла
    
    Args:
        file_path: Путь к файлу
        algorithm: Алгоритм хеширования (md5, sha1, sha256)
        
    Returns:
        Хеш файла в hex формате
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def find_files_by_extension(directory: Union[str, Path], 
                          extensions: List[str],
                          recursive: bool = True) -> List[Path]:
    """
    Поиск файлов по расширениям
    
    Args:
        directory: Папка для поиска
        extensions: Список расширений (с точкой, например ['.wav', '.mp3'])
        recursive: Рекурсивный поиск
        
    Returns:
        Список найденных файлов
    """
    directory = Path(directory)
    files = []
    
    if recursive:
        for ext in extensions:
            files.extend(directory.glob(f"**/*{ext}"))
    else:
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
    
    return files


def get_directory_size(directory: Union[str, Path]) -> int:
    """
    Получение размера папки в байтах
    
    Args:
        directory: Путь к папке
        
    Returns:
        Размер в байтах
    """
    total_size = 0
    directory = Path(directory)
    
    for file_path in directory.rglob('*'):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def format_file_size(size_bytes: int) -> str:
    """
    Форматирование размера файла в человекочитаемый вид
    
    Args:
        size_bytes: Размер в байтах
        
    Returns:
        Отформатированная строка (например, "1.5 MB")
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.1f} {units[unit_index]}"


def safe_copy_file(src: Union[str, Path], dst: Union[str, Path], 
                  backup: bool = True) -> bool:
    """
    Безопасное копирование файла с возможностью создания резервной копии
    
    Args:
        src: Исходный файл
        dst: Целевой файл
        backup: Создать резервную копию если целевой файл существует
        
    Returns:
        True если копирование успешно
    """
    try:
        src = Path(src)
        dst = Path(dst)
        
        # Создание папки назначения
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Резервная копия
        if backup and dst.exists():
            backup_path = dst.with_suffix(f"{dst.suffix}.backup")
            shutil.copy2(dst, backup_path)
        
        # Копирование
        shutil.copy2(src, dst)
        return True
        
    except Exception as e:
        logging.error(f"Ошибка копирования {src} -> {dst}: {e}")
        return False


def save_json(data: Dict[str, Any], file_path: Union[str, Path], 
              indent: int = 2, ensure_ascii: bool = False) -> bool:
    """
    Сохранение данных в JSON файл
    
    Args:
        data: Данные для сохранения
        file_path: Путь к файлу
        indent: Отступы для форматирования
        ensure_ascii: Экранировать не-ASCII символы
        
    Returns:
        True если сохранение успешно
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
        
        return True
        
    except Exception as e:
        logging.error(f"Ошибка сохранения JSON {file_path}: {e}")
        return False


def load_json(file_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """
    Загрузка данных из JSON файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Загруженные данные или None при ошибке
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки JSON {file_path}: {e}")
        return None


def save_pickle(data: Any, file_path: Union[str, Path]) -> bool:
    """
    Сохранение данных в pickle файл
    
    Args:
        data: Данные для сохранения
        file_path: Путь к файлу
        
    Returns:
        True если сохранение успешно
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        
        return True
        
    except Exception as e:
        logging.error(f"Ошибка сохранения pickle {file_path}: {e}")
        return False


def load_pickle(file_path: Union[str, Path]) -> Optional[Any]:
    """
    Загрузка данных из pickle файла
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Загруженные данные или None при ошибке
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Ошибка загрузки pickle {file_path}: {e}")
        return None


def clean_directory(directory: Union[str, Path], 
                   pattern: str = "*",
                   recursive: bool = False,
                   dry_run: bool = False) -> List[Path]:
    """
    Очистка папки по шаблону
    
    Args:
        directory: Папка для очистки
        pattern: Шаблон файлов для удаления
        recursive: Рекурсивная очистка
        dry_run: Только показать что будет удалено
        
    Returns:
        Список удаленных (или помеченных к удалению) файлов
    """
    directory = Path(directory)
    if not directory.exists():
        return []
    
    removed_files = []
    
    if recursive:
        files_to_remove = directory.rglob(pattern)
    else:
        files_to_remove = directory.glob(pattern)
    
    for file_path in files_to_remove:
        if file_path.is_file():
            if not dry_run:
                try:
                    file_path.unlink()
                    removed_files.append(file_path)
                except Exception as e:
                    logging.error(f"Не удалось удалить {file_path}: {e}")
            else:
                removed_files.append(file_path)
    
    return removed_files


def create_symlink(src: Union[str, Path], dst: Union[str, Path]) -> bool:
    """
    Создание символической ссылки
    
    Args:
        src: Исходный файл/папка
        dst: Путь к ссылке
        
    Returns:
        True если создание успешно
    """
    try:
        src = Path(src).resolve()
        dst = Path(dst)
        
        # Создание папки для ссылки
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        # Удаление существующей ссылки
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        
        dst.symlink_to(src)
        return True
        
    except Exception as e:
        logging.error(f"Ошибка создания символической ссылки {src} -> {dst}: {e}")
        return False


def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Получение подробной информации о файле
    
    Args:
        file_path: Путь к файлу
        
    Returns:
        Словарь с информацией о файле
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        return {"error": "Файл не существует"}
    
    stat = file_path.stat()
    
    info = {
        "name": file_path.name,
        "path": str(file_path.absolute()),
        "size": stat.st_size,
        "size_formatted": format_file_size(stat.st_size),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "accessed": stat.st_atime,
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "is_symlink": file_path.is_symlink(),
        "suffix": file_path.suffix,
        "stem": file_path.stem
    }
    
    # Хеш для небольших файлов (< 100MB)
    if info["is_file"] and info["size"] < 100 * 1024 * 1024:
        try:
            info["md5"] = get_file_hash(file_path, 'md5')
        except:
            info["md5"] = None
    
    return info


def archive_directory(directory: Union[str, Path], 
                     output_file: Union[str, Path],
                     format: str = 'zip') -> bool:
    """
    Архивирование папки
    
    Args:
        directory: Папка для архивирования
        output_file: Путь к архиву
        format: Формат архива ('zip', 'tar', 'gztar', 'bztar', 'xztar')
        
    Returns:
        True если архивирование успешно
    """
    try:
        directory = Path(directory)
        output_file = Path(output_file)
        
        # Создание папки для архива
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Удаление расширения из пути (shutil.make_archive добавит его сам)
        base_name = str(output_file.with_suffix(''))
        
        shutil.make_archive(base_name, format, directory)
        return True
        
    except Exception as e:
        logging.error(f"Ошибка архивирования {directory}: {e}")
        return False


def extract_archive(archive_path: Union[str, Path], 
                   extract_to: Union[str, Path]) -> bool:
    """
    Извлечение архива
    
    Args:
        archive_path: Путь к архиву
        extract_to: Папка для извлечения
        
    Returns:
        True если извлечение успешно
    """
    try:
        archive_path = Path(archive_path)
        extract_to = Path(extract_to)
        
        # Создание папки для извлечения
        extract_to.mkdir(parents=True, exist_ok=True)
        
        shutil.unpack_archive(archive_path, extract_to)
        return True
        
    except Exception as e:
        logging.error(f"Ошибка извлечения {archive_path}: {e}")
        return False


class FileMonitor:
    """
    Простой монитор изменений файлов
    """
    
    def __init__(self):
        self.file_states = {}
    
    def add_file(self, file_path: Union[str, Path]):
        """Добавление файла для мониторинга"""
        file_path = Path(file_path)
        if file_path.exists():
            self.file_states[str(file_path)] = file_path.stat().st_mtime
    
    def check_changes(self) -> List[str]:
        """
        Проверка изменений
        
        Returns:
            Список измененных файлов
        """
        changed_files = []
        
        for file_path_str, old_mtime in self.file_states.items():
            file_path = Path(file_path_str)
            
            if file_path.exists():
                current_mtime = file_path.stat().st_mtime
                if current_mtime != old_mtime:
                    changed_files.append(file_path_str)
                    self.file_states[file_path_str] = current_mtime
            else:
                # Файл был удален
                changed_files.append(file_path_str)
                del self.file_states[file_path_str]
        
        return changed_files
    
    def remove_file(self, file_path: Union[str, Path]):
        """Удаление файла из мониторинга"""
        file_path_str = str(Path(file_path))
        self.file_states.pop(file_path_str, None)
