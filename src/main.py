"""
Основная точка входа в систему распознавания голосов
"""

import argparse
import logging
import sys
import hashlib
import os
from pathlib import Path

# Добавляем корневую папку в Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.simple_system import SimpleSpeakerSystem
from src.utils.logger_config import setup_logger
from src.data.dataset_manager import DatasetManager


def setup_argument_parser():
    """Настройка парсера аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="🎙️ Система Распознавания Голосов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python src/main.py --train                   # Обучение модели
  python src/main.py --identify voice.wav      # Идентификация голоса
        """
    )
    
    # Основные команды
    parser.add_argument('--train', action='store_true',
                      help='Обучить модель на данных (обучение только если модель неактуальна)')
    parser.add_argument('--identify', type=str, metavar='FILE',
                      help='Идентифициировать голос в аудиофайле')
    parser.add_argument('--force-train', action='store_true',
                      help='Принудительно переобучить модель, даже если она уже есть')
    # --learn удалён
    
    # Дополнительные опции
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='Подробный вывод')
    
    return parser

def get_users_hash(users_path):
    try:
        with open(users_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception:
        return None

def get_dir_hash(dir_path):
    hash_md5 = hashlib.md5()
    for root, dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            fpath = os.path.join(root, fname)
            try:
                stat = os.stat(fpath)
                hash_md5.update(fname.encode())
                hash_md5.update(str(stat.st_mtime).encode())
            except Exception:
                continue
    return hash_md5.hexdigest()

def main():
    """Основная функция программы"""
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Настройка логирования
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(level=log_level)
    
    try:
        # Инициализация системы
        system = SimpleSpeakerSystem()
        
        if args.train or args.force_train:
            logger.info("🎓 Запуск регистрации пользователей из test_data...")
            test_data_path = Path("test_data")
            users_path = Path("data/user_database/users.json")
            users_hash = get_users_hash(users_path) if users_path.exists() else None
            test_data_hash = get_dir_hash(str(test_data_path)) if test_data_path.exists() else None
            hash_file = Path("data/models/last_train_hash.txt")
            prev_hash = None
            if hash_file.exists():
                with open(hash_file, "r", encoding="utf-8") as f:
                    prev_hash = f.read().strip()
            current_hash = f"{users_hash}_{test_data_hash}"
            if not args.force_train and prev_hash == current_hash:
                logger.info("Нет изменений в пользователях и test_data, обучение не требуется.")
                acc_file = Path("data/models/last_accuracy.txt")
                if acc_file.exists():
                    with open(acc_file, "r", encoding="utf-8") as f:
                        acc = float(f.read().strip())
                    print(f"\n==============================\nФинальная точность модели на валидации: {acc:.2%}\n==============================\n")
                return 0
            # Получаем список уже зарегистрированных пользователей (имена в нижнем регистре и без пробелов)
            existing_users = set()
            if test_data_path.exists():
                existing_users = set(u['name'].strip().lower() for u in system.user_database.get_all_users_list())
                for user_dir in test_data_path.iterdir():
                    if user_dir.is_dir():
                        user_name = user_dir.name.strip().lower()
                        if user_name in existing_users:
                            logger.info(f"Пользователь {user_dir.name} уже зарегистрирован, пропускаю.")
                            continue
                        audio_files = list(user_dir.glob("*.wav"))
                        if audio_files:
                            logger.info(f"👤 Регистрирую пользователя: {user_dir.name} ({len(audio_files)} файлов)")
                            try:
                                system.register_user(user_dir.name, [str(f) for f in audio_files])
                            except Exception as e:
                                logger.warning(f"Ошибка регистрации пользователя {user_dir.name}: {e}")
            else:
                logger.info("Папка test_data не найдена, пользователи не добавлены.")
            # После регистрации пользователей всегда готовим датасет и обучаем модель
            logger.info("📊 Подготовка данных...")
            dataset_manager = DatasetManager()
            success = dataset_manager.prepare_all_datasets()
            if not success:
                logger.error("❌ Ошибка подготовки данных")
                return 1
            logger.info("🎓 Запуск обучения модели...")
            success = system.train_model()
            if success:
                logger.info("✅ Модель успешно обучена!")
            else:
                logger.error("❌ Ошибка обучения модели")
                return 1
            # После обучения сохраняем новый хэш
            with open(hash_file, "w", encoding="utf-8") as f:
                f.write(current_hash)
        
        elif args.identify:
            audio_path = Path(args.identify)
            if not audio_path.exists():
                logger.error(f"❌ Файл или папка не найдены: {audio_path}")
                return 1
            if audio_path.is_dir():
                logger.info(f"🔍 Идентификация всех файлов в папке: {audio_path}")
                audio_files = list(audio_path.glob("*.wav"))
                if not audio_files:
                    logger.info("В папке нет .wav файлов для идентификации.")
                    return 0
                for file in audio_files:
                    result = system.identify_voice(str(file))
                    if result:
                        confidence = result.get('similarity', result.get('confidence', 0))
                        logger.info(f"  {file.name}: {result['name']} (уверенность: {confidence:.2%})")
                    else:
                        logger.info(f"  {file.name}: Говорящий не найден в базе данных")
            else:
                logger.info(f"🔍 Идентификация голоса в файле: {audio_path}")
                result = system.identify_voice(str(audio_path))
                if result:
                    confidence = result.get('similarity', result.get('confidence', 0))
                    logger.info(f"🎯 Результат: {result['name']} (уверенность: {confidence:.2%})")
                else:
                    logger.info("❓ Говорящий не найден в базе данных")
        
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        logger.info("🛑 Операция прервана пользователем")
        return 1
    except Exception as e:
        logger.error(f"❌ Ошибка: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
