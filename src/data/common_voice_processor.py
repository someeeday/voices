import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Union, Optional
from tqdm import tqdm
import shutil
import librosa
import soundfile as sf


def process_audio_with_noise_reduction(input_path: Path, output_path: Path, 
                                     target_sr: int = 16000) -> bool:
    """
    Обрабатывает аудиофайл: шумоподавление, нормализация
    """
    try:
        # Загружаем аудио
        audio, sr = librosa.load(input_path, sr=target_sr)
        
        # Простое шумоподавление - убираем тишину в начале и конце
        trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Нормализация громкости
        if len(trimmed_audio) > 0:
            normalized_audio = librosa.util.normalize(trimmed_audio)
            
            # Сохраняем обработанный файл
            sf.write(output_path, normalized_audio, target_sr)
            return True
            
    except Exception as e:
        logging.error(f"Error processing audio {input_path}: {e}")
    return False


def load_or_create_cache(cache_file: Path) -> Dict:
    """Загружает кеш или создает новый"""
    if cache_file.exists():
        with open(cache_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"processed_users": [], "current_step": 0, "total_steps": 0}


def save_cache(cache_file: Path, cache_data: Dict):
    """Сохраняет кеш"""
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2, ensure_ascii=False)


def process_common_voice_complete(source_dir: Path, output_dir: Path, max_users: Optional[int] = None,
                                max_files_per_user: int = 50, audio_duration_min: float = 1.0, 
                                audio_duration_max: float = 10.0) -> bool:
    """
    Полная обработка датасета Common Voice
    """
    logger = logging.getLogger(__name__)
    
    tsv_file = source_dir / "train.tsv"
    clips_dir = source_dir / "clips"
    
    if not tsv_file.exists():
        logger.error(f"TSV file not found: {tsv_file}")
        return False
    
    if not clips_dir.exists():
        logger.error(f"Clips directory not found: {clips_dir}")
        return False
    
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Читаем TSV файл
    try:
        df = pd.read_csv(tsv_file, sep='\t')
        logger.info(f"Loaded {len(df)} entries from Common Voice TSV")
    except Exception as e:
        logger.error(f"Error reading TSV file: {e}")
        return False
    
    # Группируем по client_id
    user_groups = df.groupby('client_id')
    logger.info(f"Found {len(user_groups)} unique users")
    
    # Ограничиваем количество пользователей если нужно
    if max_users and len(user_groups) > max_users:
        selected_groups = list(user_groups)[:max_users]
        user_groups = dict(selected_groups)
        logger.info(f"Limited to {max_users} users")
    
    # Настройка кеша
    cache_file = output_dir / "processing_cache.json"
    cache_data = load_or_create_cache(cache_file)
    
    processed_count = 0
    skipped_users = 0
    
    # Основной прогресс-бар
    with tqdm(total=len(user_groups), desc="🎵 Processing Common Voice users", unit="user") as global_progress:
        for user_id, user_data in user_groups:
            if user_id in cache_data["processed_users"]:
                skipped_users += 1
                global_progress.update(1)
                continue
            
            user_dir = output_dir / str(user_id)
            user_dir.mkdir(exist_ok=True)
            
            user_processed = 0
            audio_files = user_data['path'].tolist()[:max_files_per_user]
            
            # Прогресс-бар для файлов пользователя
            for audio_file in tqdm(audio_files, desc=f"Files for {user_id}", leave=False):
                input_path = clips_dir / audio_file
                output_path = user_dir / audio_file.replace('.mp3', '.wav')
                
                if output_path.exists():
                    continue
                
                if input_path.exists():
                    # Проверяем длительность аудио
                    try:
                        duration = librosa.get_duration(path=input_path)
                        if audio_duration_min <= duration <= audio_duration_max:
                            if process_audio_with_noise_reduction(input_path, output_path):
                                user_processed += 1
                        else:
                            logger.debug(f"Skipping {audio_file}: duration {duration:.2f}s not in range")
                    except Exception as e:
                        logger.debug(f"Error processing {audio_file}: {e}")
                
                if user_processed >= max_files_per_user:
                    break
            
            if user_processed > 0:
                cache_data["processed_users"].append(user_id)
                processed_count += 1
                save_cache(cache_file, cache_data)
            
            global_progress.update(1)
            global_progress.set_postfix({"Stage": "Common Voice", "User": user_id, "Files": user_processed})
        logger.debug(f"[COMMON_VOICE] Processed {user_processed} files for {user_id}")
    # Удаляем кеш после завершения
    if cache_file.exists():
        cache_file.unlink()
    logger.info(f"[COMMON_VOICE] Complete! Processed {len(cache_data['processed_users'])} users, skipped {skipped_users} existing")
    return True


def extract_features_from_processed(processed_dir: Path, features_path: Path, speaker_info_path: Path):
    """
    Intel Ultra 9 optimized feature extraction from processed audio files
    """
    # Import AudioProcessor locally to avoid circular imports
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.audio.audio_processor import AudioProcessor
    
    print("🚀 Intel Ultra 9 feature extraction starting...")
    
    # Import modules for maximum parallelization
    import concurrent.futures
    import multiprocessing
    try:
        import psutil
        HAS_PSUTIL = True
    except ImportError:
        HAS_PSUTIL = False
        print("⚠️  psutil not found, using conservative settings")
    
    # Intel Ultra 9 system configuration
    cpu_count = multiprocessing.cpu_count()
    
    if HAS_PSUTIL:
        physical_cores = psutil.cpu_count(logical=False) or cpu_count // 2
        logical_cores = psutil.cpu_count(logical=True) or cpu_count
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        
        print(f"💻 Intel Ultra 9 detected:")
        print(f"  • Physical cores: {physical_cores}")
        print(f"  • Logical cores: {logical_cores}")
        print(f"  • Available RAM: {available_memory:.1f} GB")
        
        # Ultra aggressive configuration for Intel Ultra 9
        max_workers = min(logical_cores - 1, 20)
        batch_size = min(max(int(available_memory * 100), 500), 3000)
    else:
        max_workers = min(cpu_count - 1, 12)
        batch_size = 1000
        print(f"💻 CPU cores detected: {cpu_count}")
    
    print(f"🔥 Intel Ultra 9 config: {max_workers} workers, {batch_size} batch size")
    
    audio_processor = AudioProcessor()
    features = []
    labels = []
    speaker_info = {}
    
    user_dirs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
    print(f"📁 Found {len(user_dirs)} users")
    
    # Collect all files and labels for batch processing
    all_files = []
    all_labels = []
    
    for label, user_dir in enumerate(user_dirs):
        speaker_info[label] = user_dir.name
        wav_files = list(user_dir.glob('*.wav'))
        for wav_file in wav_files:
            all_files.append(str(wav_file))
            all_labels.append(label)
    
    print(f"🎵 Total audio files: {len(all_files)}")
    
    def extract_single_feature(file_path):
        """Extract features from single file"""
        try:
            return audio_processor.extract_features(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None
    
    # Intel Ultra 9 ultra-fast batch processing
    print(f"🚀 Using {max_workers} workers for parallel processing")
    
    features_list = []
    labels_list = []
    
    # Process in ultra-large batches for Intel Ultra 9
    total_batches = (len(all_files) + batch_size - 1) // batch_size
    
    with tqdm(total=len(all_files), desc="🧠 Intel Ultra 9 feature extraction", 
              unit="file", dynamic_ncols=True) as pbar:
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(all_files))
            
            batch_files = all_files[start_idx:end_idx]
            batch_labels = all_labels[start_idx:end_idx]
            
            # Ultra-fast parallel processing with Intel Ultra 9
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_features = list(executor.map(extract_single_feature, batch_files))
            
            # Filter successful extractions
            for feature, label in zip(batch_features, batch_labels):
                if feature is not None:
                    features_list.append(feature)
                    labels_list.append(label)
            
            processed = end_idx - start_idx
            pbar.update(processed)
            pbar.set_postfix({
                "Batch": f"{batch_idx+1}/{total_batches}",
                "Processed": len(features_list),
                "Speed": f"{processed/(pbar.format_dict.get('elapsed', 1)):.1f}file/s"
            })
    
    if features_list and labels_list:
        print(f"💾 Saving {len(features_list)} feature vectors...")
        
        # Convert to numpy arrays with Intel Ultra 9 optimization
        features_array = np.array(features_list, dtype=np.float32)
        labels_array = np.array(labels_list, dtype=np.int32)
        
        # Save with progress tracking
        with tqdm(total=2, desc="💾 Saving data", unit="file") as save_pbar:
            # Save features and labels
            np.savez_compressed(features_path, features=features_array, labels=labels_array)
            save_pbar.update(1)
            save_pbar.set_postfix({"File": "training_data.npz"})
            
            # Save speaker info
            with open(speaker_info_path, 'w', encoding='utf-8') as f:
                json.dump(speaker_info, f, ensure_ascii=False, indent=2)
            save_pbar.update(1)
            save_pbar.set_postfix({"File": "speaker_info.json"})
        
        print(f"🎉 Intel Ultra 9 processing completed!")
        print(f"  • Samples: {features_array.shape[0]:,}")
        print(f"  • Speakers: {len(speaker_info):,}")
        print(f"  • Feature dimension: {features_array.shape[1]}")
        print(f"  • Data size: {features_array.nbytes / 1024 / 1024:.1f} MB")
        return True
    else:
        print("❌ Failed to extract features!")
        return False
