"""
Dataset Manager for preparing training data
Combines and manages different datasets for speaker recognition
"""

import json
import logging
import numpy as np
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
import concurrent.futures

from src.data.user_processor import UserProcessor
from src.audio.audio_processor import AudioProcessor
from src.utils.file_utils import ensure_dir_exists


class DatasetManager:
    """
    Manages dataset preparation and processing for training
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize dataset manager
        
        Args:
            data_dir: Root data directory
        """
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.datasets_dir = self.data_dir / "datasets"
        self.processed_dir = self.data_dir / "processed"
        self.cache_dir = self.data_dir / "cache"
        
        # Create directories
        ensure_dir_exists(self.datasets_dir)
        ensure_dir_exists(self.processed_dir)
        ensure_dir_exists(self.cache_dir)
        
        # Initialize processors
        self.audio_processor = AudioProcessor()
        # Initialize user processor if needed
        self.user_processor = None
        
        # Cache for processed files
        self.features_cache = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cached features from disk"""
        cache_file = self.cache_dir / "features_cache.pkl"
        try:
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    self.features_cache = pickle.load(f)
                self.logger.info(f"Loaded {len(self.features_cache)} cached features")
            else:
                self.features_cache = {}
        except Exception as e:
            self.logger.warning(f"Failed to load cache: {e}")
            self.features_cache = {}
    
    def save_cache(self):
        """Save cached features to disk"""
        cache_file = self.cache_dir / "features_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.features_cache, f)
            self.logger.debug(f"Saved {len(self.features_cache)} cached features")
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")
    
    def get_file_hash(self, file_path: Path) -> str:
        """Get hash of file for caching"""
        try:
            stat = file_path.stat()
            # Use file path, size and modification time for hash
            content = f"{file_path}_{stat.st_size}_{stat.st_mtime}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            return str(file_path)
    
    def extract_features_cached(self, audio_file: Path) -> Optional[np.ndarray]:
        """Extract features with caching"""
        file_hash = self.get_file_hash(audio_file)
        
        # Check cache first
        if file_hash in self.features_cache:
            return self.features_cache[file_hash]
        
        # Extract features
        try:
            features = self.audio_processor.extract_features(str(audio_file))
            if features is not None:
                self.features_cache[file_hash] = features
            return features
        except Exception as e:
            self.logger.warning(f"Failed to extract features from {audio_file}: {e}")
            return None

    def prepare_all_datasets(self, max_samples: int = 25000, max_speakers: int = 500, 
                           min_duration: float = 1.0, max_duration: float = 10.0,
                           samples_per_speaker: int = 10) -> bool:
        """
        Prepare all available datasets for training
        
        Args:
            max_samples: Maximum number of samples to process
            max_speakers: Maximum number of speakers
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting dataset preparation...")
        
        all_features = []
        all_labels = []
        speaker_info = {}
        current_speaker_id = 0
          # Process available datasets by checking directories
        dataset_types = ['ru', 'test_data', 'user_data']
        
        for dataset_name in dataset_types:
            try:
                self.logger.info(f"Processing {dataset_name}...")
                  # Check if dataset exists
                dataset_path = self.datasets_dir / dataset_name
                if not dataset_path.exists():
                    # Also check test_data in root
                    if dataset_name == 'test_data':
                        dataset_path = Path('test_data')
                    if not dataset_path.exists():
                        self.logger.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                        continue
                
                # Special handling for Common Voice (ru dataset)
                if dataset_name == 'ru':
                    self._process_common_voice_dataset(
                        dataset_path, all_features, all_labels, speaker_info, 
                        current_speaker_id, max_speakers, min_duration, max_duration, samples_per_speaker
                    )
                else:
                    # Process dataset using directory structure
                    self._process_directory_structure(
                        dataset_path, all_features, all_labels, speaker_info, current_speaker_id, max_speakers
                    )
                
                # Update current_speaker_id
                if speaker_info:
                    current_speaker_id = max(speaker_info.keys()) + 1
                
            except Exception as e:
                self.logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        # Check if we have any data
        if len(all_features) == 0:
            self.logger.error("No training data found! Please add datasets to data/datasets/")
            return False
        
        # Convert to numpy arrays
        all_features = np.array(all_features)
        all_labels = np.array(all_labels)
        
        # Limit samples if needed
        if len(all_features) > max_samples:
            indices = np.random.choice(len(all_features), max_samples, replace=False)
            all_features = all_features[indices]
            all_labels = all_labels[indices]
            self.logger.info(f"Limited to {max_samples} samples")
        
        # Save processed data
        output_path = self.processed_dir / "training_data.npz"
        np.savez_compressed(
            output_path,
            features=all_features,
            labels=all_labels,
            speaker_info=json.dumps(speaker_info)
        )
        
        self.logger.info(f"Saved {len(all_features)} samples, {len(speaker_info)} speakers to {output_path}")
        return True

    def _process_dataset(self, processor, dataset_path: Path, 
                        start_speaker_id: int, max_speakers: int) -> Tuple[List, List, Dict]:
        """
        Process a single dataset using its processor
        
        Args:
            processor: Dataset processor instance
            dataset_path: Path to dataset
            start_speaker_id: Starting speaker ID
            max_speakers: Maximum speakers to process
            
        Returns:
            Tuple of (features, labels, speaker_info)
        """
        features = []
        labels = []
        speaker_info = {}
        current_id = start_speaker_id
        
        try:
            # Use processor to get data
            if hasattr(processor, 'process_dataset'):
                # New style processor
                data = processor.process_dataset(
                    str(dataset_path),
                    max_speakers=max_speakers,
                    samples_per_speaker=50
                )
                
                for speaker_name, audio_files in data.items():
                    if current_id >= start_speaker_id + max_speakers:
                        break
                        
                    speaker_features = []
                    for audio_file in audio_files[:10]:  # Limit files per speaker
                        try:
                            # Extract features using audio processor
                            feature_vector = self.audio_processor.extract_features(audio_file)
                            if feature_vector is not None:
                                speaker_features.append(feature_vector)
                        except Exception as e:
                            self.logger.warning(f"Failed to extract features from {audio_file}: {e}")
                            continue
                    
                    # Add speaker if we have enough samples
                    if len(speaker_features) >= 3:
                        features.extend(speaker_features)
                        labels.extend([current_id] * len(speaker_features))
                        speaker_info[current_id] = {
                            'name': speaker_name,
                            'samples': len(speaker_features),
                            'dataset': processor.__class__.__name__
                        }
                        current_id += 1
            
            else:
                # Fallback: process files directly
                self._process_directory_structure(
                    dataset_path, features, labels, speaker_info, 
                    current_id, max_speakers
                )
                
        except Exception as e:
            self.logger.error(f"Error in dataset processing: {e}")
        
        return features, labels, speaker_info

    def _process_directory_structure(self, dataset_path: Path, features: List, 
                                   labels: List, speaker_info: Dict, 
                                   start_id: int, max_speakers: int):
        """
        Process dataset with directory structure: dataset/speaker/audio_files
        """
        current_id = start_id
        
        # Look for speaker directories
        for speaker_dir in dataset_path.iterdir():
            if not speaker_dir.is_dir() or current_id >= start_id + max_speakers:
                continue
                
            speaker_name = speaker_dir.name
            audio_files = []
            
            # Find audio files
            for ext in ['*.wav', '*.mp3', '*.ogg', '*.m4a']:
                audio_files.extend(speaker_dir.glob(ext))
            
            if len(audio_files) < 3:  # Need at least 3 samples
                continue
                
            speaker_features = []
            for audio_file in audio_files[:10]:  # Limit files per speaker
                try:
                    feature_vector = self.audio_processor.extract_features(str(audio_file))
                    if feature_vector is not None:
                        speaker_features.append(feature_vector)
                except Exception as e:
                    self.logger.warning(f"Failed to extract features from {audio_file}: {e}")
                    continue
              # Add speaker if we have enough samples
            if len(speaker_features) >= 3:
                features.extend(speaker_features)
                labels.extend([current_id] * len(speaker_features))
                speaker_info[current_id] = {
                    'name': speaker_name,
                    'samples': len(speaker_features),
                    'dataset': 'directory_structure'
                }
                current_id += 1
                self.logger.info(f"Added speaker {speaker_name} with {len(speaker_features)} samples")

    def _process_common_voice_dataset(self, dataset_path: Path, features: List, 
                                     labels: List, speaker_info: Dict, 
                                     start_id: int, max_speakers: int,
                                     min_duration: float, max_duration: float,
                                     samples_per_speaker: int):
        """
        Process Common Voice dataset using train.tsv file with progress bar and caching
        """
        tsv_file = dataset_path / "train.tsv"
        clips_dir = dataset_path / "clips"
        
        if not tsv_file.exists():
            self.logger.error(f"Common Voice TSV file not found: {tsv_file}")
            return
        
        if not clips_dir.exists():
            self.logger.error(f"Common Voice clips directory not found: {clips_dir}")
            return
        
        try:
            import pandas as pd
            
            # Read TSV file
            self.logger.info("Loading Common Voice dataset...")
            df = pd.read_csv(tsv_file, sep='\t')
            self.logger.info(f"Loaded {len(df)} entries from Common Voice")
            
            # Group by client_id (speaker)
            user_groups = df.groupby('client_id')
            self.logger.info(f"Found {len(user_groups)} unique speakers in Common Voice")
            
            current_id = start_id
            processed_speakers = 0
            
            # Create progress bar for speakers
            speakers_to_process = min(max_speakers, len(user_groups))
            pbar_speakers = tqdm(
                total=speakers_to_process,
                desc="Processing CV speakers",
                unit="speakers"
            )
            
            start_time = time.time()
            
            for client_id, group in user_groups:
                if processed_speakers >= max_speakers:
                    break
                
                # Filter by audio duration if column exists
                if 'duration' in group.columns:
                    group = group[
                        (group['duration'] >= min_duration) & 
                        (group['duration'] <= max_duration)
                    ]
                
                # Limit samples per speaker
                group = group.head(samples_per_speaker)
                
                if len(group) < 3:  # Need at least 3 samples
                    continue
                
                speaker_features = []
                valid_files = 0
                audio_files_to_process = []
                for _, row in group.iterrows():
                    audio_file = clips_dir / row['path']
                    if not audio_file.exists():
                        continue
                    audio_files_to_process.append(audio_file)
                    if len(audio_files_to_process) >= 10:
                        break
                # Параллельное извлечение признаков
                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    results = list(executor.map(self.extract_features_cached, audio_files_to_process))
                for feature_vector in results:
                    if feature_vector is not None:
                        speaker_features.append(feature_vector)
                        valid_files += 1
                
                # Add speaker if we have enough samples
                if len(speaker_features) >= 3:
                    features.extend(speaker_features)
                    labels.extend([current_id] * len(speaker_features))
                    speaker_info[current_id] = {
                        'name': f"cv_speaker_{current_id}",
                        'client_id': client_id,
                        'samples': len(speaker_features),
                        'dataset': 'common_voice'
                    }
                    current_id += 1
                    processed_speakers += 1
                    
                    # Update progress bar
                    pbar_speakers.update(1)
                    pbar_speakers.set_postfix({
                        'samples': len(speaker_features),
                        'total_features': len(features)
                    })
            
            pbar_speakers.close()
            
            # Save cache after processing
            self.save_cache()
            
            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Added {processed_speakers} speakers from Common Voice "
                f"with total {len(features)} samples in {elapsed_time:.1f}s"
            )
            
        except Exception as e:
            self.logger.error(f"Error processing Common Voice dataset: {e}")

    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about available datasets
        
        Returns:
            Dictionary with dataset information
        """
        info = {
            'datasets_dir': str(self.datasets_dir),
            'processed_dir': str(self.processed_dir),
            'available_datasets': [],
            'processed_data_exists': False
        }
        
        # Check available datasets
        if self.datasets_dir.exists():
            for item in self.datasets_dir.iterdir():
                if item.is_dir():
                    info['available_datasets'].append(item.name)
        
        # Check processed data
        training_data_path = self.processed_dir / "training_data.npz"
        if training_data_path.exists():
            info['processed_data_exists'] = True
            try:
                data = np.load(training_data_path, allow_pickle=True)
                info['num_samples'] = len(data['features'])
                info['feature_dim'] = data['features'].shape[1] if len(data['features']) > 0 else 0
                speaker_info = json.loads(data['speaker_info'].item())
                info['num_speakers'] = len(speaker_info)
            except Exception as e:
                self.logger.error(f"Error reading processed data: {e}")
        
        return info
