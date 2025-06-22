"""
Audio processor for feature extraction from voice recordings
"""

import librosa
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)


class AudioProcessor:
    """
    Class for processing audio files and extracting voice features.
    
    Extracts compact but informative features:
    - MFCC (Mel-frequency cepstral coefficients)
    - Spectral features
    - Prosodic features (pitch, rhythm)
    - Statistical parameters
    """
    
    def __init__(self, sample_rate: int = 16000, 
                 n_mfcc: int = 13,
                 max_duration: float = 10.0):
        """
        Initialize the audio processor
        
        Args:
            sample_rate: Sampling rate (16 kHz optimal for voice)
            n_mfcc: Number of MFCC coefficients
            max_duration: Maximum duration for processing (seconds)
        """
        self.logger = logging.getLogger(__name__)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.max_duration = max_duration
        self.target_length = int(sample_rate * max_duration)
        
        self.logger.debug(f"AudioProcessor initialized: sr={sample_rate}, mfcc={n_mfcc}, max_dur={max_duration}s")
    
    def load_audio(self, file_path: str) -> np.ndarray:
        """
        Load and preprocess an audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Normalized audio signal
        """
        try:
            # Load with automatic resampling
            audio, _ = librosa.load(
                file_path, 
                sr=self.sample_rate,
                duration=self.max_duration
            )
            
            # Normalize loudness
            if len(audio) > 0:
                audio = librosa.util.normalize(audio)
                
                # Pad or trim to target length
                if len(audio) < self.target_length:
                    # Zero-padding
                    audio = np.pad(audio, (0, self.target_length - len(audio)))
                else:
                    # Trimming
                    audio = audio[:self.target_length]
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio loading error {file_path}: {e}")
            raise
    
    def extract_mfcc_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features
        
        Args:
            audio: Audio signal
            
        Returns:
            MFCC features
        """
        # Extract MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=self.n_mfcc,
            n_fft=2048,
            hop_length=512
        )
        
        # Delta and delta-delta coefficients (dynamics)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        # Combine all MFCC features
        combined_mfcc = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
        
        return combined_mfcc
    
    def extract_spectral_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features
        
        Args:
            audio: Audio signal
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        # Spectral centroid (brightness)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        )
        
        # Spectral bandwidth
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate
        )
        
        # Spectral rolloff (roll-off frequency)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate
        )
        
        # Zero-crossing rate (related to pitch)
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        
        # Chroma features
        features['chroma'] = librosa.feature.chroma_stft(
            y=audio, sr=self.sample_rate
        )
        
        return features
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract prosodic features (pitch, rhythm, energy)
        
        Args:
            audio: Audio signal
            
        Returns:
            Prosodic characteristics
        """
        features = {}
        
        # Fundamental frequency (F0)
        try:
            f0 = librosa.yin(
                audio, 
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7')   # ~2093 Hz
            )
            
            # F0 statistics
            valid_f0 = f0[f0 > 0]  # Remove zero values
            if len(valid_f0) > 0:
                features['f0_mean'] = np.mean(valid_f0)
                features['f0_std'] = np.std(valid_f0)
                features['f0_min'] = np.min(valid_f0)
                features['f0_max'] = np.max(valid_f0)
                features['f0_range'] = features['f0_max'] - features['f0_min']
                features['voiced_ratio'] = len(valid_f0) / len(f0)
            else:
                # Default values if F0 not found
                features.update({
                    'f0_mean': 150.0, 'f0_std': 20.0,
                    'f0_min': 100.0, 'f0_max': 200.0,
                    'f0_range': 100.0, 'voiced_ratio': 0.5
                })
        except:
            # Fallback values
            features.update({
                'f0_mean': 150.0, 'f0_std': 20.0,
                'f0_min': 100.0, 'f0_max': 200.0,
                'f0_range': 100.0, 'voiced_ratio': 0.5
            })
        
        # Energy characteristics
        features['energy_mean'] = np.mean(audio ** 2)
        features['energy_std'] = np.std(audio ** 2)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features
    
    def extract_statistical_features(self, features_2d: np.ndarray) -> np.ndarray:
        """
        Extract statistical characteristics from 2D features
        
        Args:
            features_2d: 2D array of features (features x time)
            
        Returns:
            1D vector of statistical characteristics
        """
        stats = []
        
        for feature_row in features_2d:
            # Basic statistics
            stats.extend([
                np.mean(feature_row),      # Mean
                np.std(feature_row),       # Standard deviation
                np.min(feature_row),       # Minimum
                np.max(feature_row),       # Maximum
                np.median(feature_row),    # Median
            ])
            
            # Additional statistics
            stats.extend([
                np.percentile(feature_row, 25),  # 1st quartile
                np.percentile(feature_row, 75),  # 3rd quartile
                float(np.var(feature_row)),      # Variance
            ])
        
        return np.array(stats)
    
    def extract_features(self, file_path: str) -> np.ndarray:
        """
        Extract the full set of features from an audio file
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Feature vector for machine learning
        """
        self.logger.debug(f"Extracting features from: {file_path}")
        
        try:
            # Load audio
            audio = self.load_audio(file_path)
            
            if len(audio) == 0:
                raise ValueError("Empty audio file")
            
            # 1. MFCC features
            mfcc_features = self.extract_mfcc_features(audio)
            mfcc_stats = self.extract_statistical_features(mfcc_features)
            
            # 2. Spectral features
            spectral_features = self.extract_spectral_features(audio)
            spectral_stats = []
            for feature_name, feature_values in spectral_features.items():
                stats = self.extract_statistical_features(feature_values)
                spectral_stats.extend(stats)
            spectral_stats = np.array(spectral_stats)
            
            # 3. Prosodic features
            prosodic_features = self.extract_prosodic_features(audio)
            prosodic_vector = np.array(list(prosodic_features.values()))
            
            # Combine all features
            final_features = np.concatenate([
                mfcc_stats,
                spectral_stats, 
                prosodic_vector
            ])
            
            # Check for NaN and replace with zeros
            final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
            
            self.logger.debug(f"Extracted {len(final_features)} features")
            return final_features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error from {file_path}: {e}")
            raise
    
    def analyze_voice_characteristics(self, audio_files: List[str]) -> Dict[str, Any]:
        """
        Analyze voice characteristics based on multiple files
        
        Args:
            audio_files: List of audio files from the same speaker
            
        Returns:
            Voice characteristics
        """
        self.logger.debug(f"Analyzing voice characteristics for {len(audio_files)} files")
        
        all_f0 = []
        all_formants = []
        quality_scores = []
        
        for audio_file in audio_files:
            try:
                audio = self.load_audio(audio_file)
                
                # Fundamental frequency
                prosodic = self.extract_prosodic_features(audio)
                if prosodic['f0_mean'] > 0:
                    all_f0.append([prosodic['f0_min'], prosodic['f0_max']])
                
                # Quality assessment (based on energy and voice/silence ratio)
                quality = min(1.0, prosodic['voiced_ratio'] * 2)  # Simple metric
                quality_scores.append(quality)
                
                # Approximate formants (simplified calculation via spectral peaks)
                formants = self._estimate_formants(audio)
                all_formants.append(formants)
                
            except Exception as e:
                self.logger.warning(f"Analysis error {audio_file}: {e}")
        
        # Aggregate results
        if all_f0:
            pitch_ranges = np.array(all_f0)
            pitch_range = [
                float(np.mean(pitch_ranges[:, 0])),  # Mean minimum
                float(np.mean(pitch_ranges[:, 1]))   # Mean maximum
            ]
        else:
            pitch_range = [100, 200]  # Default
        
        if all_formants:
            formants = np.mean(all_formants, axis=0).tolist()
        else:
            formants = [500, 1500, 2500]  # Default
        
        quality_score = float(np.mean(quality_scores)) if quality_scores else 0.8
        
        return {
            "pitch_range": pitch_range,
            "formants": formants,
            "quality_score": quality_score
        }
    
    def _estimate_formants(self, audio: np.ndarray) -> List[float]:
        """
        Simplified formant estimation via spectral peaks
        
        Args:
            audio: Audio signal
            
        Returns:
            Estimated frequencies of the first three formants
        """
        try:
            # Compute spectrum
            fft = np.fft.rfft(audio)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            
            # Peak detection in typical formant ranges
            f1_range = (200, 800)    # F1: 200-800 Hz
            f2_range = (800, 2500)   # F2: 800-2500 Hz  
            f3_range = (2500, 4000)  # F3: 2500-4000 Hz
            
            formants = []
            
            for freq_range in [f1_range, f2_range, f3_range]:
                mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                if np.any(mask):
                    range_freqs = freqs[mask]
                    range_magnitude = magnitude[mask]
                    
                    if len(range_magnitude) > 0:
                        peak_idx = np.argmax(range_magnitude)
                        formant_freq = range_freqs[peak_idx]
                        formants.append(float(formant_freq))
                    else:
                        formants.append(float(np.mean(freq_range)))
                else:
                    formants.append(float(np.mean(freq_range)))
            
            return formants
            
        except Exception:
            # Default formant values
            return [500.0, 1500.0, 2500.0]
    
    def validate_audio_file(self, file_path: str, 
                          min_duration: float = 0.5,
                          max_duration: float = 30.0) -> bool:
        """
        Validate audio file
        
        Args:
            file_path: Path to file
            min_duration: Minimum duration (seconds)
            max_duration: Maximum duration (seconds)
            
        Returns:
            True if file is valid
        """
        try:
            # Check file existence
            if not Path(file_path).exists():
                return False
              # Quick load just to check duration
            duration = librosa.get_duration(path=file_path)
            
            # Check duration
            if duration < min_duration or duration > max_duration:
                self.logger.debug(f"Invalid duration {file_path}: {duration:.2f}s")
                return False
            
            # Check loading
            audio = self.load_audio(file_path)
            if len(audio) == 0:
                return False
            
            # Check for audio signal (not just silence)
            energy = np.mean(audio ** 2)
            if energy < 1e-6:  # Too quiet
                self.logger.debug(f"File too quiet: {file_path}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Validation error {file_path}: {e}")
            return False
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Retrieve information about an audio file
        
        Args:
            file_path: Path to file
            
        Returns:
            File information
        """
        try:
            duration = librosa.get_duration(filename=file_path)
            
            # Load for analysis
            audio, sr = librosa.load(file_path, sr=None)
            
            info = {
                "duration": duration,
                "sample_rate": sr,
                "samples": len(audio),
                "channels": 1,  # librosa loads in mono
                "energy": float(np.mean(audio ** 2)),
                "max_amplitude": float(np.max(np.abs(audio))),
                "file_size": Path(file_path).stat().st_size
            }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting info for {file_path}: {e}")
            return {}
    
    def preprocess_batch(self, audio_files: List[str], 
                        show_progress: bool = True) -> Tuple[np.ndarray, List[str]]:
        """
        Batch process audio files
        
        Args:
            audio_files: List of file paths
            show_progress: Show progress bar
            
        Returns:
            Feature array and list of successfully processed files
        """
        features_list = []
        valid_files = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                audio_files = tqdm(audio_files, desc="Processing audio")
            except ImportError:
                pass
        
        for audio_file in audio_files:
            try:
                if self.validate_audio_file(audio_file):
                    features = self.extract_features(audio_file)
                    features_list.append(features)
                    valid_files.append(audio_file)
                else:
                    self.logger.debug(f"Skipping invalid file: {audio_file}")
            except Exception as e:
                self.logger.warning(f"Processing error {audio_file}: {e}")
        
        if features_list:
            features_array = np.array(features_list)
            self.logger.info(f"Processed {len(features_list)} out of {len(audio_files)} files")
            return features_array, valid_files
        else:
            self.logger.error("Failed to process any files")
            return np.array([]), []
