"""
Metrics for evaluating the performance of the speaker recognition system
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceMetrics:
    """
    Class for collecting and analyzing performance metrics
    """
    
    def __init__(self):
        """Initialize metrics"""
        self.logger = logging.getLogger(__name__)
        
        # Storage for results
        self.predictions = []
        self.ground_truth = []
        self.confidence_scores = []
        self.processing_times = []
        self.speaker_names = []
        
        # Statistics by speaker
        self.speaker_stats = defaultdict(lambda: {
            'correct': 0,
            'total': 0,
            'avg_confidence': 0.0,
            'avg_time': 0.0
        })
    
    def add_prediction(self, predicted_speaker: str, true_speaker: str,
                      confidence: float, processing_time: float,
                      speaker_name: str = None):
        """
        Add a prediction result
        
        Args:
            predicted_speaker: Predicted speaker
            true_speaker: True speaker
            confidence: Model confidence
            processing_time: Processing time (ms)
            speaker_name: Speaker name (optional)
        """
        self.predictions.append(predicted_speaker)
        self.ground_truth.append(true_speaker)
        self.confidence_scores.append(confidence)
        self.processing_times.append(processing_time)
        self.speaker_names.append(speaker_name or true_speaker)
        
        # Update statistics for the speaker
        is_correct = predicted_speaker == true_speaker
        stats = self.speaker_stats[true_speaker]
        
        # Update averages
        total = stats['total']
        stats['avg_confidence'] = (stats['avg_confidence'] * total + confidence) / (total + 1)
        stats['avg_time'] = (stats['avg_time'] * total + processing_time) / (total + 1)
        
        if is_correct:
            stats['correct'] += 1
        stats['total'] += 1
    
    def calculate_accuracy_metrics(self) -> Dict[str, float]:
        """
        Calculate accuracy metrics
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.predictions:
            return {}
        
        # Convert to numpy arrays
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # For multiclass classification
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'total_predictions': len(self.predictions)
        }
        
        return metrics
    
    def calculate_confidence_metrics(self) -> Dict[str, float]:
        """
        Analyze model confidence
        
        Returns:
            Confidence metrics
        """
        if not self.confidence_scores:
            return {}
        
        confidences = np.array(self.confidence_scores)
        
        # Correct and incorrect predictions
        correct_mask = np.array(self.predictions) == np.array(self.ground_truth)
        correct_confidences = confidences[correct_mask]
        incorrect_confidences = confidences[~correct_mask]
        
        metrics = {
            'avg_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'median_confidence': float(np.median(confidences))
        }
        
        if len(correct_confidences) > 0:
            metrics['avg_confidence_correct'] = float(np.mean(correct_confidences))
        
        if len(incorrect_confidences) > 0:
            metrics['avg_confidence_incorrect'] = float(np.mean(incorrect_confidences))
        
        return metrics
    
    def calculate_timing_metrics(self) -> Dict[str, float]:
        """
        Analyze processing time
        
        Returns:
            Timing metrics
        """
        if not self.processing_times:
            return {}
        
        times = np.array(self.processing_times)
        
        metrics = {
            'avg_processing_time_ms': float(np.mean(times)),
            'std_processing_time_ms': float(np.std(times)),
            'min_processing_time_ms': float(np.min(times)),
            'max_processing_time_ms': float(np.max(times)),
            'median_processing_time_ms': float(np.median(times)),
            'p95_processing_time_ms': float(np.percentile(times, 95)),
            'p99_processing_time_ms': float(np.percentile(times, 99))
        }
        
        return metrics
    
    def calculate_speaker_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Metrics for each speaker
        
        Returns:
            Metrics for each speaker
        """
        speaker_metrics = {}
        
        for speaker_id, stats in self.speaker_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                
                speaker_metrics[speaker_id] = {
                    'accuracy': float(accuracy),
                    'total_samples': stats['total'],
                    'correct_predictions': stats['correct'],
                    'avg_confidence': float(stats['avg_confidence']),
                    'avg_processing_time_ms': float(stats['avg_time'])
                }
        
        return speaker_metrics
    
    def get_confusion_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Compute the confusion matrix
        
        Returns:
            Tuple (confusion matrix, list of labels)
        """
        if not self.predictions:
            return np.array([]), []
        
        y_true = np.array(self.ground_truth)
        y_pred = np.array(self.predictions)
        
        # Get unique labels
        labels = sorted(list(set(y_true) | set(y_pred)))
        
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        return cm, labels
    
    def get_full_report(self) -> Dict[str, Any]:
        """
        Comprehensive performance report
        
        Returns:
            Comprehensive report
        """
        report = {
            'summary': {
                'total_predictions': len(self.predictions),
                'unique_speakers': len(self.speaker_stats),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'accuracy_metrics': self.calculate_accuracy_metrics(),
            'confidence_metrics': self.calculate_confidence_metrics(),
            'timing_metrics': self.calculate_timing_metrics(),
            'speaker_metrics': self.calculate_speaker_metrics()
        }
        
        # Add sklearn classification report
        if self.predictions and self.ground_truth:
            try:
                sklearn_report = classification_report(
                    self.ground_truth, 
                    self.predictions,
                    output_dict=True,
                    zero_division=0
                )
                report['classification_report'] = sklearn_report
            except Exception as e:
                self.logger.warning(f"Failed to create classification report: {e}")
        
        return report
    
    def save_report(self, output_file: str):
        """
        Save report to file
        
        Args:
            output_file: Path to output file
        """
        report = self.get_full_report()
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📊 Report saved: {output_path}")
    
    def print_summary(self):
        """Print a brief report to the console"""
        if not self.predictions:
            print("❌ No data for analysis")
            return
        
        accuracy_metrics = self.calculate_accuracy_metrics()
        timing_metrics = self.calculate_timing_metrics()
        confidence_metrics = self.calculate_confidence_metrics()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        
        print(f"📈 ACCURACY:")
        print(f"   Overall accuracy: {accuracy_metrics.get('accuracy', 0):.2%}")
        print(f"   Precision: {accuracy_metrics.get('precision', 0):.2%}")
        print(f"   Recall: {accuracy_metrics.get('recall', 0):.2%}")
        print(f"   F1-score: {accuracy_metrics.get('f1_score', 0):.2%}")
        print(f"   Total predictions: {accuracy_metrics.get('total_predictions', 0)}")
        
        print(f"\n⏱️ PROCESSING TIME:")
        print(f"   Average time: {timing_metrics.get('avg_processing_time_ms', 0):.1f} ms")
        print(f"   Median time: {timing_metrics.get('median_processing_time_ms', 0):.1f} ms")
        print(f"   95th percentile: {timing_metrics.get('p95_processing_time_ms', 0):.1f} ms")
        
        print(f"\n🎯 CONFIDENCE:")
        print(f"   Average confidence: {confidence_metrics.get('avg_confidence', 0):.2%}")
        print(f"   Confidence (correct): {confidence_metrics.get('avg_confidence_correct', 0):.2%}")
        print(f"   Confidence (incorrect): {confidence_metrics.get('avg_confidence_incorrect', 0):.2%}")
        
        print(f"\n👥 SPEAKERS:")
        speaker_metrics = self.calculate_speaker_metrics()
        print(f"   Unique speakers: {len(speaker_metrics)}")
        
        # Top-5 by accuracy
        sorted_speakers = sorted(
            speaker_metrics.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )[:5]
        
        print("   Top-5 by accuracy:")
        for speaker_id, metrics in sorted_speakers:
            name = next((name for i, name in enumerate(self.speaker_names) 
                        if self.ground_truth[i] == speaker_id), speaker_id)
            print(f"     {name}: {metrics['accuracy']:.1%} ({metrics['total_samples']} samples)")
        
        print("="*60)
    
    def plot_confusion_matrix(self, output_file: str = None, figsize: Tuple[int, int] = (10, 8)):
        """
        Plot the confusion matrix
        
        Args:
            output_file: Path for saving the plot
            figsize: Figure size
        """
        try:
            cm, labels = self.get_confusion_matrix()
            
            if cm.size == 0:
                self.logger.warning("No data to plot confusion matrix")
                return
            
            plt.figure(figsize=figsize)
            
            # Use percentages for better readability
            cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
            
            sns.heatmap(
                cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                cbar_kws={'label': 'Percentage (%)'}
            )
            
            plt.title('Confusion Matrix (%)', fontsize=16)
            plt.xlabel('Predicted Speaker', fontsize=12)
            plt.ylabel('True Speaker', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"📈 Confusion matrix saved: {output_file}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib/seaborn not installed, plot not created")
        except Exception as e:
            self.logger.error(f"Error creating plot: {e}")
    
    def plot_timing_distribution(self, output_file: str = None):
        """
        Plot the distribution of processing times
        
        Args:
            output_file: Path for saving the plot
        """
        try:
            if not self.processing_times:
                self.logger.warning("No processing time data")
                return
            
            plt.figure(figsize=(10, 6))
            
            times = np.array(self.processing_times)
            
            # Histogram
            plt.subplot(1, 2, 1)
            plt.hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Processing Time (ms)')
            plt.ylabel('Count')
            plt.title('Distribution of Processing Times')
            plt.grid(True, alpha=0.3)
            
            # Box plot
            plt.subplot(1, 2, 2)
            plt.boxplot(times, vert=True)
            plt.ylabel('Processing Time (ms)')
            plt.title('Box Plot of Processing Times')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"📈 Timing distribution plot saved: {output_file}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not installed, plot not created")
        except Exception as e:
            self.logger.error(f"Error creating timing distribution plot: {e}")
    
    def plot_confidence_analysis(self, output_file: str = None):
        """
        Analyze model confidence
        
        Args:
            output_file: Path for saving the plot
        """
        try:
            if not self.confidence_scores:
                self.logger.warning("No confidence data")
                return
            
            confidences = np.array(self.confidence_scores)
            correct_mask = np.array(self.predictions) == np.array(self.ground_truth)
            
            plt.figure(figsize=(12, 6))
            
            # Confidence distribution for correct and incorrect predictions
            plt.subplot(1, 2, 1)
            if np.any(correct_mask):
                plt.hist(confidences[correct_mask], bins=20, alpha=0.7, 
                        label='Correct', color='green', density=True)
            if np.any(~correct_mask):
                plt.hist(confidences[~correct_mask], bins=20, alpha=0.7, 
                        label='Incorrect', color='red', density=True)
            
            plt.xlabel('Confidence')
            plt.ylabel('Density')
            plt.title('Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Confidence vs accuracy plot
            plt.subplot(1, 2, 2)
            
            # Binning for analysis
            confidence_bins = np.linspace(0, 1, 11)
            bin_centers = (confidence_bins[:-1] + confidence_bins[1:]) / 2
            bin_accuracies = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (confidences >= confidence_bins[i]) & (confidences < confidence_bins[i+1])
                if np.any(mask):
                    accuracy = np.mean(correct_mask[mask])
                    bin_accuracies.append(accuracy)
                else:
                    bin_accuracies.append(0)
            
            plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8)
            plt.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5, label='Ideal Calibration')
            plt.xlabel('Confidence')
            plt.ylabel('Accuracy')
            plt.title('Model Calibration')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_file:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.info(f"📈 Confidence analysis plot saved: {output_file}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("matplotlib not installed, plot not created")
        except Exception as e:
            self.logger.error(f"Error creating confidence analysis plot: {e}")
    
    def clear(self):
        """Clear accumulated data"""
        self.predictions.clear()
        self.ground_truth.clear()
        self.confidence_scores.clear()
        self.processing_times.clear()
        self.speaker_names.clear()
        self.speaker_stats.clear()
        
        self.logger.debug("🧹 Metrics cleared")
    
    def export_detailed_results(self, output_file: str):
        """
        Export detailed results to CSV-like format
        
        Args:
            output_file: Path to output file
        """
        if not self.predictions:
            self.logger.warning("No data for export")
            return
        
        detailed_results = []
        
        for i in range(len(self.predictions)):
            result = {
                'index': i,
                'predicted_speaker': self.predictions[i],
                'true_speaker': self.ground_truth[i],
                'speaker_name': self.speaker_names[i],
                'confidence': self.confidence_scores[i],
                'processing_time_ms': self.processing_times[i],
                'is_correct': self.predictions[i] == self.ground_truth[i]
            }
            detailed_results.append(result)
        
        # Save as JSON for simplicity
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📋 Detailed results exported: {output_path}")
