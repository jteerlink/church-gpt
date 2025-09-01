"""
Training Configuration with Evaluation Integration
Provides training arguments and configuration optimized for LDS content fine-tuning.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import torch
import pandas as pd
import math
import numpy as np
from .evaluation import EarlyStoppingCallback, compute_training_metrics


@dataclass
class TrainingConfiguration:
    """Configuration class for fine-tuning training."""
    
    # Model configuration
    model_name: str = "google/gemma-7b"
    tokenizer_name: str = "google/gemma-7b"
    max_length: int = 512
    
    # Training parameters
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    
    # Evaluation configuration
    evaluation_strategy: str = "steps"
    eval_steps: int = 50
    save_steps: int = 25
    logging_steps: int = 10
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Optimization
    fp16: bool = True
    dataloader_pin_memory: bool = True
    gradient_checkpointing: bool = True
    
    # Output and reporting
    output_dir: str = "./results"
    run_name: Optional[str] = None
    report_to: Optional[str] = None


def create_optimized_training_args(config: TrainingConfiguration) -> Dict[str, Any]:
    """
    Create optimized training arguments from configuration.
    
    Args:
        config: Training configuration object
    
    Returns:
        Dictionary of training arguments compatible with HuggingFace Trainer
    """
    training_args = {
        'output_dir': config.output_dir,
        'num_train_epochs': config.num_train_epochs,
        'per_device_train_batch_size': config.per_device_train_batch_size,
        'per_device_eval_batch_size': config.per_device_eval_batch_size,
        'gradient_accumulation_steps': config.gradient_accumulation_steps,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'warmup_steps': config.warmup_steps,
        
        # Evaluation configuration
        'evaluation_strategy': config.evaluation_strategy,
        'eval_steps': config.eval_steps,
        'save_steps': config.save_steps,
        'logging_steps': config.logging_steps,
        
        # Optimization
        'fp16': config.fp16,
        'dataloader_pin_memory': config.dataloader_pin_memory,
        'gradient_checkpointing': config.gradient_checkpointing,
        
        # Checkpointing and saving
        'save_total_limit': 3,
        'load_best_model_at_end': True,
        'metric_for_best_model': 'eval_loss',
        'greater_is_better': False,
        
        # Reporting
        'report_to': config.report_to,
        'run_name': config.run_name
    }
    
    return training_args


def create_quantization_config() -> Dict[str, Any]:
    """
    Create modern quantization configuration.
    
    Returns:
        BitsAndBytesConfig compatible dictionary
    """
    return {
        'load_in_4bit': True,
        'bnb_4bit_use_double_quant': True,
        'bnb_4bit_quant_type': "nf4",
        'bnb_4bit_compute_dtype': torch.bfloat16
    }


def create_lora_config() -> Dict[str, Any]:
    """
    Create optimized LoRA configuration.
    
    Returns:
        LoraConfig compatible dictionary
    """
    return {
        'r': 32,  # Increased rank for better expressivity
        'lora_alpha': 64,  # Balanced scaling factor
        'target_modules': [
            "q_proj", "k_proj", "v_proj", "o_proj",  # All attention
            "gate_proj", "up_proj", "down_proj"      # MLP layers
        ],
        'lora_dropout': 0.1,
        'bias': "none",
        'task_type': "CAUSAL_LM"
    }


class EvaluationIntegratedTrainer:
    """Training wrapper with integrated evaluation framework."""
    
    def __init__(self, config: TrainingConfiguration):
        """
        Initialize trainer with evaluation integration.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.early_stopping = EarlyStoppingCallback(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_threshold,
            metric='eval_loss',
            mode='min'
        )
        self.training_metrics = []
        self.evaluation_results = []
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Get training arguments with evaluation integration."""
        return create_optimized_training_args(self.config)
    
    def get_compute_metrics_function(self):
        """Get metrics computation function for trainer."""
        return compute_training_metrics
    
    def log_evaluation_step(self, epoch: int, step: int, logs: Dict[str, float]):
        """Log evaluation step with early stopping check."""
        # Record metrics
        log_entry = {
            'epoch': epoch,
            'step': step,
            'timestamp': pd.Timestamp.now(),
            **logs
        }
        self.training_metrics.append(log_entry)
        
        # Check early stopping
        should_stop = self.early_stopping(logs, epoch)
        
        if should_stop:
            print(f"Early stopping triggered at epoch {epoch}, step {step}")
            print(f"Best {self.early_stopping.metric}: {self.early_stopping.get_best_value():.4f}")
        
        return should_stop
    
    def generate_training_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        if not self.training_metrics:
            return {'error': 'No training metrics recorded'}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.training_metrics)
        
        # Calculate training progression
        final_metrics = df.iloc[-1].to_dict()
        best_eval_loss = df['eval_loss'].min() if 'eval_loss' in df.columns else None
        
        # Training efficiency analysis
        efficiency_metrics = {
            'total_steps': len(df),
            'final_epoch': df['epoch'].max(),
            'training_duration': (df['timestamp'].max() - df['timestamp'].min()).total_seconds(),
            'convergence_efficiency': self._calculate_convergence_efficiency(df)
        }
        
        # Performance trends
        performance_trends = {
            'loss_trend': 'improving' if self._is_improving_trend(df, 'eval_loss') else 'plateauing',
            'best_eval_loss': best_eval_loss,
            'final_eval_loss': final_metrics.get('eval_loss'),
            'early_stopped': self.early_stopping.should_stop,
            'stopped_at_epoch': self.early_stopping.stopped_epoch if self.early_stopping.should_stop else None
        }
        
        return {
            'final_metrics': final_metrics,
            'efficiency_metrics': efficiency_metrics,
            'performance_trends': performance_trends,
            'early_stopping_summary': {
                'triggered': self.early_stopping.should_stop,
                'best_value': self.early_stopping.get_best_value(),
                'patience_used': self.early_stopping.wait_count,
                'total_patience': self.early_stopping.patience
            }
        }
    
    def _calculate_convergence_efficiency(self, df: pd.DataFrame) -> float:
        """Calculate how efficiently the model converged."""
        if 'eval_loss' not in df.columns or len(df) < 2:
            return 0.0
        
        initial_loss = df['eval_loss'].iloc[0]
        final_loss = df['eval_loss'].iloc[-1]
        best_loss = df['eval_loss'].min()
        
        if initial_loss <= final_loss:
            return 0.0  # No improvement
        
        # Efficiency: how much of potential improvement was achieved
        potential_improvement = initial_loss - best_loss
        actual_improvement = initial_loss - final_loss
        
        return actual_improvement / potential_improvement if potential_improvement > 0 else 0.0
    
    def _is_improving_trend(self, df: pd.DataFrame, metric: str, window: int = 5) -> bool:
        """Check if metric shows improving trend in recent window."""
        if metric not in df.columns or len(df) < window:
            return False
        
        recent_values = df[metric].tail(window).values
        
        # Check if there's a downward trend (for loss metrics)
        correlation = np.corrcoef(range(len(recent_values)), recent_values)[0, 1]
        return correlation < -0.1  # Negative correlation indicates improvement