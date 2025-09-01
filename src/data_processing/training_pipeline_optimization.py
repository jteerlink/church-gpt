"""
Training Pipeline Optimization (Phase 5)
Implements advanced training optimizations including memory efficiency, gradient accumulation,
mixed precision training, resume capability, and comprehensive monitoring.
"""

import os
import json
import time
import torch
import logging
from typing import Dict, List, Any, Optional, Iterator, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
import numpy as np
import psutil
from torch.utils.data import DataLoader, Dataset
from contextlib import contextmanager


class OptimizationLevel(Enum):
    """Training optimization levels."""
    MINIMAL = "minimal"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"


class TrainingStage(Enum):
    """Training pipeline stages."""
    INITIALIZATION = "initialization"
    DATA_LOADING = "data_loading"
    TRAINING = "training"
    VALIDATION = "validation"
    CHECKPOINTING = "checkpointing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class MemoryConfig:
    """Memory optimization configuration."""
    max_memory_usage: float = 0.85  # Maximum memory usage ratio
    gradient_checkpointing: bool = True
    pin_memory: bool = True
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    enable_memory_mapping: bool = True
    cache_size_limit: int = 1000  # Number of cached items


@dataclass
class GradientConfig:
    """Gradient accumulation configuration."""
    accumulation_steps: int = 16
    max_grad_norm: float = 1.0
    sync_gradients: bool = True
    gradient_checkpointing: bool = True
    adaptive_accumulation: bool = False
    memory_threshold: float = 0.8


@dataclass
class MixedPrecisionConfig:
    """Mixed precision training configuration."""
    enabled: bool = True
    dtype: str = "fp16"  # fp16, bf16, or fp32
    loss_scale: Optional[str] = "dynamic"
    init_scale: float = 65536.0
    growth_interval: int = 2000
    backoff_factor: float = 0.5
    growth_factor: float = 2.0


@dataclass
class CheckpointConfig:
    """Checkpoint and resume configuration."""
    save_every_n_steps: int = 100
    save_every_n_epochs: int = 1
    keep_last_n_checkpoints: int = 3
    save_optimizer_state: bool = True
    save_scheduler_state: bool = True
    checkpoint_dir: str = "./checkpoints"
    resume_from_latest: bool = True


@dataclass
class MonitoringConfig:
    """Training monitoring configuration."""
    log_every_n_steps: int = 10
    eval_every_n_steps: int = 50
    monitor_memory: bool = True
    monitor_performance: bool = True
    save_metrics: bool = True
    metrics_dir: str = "./metrics"
    enable_profiling: bool = False
    profile_steps: int = 100


@dataclass
class TrainingState:
    """Complete training state for resuming."""
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    best_loss: float = float('inf')
    best_metric: float = 0.0
    learning_rate: float = 0.0
    stage: TrainingStage = TrainingStage.INITIALIZATION
    start_time: float = 0.0
    total_training_time: float = 0.0
    metrics_history: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metrics_history is None:
            self.metrics_history = []


class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset with lazy loading and caching."""
    
    def __init__(self, data: List[Dict[str, Any]], config: MemoryConfig):
        self.data_references = data  # Store references, not full data
        self.config = config
        self.cache = {}
        self.cache_order = []
        self.memory_mapped = {}
        
        # Setup memory mapping if enabled
        if config.enable_memory_mapping:
            self._setup_memory_mapping()
    
    def _setup_memory_mapping(self):
        """Setup memory mapping for large data files."""
        # This would be implemented to memory-map large text files
        # For now, we'll use a simple implementation
        pass
    
    def __len__(self):
        return len(self.data_references)
    
    def __getitem__(self, idx):
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load data lazily
        item = self._load_item(idx)
        
        # Cache management
        if len(self.cache) < self.config.cache_size_limit:
            self.cache[idx] = item
            self.cache_order.append(idx)
        else:
            # Remove oldest item if cache is full
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
            self.cache[idx] = item
            self.cache_order.append(idx)
        
        return item
    
    def _load_item(self, idx):
        """Load individual item (to be overridden)."""
        return self.data_references[idx]
    
    def clear_cache(self):
        """Clear the cache to free memory."""
        self.cache.clear()
        self.cache_order.clear()


class AdaptiveGradientAccumulator:
    """Adaptive gradient accumulation based on memory usage."""
    
    def __init__(self, config: GradientConfig):
        self.config = config
        self.current_steps = 0
        self.target_steps = config.accumulation_steps
        self.memory_monitor = MemoryMonitor()
    
    def should_accumulate(self) -> bool:
        """Determine if gradients should be accumulated."""
        if not self.config.adaptive_accumulation:
            return self.current_steps < self.config.accumulation_steps
        
        # Adaptive logic based on memory usage
        memory_usage_ratio = self.memory_monitor.get_memory_usage_ratio()
        if memory_usage_ratio > self.config.memory_threshold:
            # Reduce accumulation steps if memory is high
            self.target_steps = max(1, self.config.accumulation_steps // 2)
        else:
            self.target_steps = self.config.accumulation_steps
        
        return self.current_steps < self.target_steps
    
    def step(self):
        """Increment accumulation step."""
        self.current_steps += 1
    
    def reset(self):
        """Reset accumulation counter."""
        self.current_steps = 0
    
    def get_scale_factor(self) -> float:
        """Get scaling factor for loss."""
        return 1.0 / self.target_steps


class MemoryMonitor:
    """Monitor system and GPU memory usage."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        stats = {}
        
        # CPU memory
        memory_info = self.process.memory_info()
        stats['cpu_memory_mb'] = memory_info.rss / 1024 / 1024
        stats['cpu_memory_percent'] = self.process.memory_percent()
        
        # GPU memory
        if self.gpu_available:
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_stats(i)
                allocated = gpu_memory.get('allocated_bytes.all.current', 0) / 1024 / 1024
                reserved = gpu_memory.get('reserved_bytes.all.current', 0) / 1024 / 1024
                stats[f'gpu_{i}_allocated_mb'] = allocated
                stats[f'gpu_{i}_reserved_mb'] = reserved
        
        return stats
    
    def get_memory_usage_ratio(self) -> float:
        """Get overall memory usage ratio (0.0 to 1.0)."""
        cpu_percent = self.process.memory_percent() / 100.0
        
        if self.gpu_available:
            gpu_ratios = []
            for i in range(torch.cuda.device_count()):
                total_memory = torch.cuda.get_device_properties(i).total_memory
                allocated = torch.cuda.memory_allocated(i)
                gpu_ratios.append(allocated / total_memory)
            
            if gpu_ratios:
                return max(cpu_percent, max(gpu_ratios))
        
        return cpu_percent


class PerformanceProfiler:
    """Profile training performance and identify bottlenecks."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.profiles = {}
        self.step_times = []
        self.current_step_start = None
    
    @contextmanager
    def profile(self, operation_name: str):
        """Context manager for profiling operations."""
        start_time = time.time()
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_name not in self.profiles:
                self.profiles[operation_name] = []
            self.profiles[operation_name].append(duration)
    
    def start_step(self):
        """Mark the start of a training step."""
        self.current_step_start = time.time()
    
    def end_step(self):
        """Mark the end of a training step."""
        if self.current_step_start:
            step_time = time.time() - self.current_step_start
            self.step_times.append(step_time)
            self.current_step_start = None
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}
        
        # Step timing
        if self.step_times:
            stats['step_times'] = {
                'mean': np.mean(self.step_times),
                'std': np.std(self.step_times),
                'min': np.min(self.step_times),
                'max': np.max(self.step_times),
                'total_steps': len(self.step_times)
            }
        
        # Operation profiling
        for operation, times in self.profiles.items():
            stats[f'{operation}_timing'] = {
                'mean': np.mean(times),
                'total_time': np.sum(times),
                'count': len(times)
            }
        
        return stats


class CheckpointManager:
    """Manage training checkpoints with smart saving and loading."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def save_checkpoint(self, state: TrainingState, model_state: Dict[str, Any],
                       optimizer_state: Optional[Dict[str, Any]] = None,
                       scheduler_state: Optional[Dict[str, Any]] = None,
                       extra_data: Optional[Dict[str, Any]] = None) -> str:
        """Save a complete training checkpoint."""
        checkpoint_name = f"checkpoint_step_{state.global_step}_epoch_{state.epoch}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        checkpoint_data = {
            'training_state': asdict(state),
            'model_state_dict': model_state,
            'checkpoint_info': {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': time.time(),
                'pytorch_version': torch.__version__
            }
        }
        
        if self.config.save_optimizer_state and optimizer_state:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
        
        if self.config.save_scheduler_state and scheduler_state:
            checkpoint_data['scheduler_state_dict'] = scheduler_state
        
        if extra_data:
            checkpoint_data['extra_data'] = extra_data
        
        try:
            torch.save(checkpoint_data, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Clean up old checkpoints
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
        
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Load a training checkpoint."""
        if checkpoint_path is None:
            if self.config.resume_from_latest:
                checkpoint_path = self._find_latest_checkpoint()
            else:
                return None
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return checkpoint_data
        
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint file."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoint_files:
            return None
        
        # Sort by modification time
        latest_checkpoint = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        return str(latest_checkpoint)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only the most recent ones."""
        checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if len(checkpoint_files) <= self.config.keep_last_n_checkpoints:
            return
        
        # Sort by modification time and remove oldest
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        for old_checkpoint in checkpoint_files[self.config.keep_last_n_checkpoints:]:
            try:
                old_checkpoint.unlink()
                self.logger.info(f"Removed old checkpoint: {old_checkpoint}")
            except Exception as e:
                self.logger.warning(f"Failed to remove old checkpoint {old_checkpoint}: {e}")


class TrainingMetricsCollector:
    """Collect and manage training metrics."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.metrics_dir = Path(config.metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        self.step_metrics = []
        self.epoch_metrics = []
        self.memory_monitor = MemoryMonitor()
        self.profiler = PerformanceProfiler(config)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def log_step_metrics(self, step: int, metrics: Dict[str, Any]):
        """Log metrics for a training step."""
        step_data = {
            'step': step,
            'timestamp': time.time(),
            **metrics
        }
        
        # Add memory metrics if enabled
        if self.config.monitor_memory:
            memory_stats = self.memory_monitor.get_memory_usage()
            step_data.update(memory_stats)
        
        # Add performance metrics if enabled
        if self.config.monitor_performance:
            perf_stats = self.profiler.get_performance_stats()
            step_data.update(perf_stats)
        
        self.step_metrics.append(step_data)
        
        # Save metrics if enabled
        if self.config.save_metrics and step % self.config.log_every_n_steps == 0:
            self._save_step_metrics()
    
    def log_epoch_metrics(self, epoch: int, metrics: Dict[str, Any]):
        """Log metrics for a training epoch."""
        epoch_data = {
            'epoch': epoch,
            'timestamp': time.time(),
            **metrics
        }
        
        self.epoch_metrics.append(epoch_data)
        
        if self.config.save_metrics:
            self._save_epoch_metrics()
    
    def _save_step_metrics(self):
        """Save step metrics to file."""
        if not self.step_metrics:
            return
        
        metrics_file = self.metrics_dir / "step_metrics.jsonl"
        try:
            with open(metrics_file, 'a') as f:
                for metric in self.step_metrics[-self.config.log_every_n_steps:]:
                    f.write(json.dumps(metric) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save step metrics: {e}")
    
    def _save_epoch_metrics(self):
        """Save epoch metrics to file."""
        if not self.epoch_metrics:
            return
        
        metrics_file = self.metrics_dir / "epoch_metrics.jsonl"
        try:
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(self.epoch_metrics[-1]) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to save epoch metrics: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of all collected metrics."""
        summary = {
            'total_steps': len(self.step_metrics),
            'total_epochs': len(self.epoch_metrics),
            'collection_time': time.time()
        }
        
        # Performance summary
        if self.step_metrics:
            recent_steps = self.step_metrics[-100:]  # Last 100 steps
            if recent_steps:
                losses = [s.get('loss', 0) for s in recent_steps if 'loss' in s]
                if losses:
                    summary['recent_loss_mean'] = np.mean(losses)
                    summary['recent_loss_std'] = np.std(losses)
        
        return summary


class OptimizedTrainingPipeline:
    """Complete optimized training pipeline with all enhancements."""
    
    def __init__(self, 
                 memory_config: MemoryConfig = None,
                 gradient_config: GradientConfig = None,
                 mixed_precision_config: MixedPrecisionConfig = None,
                 checkpoint_config: CheckpointConfig = None,
                 monitoring_config: MonitoringConfig = None,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED):
        
        # Initialize configurations with defaults
        self.memory_config = memory_config or MemoryConfig()
        self.gradient_config = gradient_config or GradientConfig()
        self.mixed_precision_config = mixed_precision_config or MixedPrecisionConfig()
        self.checkpoint_config = checkpoint_config or CheckpointConfig()
        self.monitoring_config = monitoring_config or MonitoringConfig()
        self.optimization_level = optimization_level
        
        # Apply optimization level presets
        self._apply_optimization_presets()
        
        # Initialize components
        self.gradient_accumulator = AdaptiveGradientAccumulator(self.gradient_config)
        self.checkpoint_manager = CheckpointManager(self.checkpoint_config)
        self.metrics_collector = TrainingMetricsCollector(self.monitoring_config)
        self.memory_monitor = MemoryMonitor()
        
        # Training state
        self.training_state = TrainingState()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
    
    def _apply_optimization_presets(self):
        """Apply optimization presets based on optimization level."""
        if self.optimization_level == OptimizationLevel.MINIMAL:
            self.memory_config.gradient_checkpointing = False
            self.memory_config.num_workers = 2
            self.gradient_config.accumulation_steps = 8
            self.mixed_precision_config.enabled = False
            
        elif self.optimization_level == OptimizationLevel.BALANCED:
            # Keep defaults
            pass
            
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            self.memory_config.max_memory_usage = 0.9
            self.memory_config.num_workers = 6
            self.gradient_config.accumulation_steps = 32
            self.gradient_config.adaptive_accumulation = True
            self.mixed_precision_config.dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            
        elif self.optimization_level == OptimizationLevel.MAXIMUM:
            self.memory_config.max_memory_usage = 0.95
            self.memory_config.num_workers = 8
            self.memory_config.prefetch_factor = 4
            self.gradient_config.accumulation_steps = 64
            self.gradient_config.adaptive_accumulation = True
            self.mixed_precision_config.dtype = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
            self.monitoring_config.enable_profiling = True
    
    def create_optimized_dataloader(self, dataset: Dataset, batch_size: int, 
                                  shuffle: bool = True) -> DataLoader:
        """Create optimized data loader with memory-efficient settings."""
        # Convert to memory-efficient dataset if needed
        if not isinstance(dataset, MemoryEfficientDataset):
            if hasattr(dataset, 'data'):
                dataset = MemoryEfficientDataset(dataset.data, self.memory_config)
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.memory_config.num_workers,
            pin_memory=self.memory_config.pin_memory,
            prefetch_factor=self.memory_config.prefetch_factor,
            persistent_workers=self.memory_config.persistent_workers,
            drop_last=True  # For consistent batch sizes
        )
        
        return dataloader
    
    def setup_mixed_precision(self):
        """Setup mixed precision training."""
        if not self.mixed_precision_config.enabled:
            return None, None
        
        # Create GradScaler for mixed precision
        if self.mixed_precision_config.dtype in ["fp16", "bf16"]:
            scaler = torch.cuda.amp.GradScaler(
                init_scale=self.mixed_precision_config.init_scale,
                growth_interval=self.mixed_precision_config.growth_interval,
                backoff_factor=self.mixed_precision_config.backoff_factor,
                growth_factor=self.mixed_precision_config.growth_factor,
                enabled=True
            )
            
            # Determine autocast dtype
            if self.mixed_precision_config.dtype == "bf16":
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16
            
            return scaler, autocast_dtype
        
        return None, None
    
    def resume_training(self) -> bool:
        """Attempt to resume training from the latest checkpoint."""
        checkpoint_data = self.checkpoint_manager.load_checkpoint()
        
        if checkpoint_data is None:
            self.logger.info("No checkpoint found, starting from scratch")
            return False
        
        try:
            # Restore training state
            training_state_dict = checkpoint_data.get('training_state', {})
            for key, value in training_state_dict.items():
                if hasattr(self.training_state, key):
                    setattr(self.training_state, key, value)
            
            self.logger.info(f"Resumed training from step {self.training_state.global_step}, "
                           f"epoch {self.training_state.epoch}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume training: {e}")
            return False
    
    def should_checkpoint(self) -> bool:
        """Determine if a checkpoint should be saved."""
        step_condition = (self.checkpoint_config.save_every_n_steps > 0 and 
                         self.training_state.global_step % self.checkpoint_config.save_every_n_steps == 0)
        
        epoch_condition = (self.checkpoint_config.save_every_n_epochs > 0 and 
                          self.training_state.step == 0)  # Beginning of epoch
        
        return step_condition or epoch_condition
    
    def optimize_step(self, loss: torch.Tensor, model, optimizer, scaler=None) -> bool:
        """Perform optimized training step with gradient accumulation."""
        # Scale loss for gradient accumulation
        scale_factor = self.gradient_accumulator.get_scale_factor()
        scaled_loss = loss * scale_factor
        
        # Backward pass with mixed precision
        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        self.gradient_accumulator.step()
        
        # Check if we should update parameters
        if not self.gradient_accumulator.should_accumulate():
            if scaler is not None:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                
                # Clip gradients
                if self.gradient_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 self.gradient_config.max_grad_norm)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                # Clip gradients
                if self.gradient_config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                                 self.gradient_config.max_grad_norm)
                
                optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()
            self.gradient_accumulator.reset()
            
            return True  # Parameter update occurred
        
        return False  # Still accumulating
    
    def log_training_metrics(self, metrics: Dict[str, Any]):
        """Log comprehensive training metrics."""
        # Add training state info
        metrics.update({
            'epoch': self.training_state.epoch,
            'step': self.training_state.step,
            'global_step': self.training_state.global_step,
            'learning_rate': self.training_state.learning_rate
        })
        
        self.metrics_collector.log_step_metrics(self.training_state.global_step, metrics)
        
        # Log to console if needed
        if self.training_state.global_step % self.monitoring_config.log_every_n_steps == 0:
            self.logger.info(f"Step {self.training_state.global_step}: {metrics}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'optimization_level': self.optimization_level.value,
            'training_state': asdict(self.training_state),
            'configurations': {
                'memory': asdict(self.memory_config),
                'gradient': asdict(self.gradient_config),
                'mixed_precision': asdict(self.mixed_precision_config),
                'checkpoint': asdict(self.checkpoint_config),
                'monitoring': asdict(self.monitoring_config)
            },
            'performance_metrics': self.metrics_collector.get_metrics_summary(),
            'memory_usage': self.memory_monitor.get_memory_usage(),
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
        }
        
        return report


# Factory functions for easy pipeline creation
def create_training_pipeline(optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                           checkpoint_dir: str = "./checkpoints",
                           metrics_dir: str = "./metrics") -> OptimizedTrainingPipeline:
    """Create an optimized training pipeline with sensible defaults."""
    checkpoint_config = CheckpointConfig(checkpoint_dir=checkpoint_dir)
    monitoring_config = MonitoringConfig(metrics_dir=metrics_dir)
    
    return OptimizedTrainingPipeline(
        checkpoint_config=checkpoint_config,
        monitoring_config=monitoring_config,
        optimization_level=optimization_level
    )


def create_memory_efficient_config(max_memory_usage: float = 0.85) -> MemoryConfig:
    """Create memory-efficient configuration."""
    return MemoryConfig(
        max_memory_usage=max_memory_usage,
        gradient_checkpointing=True,
        pin_memory=True,
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        enable_memory_mapping=True,
        cache_size_limit=1000
    )


def create_gradient_accumulation_config(accumulation_steps: int = 16) -> GradientConfig:
    """Create gradient accumulation configuration."""
    return GradientConfig(
        accumulation_steps=accumulation_steps,
        max_grad_norm=1.0,
        sync_gradients=True,
        gradient_checkpointing=True,
        adaptive_accumulation=True,
        memory_threshold=0.8
    )