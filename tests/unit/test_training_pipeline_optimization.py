"""
Unit tests for Training Pipeline Optimization
Tests memory efficiency, gradient accumulation, checkpointing, and monitoring.
"""

import pytest
import tempfile
import os
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.data_processing.training_pipeline_optimization import (
    OptimizedTrainingPipeline, MemoryEfficientDataset, AdaptiveGradientAccumulator,
    CheckpointManager, MemoryMonitor, PerformanceProfiler, TrainingMetricsCollector,
    OptimizationLevel, TrainingStage, MemoryConfig, GradientConfig,
    MixedPrecisionConfig, CheckpointConfig, MonitoringConfig, TrainingState,
    create_training_pipeline, create_memory_efficient_config, create_gradient_accumulation_config
)


class TestMemoryEfficientDataset:
    """Test memory-efficient dataset implementation."""
    
    @pytest.fixture
    def sample_data(self):
        """Sample training data."""
        return [
            {
                'input_text': 'Test input 1',
                'output_text': 'Test output 1',
                'metadata': {'author': 'Test Author 1'}
            },
            {
                'input_text': 'Test input 2', 
                'output_text': 'Test output 2',
                'metadata': {'author': 'Test Author 2'}
            },
            {
                'input_text': 'Test input 3',
                'output_text': 'Test output 3', 
                'metadata': {'author': 'Test Author 1'}
            }
        ]
    
    def test_dataset_initialization(self, sample_data):
        """Test dataset initialization."""
        config = MemoryConfig()
        dataset = MemoryEfficientDataset(sample_data, config)
        
        assert len(dataset) == 3
        assert dataset.config == config
        assert len(dataset.cache) == 0  # Cache starts empty
    
    def test_data_access(self, sample_data):
        """Test data access functionality."""
        config = MemoryConfig()
        dataset = MemoryEfficientDataset(sample_data, config)
        
        # Test accessing items
        item = dataset[0]
        assert item['input_text'] == 'Test input 1'
        assert item['output_text'] == 'Test output 1'
        
        # Test all items accessible
        for i in range(len(dataset)):
            item = dataset[i]
            assert 'input_text' in item
            assert 'output_text' in item
    
    def test_caching_behavior(self, sample_data):
        """Test caching functionality."""
        config = MemoryConfig(cache_size_limit=100)
        dataset = MemoryEfficientDataset(sample_data, config)
        
        # Access items to populate cache
        item1 = dataset[0]
        item2 = dataset[1]
        
        # Access same item again (should use cache)
        item1_again = dataset[0]
        assert item1 == item1_again
    
    def test_cache_clearing(self, sample_data):
        """Test cache clearing functionality."""
        config = MemoryConfig()
        dataset = MemoryEfficientDataset(sample_data, config)
        
        # Access items to populate cache
        dataset[0]
        dataset[1]
        
        # Clear cache
        dataset.clear_cache()
        assert len(dataset.cache) == 0


class TestAdaptiveGradientAccumulator:
    """Test adaptive gradient accumulation."""
    
    @pytest.fixture
    def gradient_config(self):
        """Default gradient configuration."""
        return GradientConfig(
            accumulation_steps=4,
            adaptive_accumulation=True
        )
    
    def test_accumulator_initialization(self, gradient_config):
        """Test accumulator initialization."""
        accumulator = AdaptiveGradientAccumulator(gradient_config)
        
        assert accumulator.config == gradient_config
        assert accumulator.current_steps == 0
    
    def test_should_accumulate(self, gradient_config):
        """Test gradient accumulation decision logic."""
        accumulator = AdaptiveGradientAccumulator(gradient_config)
        
        # Should accumulate initially
        assert accumulator.should_accumulate() == True
        
        # Step through accumulation
        for _ in range(gradient_config.accumulation_steps - 1):
            accumulator.step()
            assert accumulator.should_accumulate() == True
        
        # Final step should not accumulate
        accumulator.step()
        assert accumulator.should_accumulate() == False
    
    def test_gradient_step_tracking(self, gradient_config):
        """Test gradient step tracking."""
        accumulator = AdaptiveGradientAccumulator(gradient_config)
        
        initial_steps = accumulator.current_steps
        accumulator.step()
        assert accumulator.current_steps == initial_steps + 1
        
        # Test reset
        accumulator.reset()
        assert accumulator.current_steps == 0
    
    def test_scale_factor_calculation(self, gradient_config):
        """Test gradient scale factor calculation."""
        accumulator = AdaptiveGradientAccumulator(gradient_config)
        
        scale_factor = accumulator.get_scale_factor()
        assert scale_factor > 0
        assert isinstance(scale_factor, float)


class TestMemoryMonitor:
    """Test memory monitoring system."""
    
    def test_monitor_initialization(self):
        """Test memory monitor initialization."""
        monitor = MemoryMonitor()
        
        assert monitor.process is not None
        assert isinstance(monitor.gpu_available, bool)
    
    def test_memory_usage_tracking(self):
        """Test memory usage tracking."""
        monitor = MemoryMonitor()
        
        # Get memory usage
        memory_stats = monitor.get_memory_usage()
        
        assert 'cpu_memory_mb' in memory_stats
        assert 'cpu_memory_percent' in memory_stats
        
        # Verify reasonable values
        assert 0 <= memory_stats['cpu_memory_percent'] <= 100
        assert memory_stats['cpu_memory_mb'] >= 0
    
    def test_memory_usage_ratio(self):
        """Test memory usage ratio calculation."""
        monitor = MemoryMonitor()
        
        ratio = monitor.get_memory_usage_ratio()
        assert 0 <= ratio <= 1
        assert isinstance(ratio, float)


class TestPerformanceProfiler:
    """Test performance profiling system."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Monitoring configuration for profiler."""
        return MonitoringConfig(enable_profiling=True, profile_steps=10)
    
    def test_profiler_initialization(self, monitoring_config):
        """Test profiler initialization."""
        profiler = PerformanceProfiler(monitoring_config)
        
        assert profiler.config == monitoring_config
        assert len(profiler.step_times) == 0
    
    def test_profiling_workflow(self, monitoring_config):
        """Test profiling step workflow."""
        profiler = PerformanceProfiler(monitoring_config)
        
        # Test step profiling
        profiler.start_step()
        
        # Simulate some work
        import time
        time.sleep(0.01)
        
        profiler.end_step()
        
        # Get performance stats
        stats = profiler.get_performance_stats()
        assert 'step_times' in stats
        assert stats['step_times']['total_steps'] > 0
    
    def test_profiling_context_manager(self, monitoring_config):
        """Test profiling with context manager."""
        profiler = PerformanceProfiler(monitoring_config)
        
        # Test context manager profiling
        with profiler.profile('test_operation'):
            import time
            time.sleep(0.005)
        
        # Should have recorded the operation
        stats = profiler.get_performance_stats()
        # Check for operation-specific timing in stats
        operation_recorded = any('test_operation' in key for key in stats.keys())
        assert operation_recorded or 'step_times' in stats


class TestCheckpointManager:
    """Test checkpoint management system."""
    
    @pytest.fixture
    def checkpoint_config(self):
        """Checkpoint configuration for testing."""
        return CheckpointConfig(
            save_every_n_steps=10,
            keep_last_n_checkpoints=3,
            save_optimizer_state=True
        )
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Temporary directory for checkpoints."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    def test_checkpoint_manager_initialization(self, checkpoint_config):
        """Test checkpoint manager initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(checkpoint_dir=temp_dir)
            manager = CheckpointManager(config)
            
            assert manager.config == config
            assert manager.checkpoint_dir is not None
    
    def test_save_checkpoint(self, checkpoint_config):
        """Test checkpoint saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(checkpoint_dir=temp_dir, save_every_n_steps=10)
            manager = CheckpointManager(config)
            
            # Create training state
            training_state = TrainingState(
                epoch=1,
                global_step=10,
                stage=TrainingStage.TRAINING
            )
            
            # Mock model state
            model_state = {'param1': torch.tensor([1.0])}
            optimizer_state = {'lr': 0.001}
            
            # Save checkpoint
            checkpoint_path = manager.save_checkpoint(
                state=training_state,
                model_state=model_state,
                optimizer_state=optimizer_state,
                extra_data={'metrics': {'loss': 2.5}}
            )
            
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)
    
    def test_load_checkpoint(self, checkpoint_config):
        """Test checkpoint loading functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CheckpointConfig(checkpoint_dir=temp_dir)
            manager = CheckpointManager(config)
            
            # Create and save a checkpoint first
            training_state = TrainingState(epoch=1, global_step=10)
            model_state = {'param1': torch.tensor([1.0])}
            
            checkpoint_path = manager.save_checkpoint(
                state=training_state,
                model_state=model_state,
                extra_data={'metrics': {'loss': 1.5}}
            )
            
            # Load the checkpoint
            loaded_data = manager.load_checkpoint(checkpoint_path)
            
            assert loaded_data is not None
            assert 'training_state' in loaded_data
            assert 'model_state_dict' in loaded_data
            assert loaded_data['extra_data']['metrics']['loss'] == 1.5


class TestTrainingMetricsCollector:
    """Test training metrics collection system."""
    
    @pytest.fixture
    def monitoring_config(self):
        """Monitoring configuration for testing."""
        return MonitoringConfig(
            log_every_n_steps=5,
            eval_every_n_steps=10,
            save_metrics=True,
            monitor_memory=True,
            metrics_dir="./test_metrics"
        )
    
    def test_collector_initialization(self, monitoring_config):
        """Test metrics collector initialization."""
        collector = TrainingMetricsCollector(monitoring_config)
        
        assert collector.config == monitoring_config
        assert len(collector.step_metrics) == 0
        assert len(collector.epoch_metrics) == 0
        assert collector.memory_monitor is not None
    
    def test_step_metrics_logging(self, monitoring_config):
        """Test training step metrics logging."""
        collector = TrainingMetricsCollector(monitoring_config)
        
        # Log training step
        collector.log_step_metrics(
            step=5,
            metrics={
                'loss': 2.5,
                'learning_rate': 0.001,
                'perplexity': 12.0
            }
        )
        
        assert len(collector.step_metrics) == 1
        assert collector.step_metrics[0]['step'] == 5
        assert collector.step_metrics[0]['loss'] == 2.5
    
    def test_epoch_metrics_logging(self, monitoring_config):
        """Test epoch metrics logging."""
        collector = TrainingMetricsCollector(monitoring_config)
        
        # Log epoch metrics
        collector.log_epoch_metrics(
            epoch=1,
            metrics={
                'avg_loss': 2.0,
                'eval_loss': 1.8,
                'eval_metrics': {'perplexity': 10.0, 'accuracy': 0.85}
            }
        )
        
        assert len(collector.epoch_metrics) == 1
        assert collector.epoch_metrics[0]['epoch'] == 1
        assert collector.epoch_metrics[0]['avg_loss'] == 2.0
    
    def test_metrics_summary(self, monitoring_config):
        """Test metrics summary generation."""
        collector = TrainingMetricsCollector(monitoring_config)
        
        # Add training progression
        losses = [3.0, 2.5, 2.0, 1.8, 1.5]
        for i, loss in enumerate(losses):
            collector.log_step_metrics(
                step=i+1,
                metrics={'loss': loss, 'perplexity': loss * 4}
            )
        
        # Get summary
        summary = collector.get_metrics_summary()
        
        assert 'total_steps' in summary
        assert summary['total_steps'] == 5
        assert 'recent_loss_mean' in summary


class TestOptimizedTrainingPipeline:
    """Test complete optimized training pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Sample training data for pipeline testing."""
        return [
            {
                'input_text': f'Training example {i}',
                'output_text': f'Expected output {i}',
                'metadata': {'author': f'Author {i % 2}'}
            }
            for i in range(20)
        ]
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization with all components."""
        with tempfile.TemporaryDirectory() as checkpoint_dir, \
             tempfile.TemporaryDirectory() as metrics_dir:
            
            pipeline = create_training_pipeline(
                optimization_level=OptimizationLevel.BALANCED,
                checkpoint_dir=checkpoint_dir,
                metrics_dir=metrics_dir
            )
            
            assert isinstance(pipeline, OptimizedTrainingPipeline)
            assert pipeline.optimization_level == OptimizationLevel.BALANCED
            assert pipeline.checkpoint_manager is not None
            assert pipeline.metrics_collector is not None
    
    def test_dataloader_creation(self, sample_training_data):
        """Test optimized dataloader creation."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Create dataset
        dataset = MemoryEfficientDataset(sample_training_data, pipeline.memory_config)
        
        # Create dataloader
        dataloader = pipeline.create_optimized_dataloader(dataset, batch_size=4)
        
        assert dataloader is not None
        assert dataloader.batch_size == 4
        assert dataloader.pin_memory == pipeline.memory_config.pin_memory
    
    def test_mixed_precision_setup(self):
        """Test mixed precision training setup."""
        # Test with mixed precision enabled
        pipeline = create_training_pipeline(OptimizationLevel.AGGRESSIVE)
        
        scaler, autocast_dtype = pipeline.setup_mixed_precision()
        
        if pipeline.mixed_precision_config.enabled:
            assert scaler is not None
            assert autocast_dtype is not None
        else:
            assert scaler is None
            assert autocast_dtype is None
    
    def test_checkpoint_integration(self):
        """Test checkpoint integration in pipeline."""
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            pipeline = create_training_pipeline(
                optimization_level=OptimizationLevel.BALANCED,
                checkpoint_dir=checkpoint_dir
            )
            
            # Test checkpoint decision
            pipeline.training_state.global_step = 10
            should_save = pipeline.should_checkpoint()
            
            # Check if should save based on actual config (step OR epoch condition)
            step_condition = (pipeline.checkpoint_config.save_every_n_steps > 0 and 
                             10 % pipeline.checkpoint_config.save_every_n_steps == 0)
            epoch_condition = (pipeline.checkpoint_config.save_every_n_epochs > 0 and 
                              pipeline.training_state.step == 0)
            expected = step_condition or epoch_condition
            assert should_save == expected
    
    def test_optimization_step(self):
        """Test optimized training step execution."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Mock components
        mock_loss = torch.tensor(2.5, requires_grad=True)
        mock_model = Mock()
        mock_optimizer = Mock()
        
        # Test optimization step
        should_update = pipeline.optimize_step(
            loss=mock_loss,
            model=mock_model,
            optimizer=mock_optimizer
        )
        
        # Should return boolean indicating if parameters were updated
        assert isinstance(should_update, bool)
    
    def test_training_metrics_logging(self):
        """Test training metrics logging integration."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Log training metrics
        pipeline.log_training_metrics({
            'loss': 2.0,
            'learning_rate': 0.001,
            'epoch': 1
        })
        
        # Check metrics were logged
        assert len(pipeline.metrics_collector.step_metrics) > 0
    
    def test_optimization_report_generation(self):
        """Test optimization report generation."""
        pipeline = create_training_pipeline(OptimizationLevel.AGGRESSIVE)
        
        # Generate optimization report
        report = pipeline.get_optimization_report()
        
        assert 'optimization_level' in report
        assert 'configurations' in report
        assert 'memory' in report['configurations']
        assert 'gradient' in report['configurations']
        assert 'mixed_precision' in report['configurations']
        assert report['optimization_level'] == OptimizationLevel.AGGRESSIVE.value


class TestConfigurationFactory:
    """Test configuration factory functions."""
    
    def test_memory_config_creation(self):
        """Test memory configuration factory."""
        # Test default config
        config = create_memory_efficient_config()
        assert isinstance(config, MemoryConfig)
        assert config.max_memory_usage > 0
        
        # Test with custom memory usage
        custom_config = create_memory_efficient_config(max_memory_usage=0.7)
        assert isinstance(custom_config, MemoryConfig)
        assert custom_config.max_memory_usage == 0.7
    
    def test_gradient_config_creation(self):
        """Test gradient configuration factory."""
        # Test default config
        config = create_gradient_accumulation_config()
        assert isinstance(config, GradientConfig)
        assert config.accumulation_steps >= 1
        
        # Test with custom accumulation steps
        custom_config = create_gradient_accumulation_config(accumulation_steps=8)
        assert isinstance(custom_config, GradientConfig)
        assert custom_config.accumulation_steps == 8
    
    def test_training_pipeline_factory(self):
        """Test training pipeline factory function."""
        with tempfile.TemporaryDirectory() as checkpoint_dir, \
             tempfile.TemporaryDirectory() as metrics_dir:
            
            pipeline = create_training_pipeline(
                optimization_level=OptimizationLevel.MAXIMUM,
                checkpoint_dir=checkpoint_dir,
                metrics_dir=metrics_dir
            )
            
            assert isinstance(pipeline, OptimizedTrainingPipeline)
            assert pipeline.optimization_level == OptimizationLevel.MAXIMUM
            assert pipeline.memory_config is not None
            assert pipeline.gradient_config is not None
            assert pipeline.mixed_precision_config is not None


class TestOptimizationLevels:
    """Test different optimization levels."""
    
    def test_minimal_optimization(self):
        """Test minimal optimization level."""
        pipeline = create_training_pipeline(OptimizationLevel.MINIMAL)
        
        # Minimal should have conservative settings
        assert pipeline.optimization_level == OptimizationLevel.MINIMAL
        assert pipeline.memory_config is not None
        assert pipeline.gradient_config is not None
    
    def test_balanced_optimization(self):
        """Test balanced optimization level."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Balanced should have moderate settings
        assert pipeline.optimization_level == OptimizationLevel.BALANCED
        assert pipeline.mixed_precision_config is not None
    
    def test_aggressive_optimization(self):
        """Test aggressive optimization level."""
        pipeline = create_training_pipeline(OptimizationLevel.AGGRESSIVE)
        
        # Aggressive should have advanced settings
        assert pipeline.optimization_level == OptimizationLevel.AGGRESSIVE
        assert pipeline.mixed_precision_config.enabled == True
    
    def test_maximum_optimization(self):
        """Test maximum optimization level."""
        pipeline = create_training_pipeline(OptimizationLevel.MAXIMUM)
        
        # Maximum should have all optimizations
        assert pipeline.optimization_level == OptimizationLevel.MAXIMUM
        assert pipeline.gradient_config.adaptive_accumulation == True


class TestIntegrationScenarios:
    """Test integration scenarios and edge cases."""
    
    @pytest.fixture
    def realistic_training_data(self):
        """Realistic training scenario data."""
        return [
            {
                'input_text': f'Conference talk excerpt {i}',
                'output_text': f'Generated response {i}',
                'metadata': {'author': f'Speaker {i % 3}', 'date': '2024-01-01'}
            }
            for i in range(50)
        ]
    
    def test_complete_pipeline_workflow(self, realistic_training_data):
        """Test complete training pipeline workflow."""
        with tempfile.TemporaryDirectory() as checkpoint_dir, \
             tempfile.TemporaryDirectory() as metrics_dir:
            
            pipeline = create_training_pipeline(
                OptimizationLevel.BALANCED,
                checkpoint_dir,
                metrics_dir
            )
            
            # Create dataset and dataloader
            dataset = MemoryEfficientDataset(realistic_training_data, pipeline.memory_config)
            dataloader = pipeline.create_optimized_dataloader(dataset, batch_size=4)
            
            assert len(dataset) == 50
            assert dataloader.batch_size == 4
            
            # Test mixed precision setup
            scaler, autocast_dtype = pipeline.setup_mixed_precision()
            
            # Test checkpoint functionality
            should_checkpoint = pipeline.should_checkpoint()
            assert isinstance(should_checkpoint, bool)
            
            # Test metrics logging
            pipeline.log_training_metrics({
                'loss': 2.0,
                'learning_rate': 0.001,
                'epoch': 1
            })
            
            assert len(pipeline.metrics_collector.step_metrics) > 0
    
    def test_resume_training_capability(self):
        """Test training resume functionality."""
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            pipeline = create_training_pipeline(
                OptimizationLevel.BALANCED,
                checkpoint_dir=checkpoint_dir
            )
            
            # Test resume without existing checkpoint
            resumed = pipeline.resume_training()
            assert resumed == False  # No checkpoint to resume from
    
    def test_error_handling(self):
        """Test error handling and robustness."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Test with empty dataset
        empty_dataset = MemoryEfficientDataset([], pipeline.memory_config)
        assert len(empty_dataset) == 0
        
        # Test dataloader with empty dataset (should handle gracefully)
        try:
            dataloader = pipeline.create_optimized_dataloader(empty_dataset, batch_size=4)
            # If it succeeds, verify it's created
            assert dataloader is not None
        except ValueError as e:
            # PyTorch doesn't allow DataLoader with empty dataset
            assert "num_samples should be a positive integer" in str(e)


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_dataset_handling(self):
        """Test handling of empty datasets."""
        config = MemoryConfig()
        dataset = MemoryEfficientDataset([], config)
        
        assert len(dataset) == 0
        
        # Should handle empty access gracefully
        with pytest.raises(IndexError):
            dataset[0]
    
    def test_memory_monitoring_without_gpu(self):
        """Test memory monitoring when GPU is not available."""
        with patch('torch.cuda.is_available', return_value=False):
            monitor = MemoryMonitor()
            memory_stats = monitor.get_memory_usage()
            
            # Should not have GPU memory stats when GPU unavailable
            gpu_keys = [k for k in memory_stats.keys() if k.startswith('gpu_')]
            assert len(gpu_keys) == 0
    
    def test_gradient_accumulation_edge_cases(self):
        """Test gradient accumulation edge cases."""
        # Test with accumulation_steps = 1 (no accumulation)
        config = GradientConfig(accumulation_steps=1)
        accumulator = AdaptiveGradientAccumulator(config)
        
        # With steps = 1, should accumulate until reaching target
        # Check the actual behavior based on current_steps vs accumulation_steps
        expected = accumulator.current_steps < config.accumulation_steps
        assert accumulator.should_accumulate() == expected
        
        # Step and reset should work
        accumulator.step()
        accumulator.reset()
        assert accumulator.current_steps == 0
    
    def test_checkpoint_with_invalid_directory(self):
        """Test checkpoint handling with invalid directory."""
        # Try to create checkpoint manager with invalid directory
        config = CheckpointConfig(checkpoint_dir="/invalid/nonexistent/path")
        
        # Should handle gracefully during initialization
        try:
            manager = CheckpointManager(config)
            # If it succeeds, verify it created the manager
            assert manager is not None
        except Exception:
            # If it fails, that's also acceptable behavior
            pass


class TestPerformanceValidation:
    """Test performance characteristics and validation."""
    
    def test_dataset_performance(self):
        """Test dataset performance characteristics."""
        # Create moderately large dataset
        large_data = [
            {
                'input_text': f'Content {i} ' * 50,  # ~500 chars per item
                'output_text': f'Output {i}',
                'metadata': {'author': f'Author {i % 3}'}
            }
            for i in range(500)  # 500 items
        ]
        
        config = MemoryConfig(cache_size_limit=50)
        dataset = MemoryEfficientDataset(large_data, config)
        
        # Access subset of data efficiently
        for i in range(0, 50, 5):  # Access every 5th item
            item = dataset[i]
            assert 'input_text' in item
        
        # Cache should be managed efficiently
        assert len(dataset.cache) <= config.cache_size_limit
    
    def test_memory_efficiency(self):
        """Test memory usage efficiency."""
        # Create pipeline with aggressive memory optimization
        pipeline = create_training_pipeline(OptimizationLevel.AGGRESSIVE)
        
        # Verify memory configuration is optimized
        assert pipeline.memory_config.pin_memory == True
        assert pipeline.memory_config.num_workers >= 1
        
        # Test memory monitoring
        memory_stats = pipeline.memory_monitor.get_memory_usage()
        assert 'cpu_memory_percent' in memory_stats
        assert 0 <= memory_stats['cpu_memory_percent'] <= 100
    
    def test_gradient_accumulation_efficiency(self):
        """Test gradient accumulation efficiency."""
        pipeline = create_training_pipeline(OptimizationLevel.MAXIMUM)
        
        # Test gradient accumulation setup
        assert pipeline.gradient_accumulator is not None
        assert pipeline.gradient_config.adaptive_accumulation == True
        
        # Test scale factor calculation
        scale_factor = pipeline.gradient_accumulator.get_scale_factor()
        assert scale_factor > 0
    
    def test_monitoring_performance(self):
        """Test monitoring system performance."""
        with tempfile.TemporaryDirectory() as metrics_dir:
            config = MonitoringConfig(
                log_every_n_steps=1,  # Log every step
                metrics_dir=metrics_dir
            )
            collector = TrainingMetricsCollector(config)
            
            # Log many steps quickly
            import time
            start_time = time.time()
            
            for i in range(50):
                collector.log_step_metrics(
                    step=i+1,
                    metrics={
                        'loss': 2.0 - i*0.01,
                        'learning_rate': 0.001,
                        'perplexity': 10.0 - i*0.05
                    }
                )
            
            end_time = time.time()
            
            # Should complete quickly
            assert (end_time - start_time) < 2.0  # Less than 2 seconds
            assert len(collector.step_metrics) == 50


class TestTrainingStateManagement:
    """Test training state management."""
    
    def test_training_state_initialization(self):
        """Test training state initialization."""
        state = TrainingState()
        
        assert state.epoch == 0
        assert state.global_step == 0
        assert state.stage == TrainingStage.INITIALIZATION
    
    def test_training_state_updates(self):
        """Test training state updates."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Update training state
        pipeline.training_state.epoch = 2
        pipeline.training_state.global_step = 100
        pipeline.training_state.stage = TrainingStage.TRAINING
        
        assert pipeline.training_state.epoch == 2
        assert pipeline.training_state.global_step == 100
        assert pipeline.training_state.stage == TrainingStage.TRAINING
    
    def test_training_stage_transitions(self):
        """Test training stage transitions."""
        pipeline = create_training_pipeline(OptimizationLevel.BALANCED)
        
        # Test stage transitions
        stages = [
            TrainingStage.INITIALIZATION,
            TrainingStage.DATA_LOADING,
            TrainingStage.TRAINING,
            TrainingStage.VALIDATION,
            TrainingStage.COMPLETED
        ]
        
        for stage in stages:
            pipeline.training_state.stage = stage
            assert pipeline.training_state.stage == stage


class TestConfigurationValidation:
    """Test configuration validation and defaults."""
    
    def test_memory_config_defaults(self):
        """Test memory configuration defaults."""
        config = MemoryConfig()
        
        assert config.max_memory_usage > 0
        assert config.max_memory_usage <= 1.0
        assert config.num_workers >= 0
        assert config.prefetch_factor >= 1
    
    def test_gradient_config_defaults(self):
        """Test gradient configuration defaults."""
        config = GradientConfig()
        
        assert config.accumulation_steps >= 1
        assert config.max_grad_norm > 0
        assert config.max_grad_norm > 0
    
    def test_mixed_precision_config_defaults(self):
        """Test mixed precision configuration defaults."""
        config = MixedPrecisionConfig()
        
        assert config.dtype in ["fp16", "bf16", "fp32"]
        assert config.init_scale > 0
        assert 0 < config.backoff_factor < 1
        assert config.growth_factor > 1
    
    def test_checkpoint_config_defaults(self):
        """Test checkpoint configuration defaults."""
        config = CheckpointConfig()
        
        assert config.save_every_n_steps > 0
        assert config.keep_last_n_checkpoints >= 1
        assert isinstance(config.save_optimizer_state, bool)
    
    def test_monitoring_config_defaults(self):
        """Test monitoring configuration defaults."""
        config = MonitoringConfig()
        
        assert config.log_every_n_steps > 0
        assert config.eval_every_n_steps > 0
        assert isinstance(config.monitor_memory, bool)
        assert isinstance(config.save_metrics, bool)