"""
Integration tests for Evaluation Framework
Tests complete evaluation workflow with realistic conference content.
"""

import pytest
import tempfile
import pandas as pd
import math
from src.data_processing.evaluation import (
    TrainingEvaluator, EvaluationMetrics, ValidationSetCreator,
    create_evaluation_framework
)
from src.data_processing.training_config import (
    TrainingConfiguration, EvaluationIntegratedTrainer,
    create_optimized_training_args, create_lora_config, create_quantization_config
)


class TestEvaluationIntegration:
    """Integration tests for complete evaluation workflow."""
    
    @pytest.fixture
    def realistic_training_data(self):
        """Realistic training data from multiple General Conference speakers."""
        return [
            {
                'response': '''Brothers and sisters, I testify that faith is a principle of action and power. When we exercise faith in Jesus Christ, we are blessed with His Spirit and guidance.

Let me share an experience from my youth that taught me about the power of faith. Through faith, we can overcome any obstacle that stands in our way.''',
                'metadata': {
                    'author': 'Dieter F. Uchtdorf',
                    'title': 'The Power of Faith',
                    'date': '2020-04-01'
                }
            },
            {
                'response': '''My dear friends, I know that Jesus Christ lives. I have felt His love and witnessed His power in my life. This testimony has grown stronger through years of faithful service.

I invite each of you to seek your own witness of the Savior's reality and love.''',
                'metadata': {
                    'author': 'Dieter F. Uchtdorf', 
                    'title': 'Finding Faith',
                    'date': '2020-10-01'
                }
            },
            {
                'response': '''I know that God lives and that Jesus is the Christ. This knowledge has come through personal revelation and the witness of the Holy Ghost.

My testimony is that the Church of Jesus Christ of Latter-day Saints is the Lord's true church upon the earth.''',
                'metadata': {
                    'author': 'Russell M. Nelson',
                    'title': 'My Testimony',
                    'date': '2021-04-01'
                }
            },
            {
                'response': '''The Lord has blessed us with prophets to guide us in these latter days. I testify that revelation continues and that God speaks to His children today.

May we always listen to the voice of the Spirit and follow the counsel of living prophets.''',
                'metadata': {
                    'author': 'Russell M. Nelson',
                    'title': 'Continuing Revelation',
                    'date': '2021-10-01'
                }
            }
        ]
    
    def test_complete_evaluation_workflow(self, realistic_training_data):
        """Test complete evaluation workflow with realistic data."""
        evaluator = TrainingEvaluator(validation_ratio=0.25)
        
        # Prepare evaluation datasets
        eval_data = evaluator.prepare_evaluation_datasets(realistic_training_data)
        
        # Verify dataset preparation
        assert 'train_examples' in eval_data
        assert 'validation_examples' in eval_data
        assert 'author_corpuses' in eval_data
        
        # Verify author corpuses built correctly
        assert 'Dieter F. Uchtdorf' in eval_data['author_corpuses']
        assert 'Russell M. Nelson' in eval_data['author_corpuses']
        assert len(eval_data['author_corpuses']['Dieter F. Uchtdorf']) == 2
        assert len(eval_data['author_corpuses']['Russell M. Nelson']) == 2
        
        # Test model output evaluation for different authors
        uchtdorf_style_text = "Brothers and sisters, let me share an important principle about faith and hope."
        uchtdorf_evaluation = evaluator.evaluate_model_output(
            uchtdorf_style_text,
            'Dieter F. Uchtdorf',
            eval_data['author_corpuses']['Dieter F. Uchtdorf']
        )
        
        nelson_style_text = "I testify that God lives and reveals His will through living prophets."
        nelson_evaluation = evaluator.evaluate_model_output(
            nelson_style_text,
            'Russell M. Nelson', 
            eval_data['author_corpuses']['Russell M. Nelson']
        )
        
        # Both evaluations should return comprehensive results
        for evaluation in [uchtdorf_evaluation, nelson_evaluation]:
            assert 'style_similarity' in evaluation
            assert 'content_quality' in evaluation
            assert 'author_consistency' in evaluation
            assert 'overall_score' in evaluation
            assert 0 <= evaluation['overall_score'] <= 1
    
    def test_style_differentiation(self, realistic_training_data):
        """Test that evaluation can differentiate between author styles."""
        evaluator = TrainingEvaluator()
        eval_data = evaluator.prepare_evaluation_datasets(realistic_training_data)
        
        # Text that matches Uchtdorf's style
        uchtdorf_style = "Brothers and sisters, let me share an experience that taught me about faith."
        
        # Evaluate against both authors
        uchtdorf_eval = evaluator.evaluate_model_output(
            uchtdorf_style,
            'Dieter F. Uchtdorf',
            eval_data['author_corpuses']['Dieter F. Uchtdorf']
        )
        
        nelson_eval = evaluator.evaluate_model_output(
            uchtdorf_style,
            'Russell M. Nelson',
            eval_data['author_corpuses']['Russell M. Nelson']
        )
        
        # Should score higher for Uchtdorf (more similar style patterns)
        assert uchtdorf_eval['style_similarity']['rhetorical_score'] >= nelson_eval['style_similarity']['rhetorical_score']
    
    def test_validation_split_quality(self, realistic_training_data):
        """Test quality of validation split with realistic data."""
        split_creator = ValidationSetCreator(test_ratio=0.25)
        
        # Convert training examples to documents
        documents = [
            {
                'author': ex['metadata']['author'],
                'title': ex['metadata']['title'],
                'content': ex['response']
            }
            for ex in realistic_training_data
        ]
        
        train_docs, val_docs = split_creator.create_validation_split(documents)
        split_analysis = split_creator.analyze_split_quality(train_docs, val_docs)
        
        # Verify both authors represented
        train_authors = set(doc['author'] for doc in train_docs)
        val_authors = set(doc['author'] for doc in val_docs)
        
        assert len(train_authors) >= 1
        assert len(val_authors) >= 1  # Should have validation data
        
        # Verify split analysis
        assert split_analysis['author_coverage']['total_authors'] == 2
        assert split_analysis['split_summary']['total_documents'] == 4


class TestTrainingConfiguration:
    """Test training configuration and integration."""
    
    def test_training_configuration_creation(self):
        """Test training configuration creation."""
        config = TrainingConfiguration(
            model_name="test-model",
            learning_rate=1e-4,
            num_train_epochs=5
        )
        
        assert config.model_name == "test-model"
        assert config.learning_rate == 1e-4
        assert config.num_train_epochs == 5
        
        # Verify defaults
        assert config.gradient_accumulation_steps == 16
        assert config.fp16 == True
    
    def test_optimized_training_args_creation(self):
        """Test creation of optimized training arguments."""
        config = TrainingConfiguration(
            output_dir="./test_results",
            learning_rate=3e-4,
            eval_steps=25
        )
        
        args = create_optimized_training_args(config)
        
        # Verify key arguments
        assert args['output_dir'] == "./test_results"
        assert args['learning_rate'] == 3e-4
        assert args['eval_steps'] == 25
        assert args['evaluation_strategy'] == "steps"
        assert args['load_best_model_at_end'] == True
    
    def test_lora_config_creation(self):
        """Test LoRA configuration creation."""
        lora_config = create_lora_config()
        
        # Verify enhanced configuration
        assert lora_config['r'] == 32
        assert lora_config['lora_alpha'] == 64
        assert len(lora_config['target_modules']) == 7  # All attention + MLP
        assert 'q_proj' in lora_config['target_modules']
        assert 'gate_proj' in lora_config['target_modules']
    
    def test_quantization_config_creation(self):
        """Test quantization configuration creation."""
        quant_config = create_quantization_config()
        
        # Verify modern configuration
        assert quant_config['load_in_4bit'] == True
        assert quant_config['bnb_4bit_use_double_quant'] == True
        assert quant_config['bnb_4bit_quant_type'] == "nf4"


class TestEvaluationIntegratedTrainer:
    """Test evaluation-integrated trainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization with evaluation."""
        config = TrainingConfiguration(early_stopping_patience=5)
        trainer = EvaluationIntegratedTrainer(config)
        
        assert trainer.config == config
        assert trainer.early_stopping.patience == 5
        assert len(trainer.training_metrics) == 0
    
    def test_training_arguments_generation(self):
        """Test training arguments generation."""
        config = TrainingConfiguration(output_dir="./test")
        trainer = EvaluationIntegratedTrainer(config)
        
        args = trainer.get_training_arguments()
        
        assert args['output_dir'] == "./test"
        assert 'evaluation_strategy' in args
        assert 'metric_for_best_model' in args
    
    def test_evaluation_logging(self):
        """Test evaluation step logging."""
        config = TrainingConfiguration()
        trainer = EvaluationIntegratedTrainer(config)
        
        # Log some evaluation steps
        should_stop_1 = trainer.log_evaluation_step(1, 50, {'eval_loss': 2.0, 'train_loss': 2.1})
        should_stop_2 = trainer.log_evaluation_step(1, 100, {'eval_loss': 1.8, 'train_loss': 1.9})
        
        assert not should_stop_1  # Shouldn't stop yet
        assert not should_stop_2  # Still improving
        assert len(trainer.training_metrics) == 2
        
        # Verify logged data
        assert trainer.training_metrics[0]['epoch'] == 1
        assert trainer.training_metrics[0]['eval_loss'] == 2.0
    
    def test_training_report_generation(self):
        """Test comprehensive training report generation."""
        config = TrainingConfiguration()
        trainer = EvaluationIntegratedTrainer(config)
        
        # Simulate training progression
        for epoch in range(1, 4):
            for step in [50, 100, 150]:
                loss = 2.0 - (epoch * 0.3) - (step * 0.001)  # Simulated improvement
                trainer.log_evaluation_step(epoch, step, {
                    'eval_loss': loss,
                    'train_loss': loss + 0.1,
                    'perplexity': math.exp(loss)
                })
        
        report = trainer.generate_training_report()
        
        # Verify report structure
        assert 'final_metrics' in report
        assert 'efficiency_metrics' in report
        assert 'performance_trends' in report
        assert 'early_stopping_summary' in report
        
        # Verify calculations
        assert report['efficiency_metrics']['total_steps'] == 9
        assert report['efficiency_metrics']['final_epoch'] == 3
        assert report['performance_trends']['loss_trend'] == 'improving'


class TestFrameworkFactory:
    """Test evaluation framework factory function."""
    
    def test_create_evaluation_framework(self):
        """Test complete framework creation."""
        framework = create_evaluation_framework(
            validation_ratio=0.2,
            early_stopping_patience=4
        )
        
        # Verify all components
        assert 'evaluator' in framework
        assert 'early_stopping' in framework
        assert 'metrics_calculator' in framework
        assert 'split_creator' in framework
        assert 'compute_metrics_fn' in framework
        
        # Verify configuration
        assert framework['evaluator'].validation_ratio == 0.2
        assert framework['early_stopping'].patience == 4
        
        # Verify compute_metrics_fn is callable
        assert callable(framework['compute_metrics_fn'])


class TestEvaluationPerformance:
    """Test evaluation framework performance and edge cases."""
    
    def test_large_corpus_evaluation(self):
        """Test evaluation performance with large corpus."""
        # Create large training dataset
        large_training_data = []
        authors = ['Author A', 'Author B', 'Author C']
        
        for author in authors:
            for i in range(20):  # 20 examples per author
                large_training_data.append({
                    'response': f'Content from {author} example {i}. This is realistic conference-style content.',
                    'metadata': {
                        'author': author,
                        'title': f'Title {i}',
                        'date': '2024-01-01'
                    }
                })
        
        evaluator = TrainingEvaluator(validation_ratio=0.15)
        eval_data = evaluator.prepare_evaluation_datasets(large_training_data)
        
        # Verify handling of large dataset
        assert eval_data['split_analysis']['split_summary']['total_documents'] == 60
        assert len(eval_data['author_corpuses']) == 3
        
        # Test evaluation on subset
        sample_outputs = []
        for author in authors:
            evaluation = evaluator.evaluate_model_output(
                f"Generated text in {author}'s style with faith and testimony.",
                author,
                eval_data['author_corpuses'][author]
            )
            sample_outputs.append({**evaluation, 'author': author})
        
        # Generate report
        report = evaluator.generate_evaluation_report(sample_outputs)
        
        # Verify report completeness
        assert report['summary_metrics']['total_evaluations'] == 3
        assert len(report['author_analysis']) == 3
    
    def test_edge_case_handling(self):
        """Test evaluation framework edge case handling."""
        evaluator = TrainingEvaluator()
        
        # Empty training data
        empty_eval_data = evaluator.prepare_evaluation_datasets([])
        assert empty_eval_data['author_corpuses'] == {}
        
        # Single example
        single_example = [{
            'response': 'Single example content.',
            'metadata': {'author': 'Single Author', 'title': 'Single Title'}
        }]
        
        single_eval_data = evaluator.prepare_evaluation_datasets(single_example)
        assert 'Single Author' in single_eval_data['author_corpuses']
        
        # Evaluation with no reference
        evaluation = evaluator.evaluate_model_output(
            "Generated text without reference.",
            'Unknown Author',
            []
        )
        
        # Should handle gracefully
        assert 'overall_score' in evaluation
        assert 0 <= evaluation['overall_score'] <= 1
    
    def test_metrics_consistency(self):
        """Test consistency of metrics across multiple evaluations."""
        metrics_calc = EvaluationMetrics()
        
        # Same generated text and reference
        generated_text = "Faith is a powerful principle that guides our daily lives."
        reference_corpus = [
            "Faith helps us navigate life's challenges with hope.",
            "The principle of faith is central to gospel living."
        ]
        
        # Run evaluation multiple times
        results = []
        for _ in range(3):
            metrics = metrics_calc.calculate_style_similarity(
                generated_text, reference_corpus, "Test Author"
            )
            results.append(metrics['overall_similarity'])
        
        # Results should be consistent (deterministic)
        assert all(abs(r - results[0]) < 0.001 for r in results)


class TestTrainingConfigIntegration:
    """Test training configuration integration with evaluation."""
    
    def test_evaluation_integrated_trainer_workflow(self):
        """Test complete trainer workflow with evaluation."""
        config = TrainingConfiguration(
            output_dir="./test_output",
            num_train_epochs=2,
            early_stopping_patience=2
        )
        
        trainer = EvaluationIntegratedTrainer(config)
        
        # Simulate training with improving then plateauing loss
        training_logs = [
            {'eval_loss': 2.0, 'train_loss': 2.1},
            {'eval_loss': 1.5, 'train_loss': 1.6},  # Improvement
            {'eval_loss': 1.4, 'train_loss': 1.5},  # Slight improvement
            {'eval_loss': 1.41, 'train_loss': 1.51}, # No improvement, wait_count=1
            {'eval_loss': 1.42, 'train_loss': 1.52}, # No improvement, wait_count=2 >= patience=2, should stop
        ]
        
        stopped = False
        for i, logs in enumerate(training_logs):
            epoch = (i // 2) + 1
            step = (i % 2 + 1) * 50
            should_stop = trainer.log_evaluation_step(epoch, step, logs)
            if should_stop:
                stopped = True
                break
        
        # Should have stopped due to early stopping
        assert stopped
        assert trainer.early_stopping.should_stop
        
        # Generate final report
        report = trainer.generate_training_report()
        assert report['early_stopping_summary']['triggered'] == True
    
    def test_configuration_validation(self):
        """Test configuration validation and defaults."""
        # Test with minimal configuration
        config = TrainingConfiguration()
        
        # Verify reasonable defaults
        assert config.learning_rate == 2e-4
        assert config.gradient_accumulation_steps == 16
        assert config.fp16 == True
        assert config.gradient_checkpointing == True
        
        # Test LoRA configuration
        lora_config = create_lora_config()
        assert lora_config['r'] > 16  # Should be higher than basic
        assert len(lora_config['target_modules']) >= 4  # Should target multiple modules
        
        # Test quantization configuration
        quant_config = create_quantization_config()
        assert quant_config['bnb_4bit_use_double_quant'] == True