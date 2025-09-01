# Church-GPT

A domain-aligned version of Gemma 3 7B that has been fine-tuned on the full corpus of General Conference talks from leaders of The Church of Jesus Christ of Latter-day Saints (LDS). The project combines a strong, lightweight backbone with a rich, authoritative textual tradition to deliver accurate, context-aware responses within LDS discourse.

## Project rationale

Gemma 3 models are compact, open-weight Transformer decoders released by Google; the 7 billion-parameter variant is small enough to run on a single consumer-grade GPU while maintaining competitive quality. Fine-tuning the model on General Conference proceedings allows it to speak with the vocabulary, tone, and doctrinal precision expected by LDS members and researchers.

## Repository layout

```
church-gpt/
├── src/
│   ├── church_scraper/          # Main scraping library
│   │   ├── __init__.py          # Package exports  
│   │   ├── __main__.py          # Module entry point
│   │   └── core.py              # Core scraper implementation
│   └── data_processing/         # Training and evaluation components
│       ├── prompt_engineering.py        # Basic prompt templates
│       ├── advanced_prompt_engineering.py  # Phase 4: Advanced prompting
│       ├── training_pipeline_optimization.py  # Phase 5: Optimized training
│       ├── training_config.py           # Training configuration
│       └── evaluation.py               # Evaluation framework
├── tests/
│   ├── unit/                    # Unit tests (87 tests total)
│   │   ├── test_advanced_prompt_engineering.py  # 33 tests
│   │   └── test_training_pipeline_optimization.py  # 54 tests
│   └── integration/             # Integration tests
├── scripts/
│   ├── start_scraper.sh         # Interactive scraper CLI
│   ├── start_chat.sh            # Chat interface CLI
│   ├── serve.py                 # Local inference server
│   └── run_tests.py             # Test runner script
├── notebooks/
│   ├── webscrape.ipynb          # End-to-end scraping notebook
│   ├── Gemma_Validation.ipynb   # Model validation
│   └── Gemma3_Fine_Tuning.ipynb # Fine-tuning pipeline
├── docs/
│   └── INTEGRATION_TESTS_README.md # Testing documentation
├── logs/                        # Scraper logs
├── scraped_content/            # Scraped content output
└── pyproject.toml              # Project configuration
```

## Data pipeline

1. **Source authority** – All English-language General Conference talks from 1971 to present were collected together with speaker metadata (name, calling, date, session).  
2. **Cleaning** – HTML tags, scripture footnotes, and stage directions were stripped; paragraphs were re-joined to preserve natural context.  
3. **Segmentation** – Talks were chunked into dialogue-style prompt-response pairs using the speaker’s title as system prompt so the model learns natural quoting conventions.  
4. **Licensing** – Conference content is copyright Intellectual Reserve Inc.; usage here is strictly non-commercial under LDS terms of use.

### Improved Pipeline Features

The enhanced scraping system provides:

- **Robust scraping** – Implements exponential backoff retry logic, rate limiting, and progress tracking for reliable data collection
- **Resumable operations** – Automatically skips existing files, allowing interrupted scraping to resume seamlessly
- **Content organization** – Creates structured directory hierarchies (`scraped_content/general-conference/YYYY-MM/` and `scraped_content/liahona/YYYY-MM/`)
- **Rate limiting** – Respectful 1-second delays between requests (configurable)
- **Progress tracking** – Real-time progress bars with ETA calculations
- **Error recovery** – Exponential backoff for network issues and server errors
- **Logging** – Detailed file and console logging for debugging and monitoring
- **Quality assurance** – Comprehensive integration tests validate complete workflows, directory structure, and error recovery

## Fine-tuning methodology

| Aspect | Setting |
|--------|---------|
| Base model | `google/gemma-3-7b` int8-quantised weights |
| Strategy | LoRA with rank = 16 adapters on attention projection layers |
| Optimiser | AdamW, learning rate 2 × 10⁻⁴ with cosine decay |
| Hardware | 1 × NVIDIA A100-40 GB |
| Data volume | 335 M train tokens / 18 M validation tokens |
| Epochs | 3 full passes (≈ 15 k steps) |
| Runtime | ≈ 9 h |

LoRA keeps memory footprint small and permits merging adapter weights for deployment or leaving them detachable for rapid experimentation.

### Advanced Training Features

**Phase 4: Advanced Prompt Engineering**
- **Speaker-specific adaptation** – Custom prompt templates for Russell M. Nelson, Dieter F. Uchtdorf, and Jeffrey R. Holland with unique rhetorical patterns
- **Conversation-style templates** – Interview and dialogue scenarios for natural interaction patterns
- **Difficulty-aware selection** – Progressive complexity levels (Simple → Advanced) for optimal training progression
- **Template validation system** – Comprehensive validation with fallback mechanisms and context-aware formatting

**Phase 5: Training Pipeline Optimization**
- **Memory-efficient data loading** – Lazy loading with intelligent caching and memory mapping support
- **Adaptive gradient accumulation** – Dynamic adjustment based on memory usage with 4 optimization levels
- **Mixed precision training** – fp16/bf16 support with automatic loss scaling and gradient management
- **Comprehensive monitoring** – Real-time performance profiling, memory tracking, and metrics collection
- **Resume capability** – Full checkpoint management with automatic training state restoration

### Using Advanced Training Features

```python
# Advanced prompt engineering with speaker adaptation
from src.data_processing.advanced_prompt_engineering import AdvancedPromptEngine, DifficultyLevel

engine = AdvancedPromptEngine()
prompt = engine.generate_prompt(
    content="Faith is a principle of action",
    speaker="Russell M. Nelson",
    difficulty=DifficultyLevel.COMPLEX,
    mode="conversation"
)

# Optimized training pipeline with memory efficiency
from src.data_processing.training_pipeline_optimization import create_training_pipeline, OptimizationLevel

pipeline = create_training_pipeline(
    optimization_level=OptimizationLevel.AGGRESSIVE,
    checkpoint_dir="./checkpoints",
    metrics_dir="./metrics"
)

# Memory-efficient dataset and dataloader
dataset = pipeline.create_optimized_dataloader(training_data, batch_size=8)
scaler, autocast_dtype = pipeline.setup_mixed_precision()

# Training loop with optimization
for batch in dataset:
    with torch.autocast(device_type='cuda', dtype=autocast_dtype):
        loss = model(batch)
    
    should_update = pipeline.optimize_step(loss, model, optimizer, scaler)
    
    if pipeline.should_checkpoint():
        pipeline.checkpoint_manager.save_checkpoint(
            state=pipeline.training_state,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )
```

### Complete Training Workflow

For a full end-to-end training workflow with all advanced features:

```python
# complete_training_example.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from src.data_processing.advanced_prompt_engineering import AdvancedPromptEngine, DifficultyLevel
from src.data_processing.training_pipeline_optimization import create_training_pipeline, OptimizationLevel
from src.data_processing.evaluation import create_evaluation_framework

def main():
    # 1. Setup advanced prompt engineering
    prompt_engine = AdvancedPromptEngine()
    
    # 2. Create optimized training pipeline
    pipeline = create_training_pipeline(
        optimization_level=OptimizationLevel.AGGRESSIVE,
        checkpoint_dir="./checkpoints/advanced",
        metrics_dir="./metrics/advanced"
    )
    
    # 3. Setup evaluation framework
    eval_framework = create_evaluation_framework(
        validation_ratio=0.2,
        early_stopping_patience=3
    )
    
    # 4. Load and prepare model
    model_name = "google/gemma-2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 5. Apply LoRA configuration
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    # 6. Prepare training data with advanced prompts
    # Load your conference talk data here
    training_data = []  # Your scraped conference data
    
    # Generate advanced prompts for each example
    processed_data = []
    for example in training_data:
        prompt = prompt_engine.generate_prompt(
            content=example['content'],
            speaker=example.get('speaker', 'Unknown'),
            difficulty=DifficultyLevel.MODERATE,
            mode="completion"
        )
        processed_data.append({
            'input_text': prompt,
            'output_text': example['response']
        })
    
    # 7. Create memory-efficient dataset
    dataset = pipeline.create_optimized_dataloader(processed_data, batch_size=4)
    
    # 8. Setup mixed precision training
    scaler, autocast_dtype = pipeline.setup_mixed_precision()
    
    # 9. Training loop with optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    
    model.train()
    for epoch in range(3):
        for batch_idx, batch in enumerate(dataset):
            # Forward pass with mixed precision
            with torch.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(**batch)
                loss = outputs.loss
            
            # Optimized backward pass with gradient accumulation
            should_update = pipeline.optimize_step(loss, model, optimizer, scaler)
            
            # Log metrics
            if batch_idx % 10 == 0:
                pipeline.log_training_metrics({
                    'loss': loss.item(),
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'epoch': epoch,
                    'batch': batch_idx
                })
            
            # Save checkpoints
            if pipeline.should_checkpoint():
                pipeline.checkpoint_manager.save_checkpoint(
                    state=pipeline.training_state,
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    extra_data={'epoch': epoch, 'batch': batch_idx}
                )
    
    # 10. Generate training report
    report = pipeline.get_optimization_report()
    print("Training completed successfully!")
    print(f"Optimization level: {report['optimization_level']}")

if __name__ == "__main__":
    main()
```

### Quick Inference Examples

Simple inference examples for common use cases:

```python
# quick_inference.py
from src.data_processing.advanced_prompt_engineering import AdvancedPromptEngine, DifficultyLevel

# Initialize prompt engine
engine = AdvancedPromptEngine()

# Generate different types of prompts
examples = [
    {
        'content': 'Faith is the foundation of all righteousness',
        'speaker': 'Russell M. Nelson',
        'difficulty': DifficultyLevel.COMPLEX,
        'mode': 'completion'
    },
    {
        'content': 'How do we develop greater faith?',
        'speaker': 'Dieter F. Uchtdorf',
        'difficulty': DifficultyLevel.MODERATE,
        'mode': 'conversation'
    },
    {
        'content': 'The importance of scripture study',
        'speaker': 'Jeffrey R. Holland',
        'difficulty': DifficultyLevel.SIMPLE,
        'mode': 'instruction'
    }
]

for example in examples:
    prompt = engine.generate_prompt(**example)
    print(f"\n{example['speaker']} ({example['mode']}):")
    print(prompt)
    print("-" * 50)
```

## Evaluation

1. **Perplexity on held-out conference talks** (baseline 9.8 → 4.2).  
2. **Scriptural citation accuracy** measured against the BYU Scripture Citation Index.  
3. **Manual doctrine-consistency review** by subject-matter experts.
4. **Comprehensive test coverage** – 87 unit tests covering advanced prompt engineering, training optimization, evaluation framework, and integration scenarios.

## System Requirements

### For Training Pipeline
- **Python**: 3.9+ (automatically managed by `uv`)
- **GPU**: NVIDIA GPU with 16GB+ VRAM recommended for full model training
- **Memory**: 32GB+ RAM recommended for aggressive optimization levels
- **Storage**: 50GB+ for checkpoints and metrics
- **Dependencies**: PyTorch 2.0+, Transformers, PEFT, BitsAndBytes

### For Inference Only
- **GPU**: 8GB+ VRAM for fp16 inference, 4GB+ for 4-bit quantization
- **Memory**: 16GB+ RAM recommended
- **Dependencies**: PyTorch, Transformers, PEFT

## Quick-start

### Data Collection Pipeline

The project uses `uv` for fast, reliable Python package management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# or: brew install uv

# Install scraping dependencies
uv add requests beautifulsoup4

# Scrape General Conference talks (1995-present)
uv run python -m src.church_scraper --content-type conference --start-year 1995

# Scrape Liahona articles (2008-present, excluding conference months)
uv run python -m src.church_scraper --content-type liahona --start-year 2008

# Scrape both content types with custom settings
uv run python -m src.church_scraper --start-year 2020 --delay 1.5 --verbose

# Use the CLI helper script (handles uv setup automatically)
./scripts/start_scraper.sh

# Run tests
python scripts/run_tests.py           # All tests
python scripts/run_tests.py --unit    # Unit tests only
python scripts/run_tests.py --integration  # Integration tests only
```

### Model Serving

```bash
# Install serving dependencies
uv add torch transformers peft fastapi uvicorn

# Interactive chat mode (default)
uv run python scripts/serve.py --checkpoint ./checkpoints/gemma3-7b-church

# API server mode
uv run python scripts/serve.py --checkpoint ./checkpoints/gemma3-7b-church --api --port 8000

# Use the CLI helper script (handles uv setup automatically)
./scripts/start_chat.sh
```

### Testing and Validation

```bash
# Install test dependencies
uv add pytest pytest-cov

# Run all tests with the test runner
python scripts/run_tests.py

# Run comprehensive integration tests
uv run python tests/integration/run_integration_tests.py

# Validate integration test coverage
uv run python tests/integration/validate_integration_tests.py

# Run specific test suites with pytest
uv run pytest tests/unit/test_conference_scraper.py -v
uv run pytest tests/unit/test_liahona_scraper.py -v
uv run pytest tests/unit/test_advanced_prompt_engineering.py -v  # 33 tests
uv run pytest tests/unit/test_training_pipeline_optimization.py -v  # 54 tests
```

### Training Pipeline

Start the optimized training pipeline with advanced features:

```bash
# Install training dependencies
uv add torch transformers peft datasets accelerate bitsandbytes

# Basic training with balanced optimization
python -c "
from src.data_processing.training_pipeline_optimization import create_training_pipeline, OptimizationLevel
from src.data_processing.advanced_prompt_engineering import AdvancedPromptEngine

# Create optimized pipeline
pipeline = create_training_pipeline(
    optimization_level=OptimizationLevel.BALANCED,
    checkpoint_dir='./checkpoints',
    metrics_dir='./metrics'
)

# Load and prepare training data
# (Add your training data loading logic here)
print('Training pipeline initialized successfully')
"

# Advanced training with maximum optimization
python -c "
from src.data_processing.training_pipeline_optimization import create_training_pipeline, OptimizationLevel

# Maximum optimization for enterprise training
pipeline = create_training_pipeline(
    optimization_level=OptimizationLevel.MAXIMUM,
    checkpoint_dir='./checkpoints/optimized',
    metrics_dir='./metrics/optimized'
)

# Setup mixed precision training
scaler, autocast_dtype = pipeline.setup_mixed_precision()
print(f'Mixed precision setup: {autocast_dtype}')

# Check resume capability
if pipeline.resume_training():
    print('Resumed from existing checkpoint')
else:
    print('Starting fresh training')
"

# Monitor training progress
python -c "
from src.data_processing.training_pipeline_optimization import create_training_pipeline

pipeline = create_training_pipeline()
report = pipeline.get_optimization_report()
print('Optimization Report:', report['optimization_level'])
"
```

### Model Inference

Run inference with the fine-tuned model:

```bash
# Install inference dependencies
uv add torch transformers peft

# Interactive chat mode (recommended for testing)
uv run python scripts/serve.py --checkpoint ./checkpoints/gemma3-7b-church

# API server mode for applications
uv run python scripts/serve.py --checkpoint ./checkpoints/gemma3-7b-church --api --port 8000

# Use the CLI helper script (handles uv setup automatically)
./scripts/start_chat.sh

# Advanced inference with prompt engineering
python -c "
from src.data_processing.advanced_prompt_engineering import AdvancedPromptEngine, DifficultyLevel

engine = AdvancedPromptEngine()

# Generate speaker-specific prompt
prompt = engine.generate_prompt(
    content='Faith is a principle of action and power',
    speaker='Russell M. Nelson',
    difficulty=DifficultyLevel.COMPLEX,
    mode='conversation'
)

print('Generated prompt:', prompt)
"

# Programmatic inference (example)
python -c "
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Load base model and tokenizer
model_name = 'google/gemma-2-7b'
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

# Load fine-tuned adapter
model = PeftModel.from_pretrained(base_model, './checkpoints/gemma3-7b-church')

# Generate response
prompt = 'Faith is a principle that'
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=200, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
"
```

### CLI Helper Scripts

The project includes interactive bash scripts for easier operation:

**Content Scraping (`scripts/start_scraper.sh`)**:
- Interactive menu for selecting content types and date ranges
- Quick presets for common scraping scenarios
- Automatic `uv` installation and dependency management
- Progress tracking and error handling

**Model Chat Interface (`scripts/start_chat.sh`)**:
- Automatic model checkpoint detection
- Interactive and API server modes
- Configurable generation parameters
- Easy setup for local inference with `uv`

Both scripts provide `--help` options and support direct command-line usage for automation. They automatically handle `uv` installation and project setup.

## Troubleshooting

### Installing uv

If `uv` is not installed, the CLI scripts will attempt to install it automatically. For manual installation:

```bash
# Using curl (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using Homebrew on macOS
brew install uv

# Using pip (if needed)
pip install uv
```

### Dependency Management

`uv` automatically manages virtual environments and dependencies:

```bash
# Add dependencies as needed
uv add requests beautifulsoup4     # Scraping dependencies
uv add torch transformers peft     # Model serving dependencies
uv add fastapi uvicorn             # API server dependencies
uv add pytest pytest-cov          # Testing dependencies

# Run Python with managed environment
uv run python script.py
```

### Common Issues

**Project not initialized**: The CLI scripts automatically run `uv init` if needed.

**Missing dependencies**: Use `uv sync` to install all dependencies or let the CLI scripts handle it automatically.

**Python version**: `uv` automatically manages Python versions. The project requires Python 3.9+.

**GPU support for PyTorch**: For CUDA support, run:
```bash
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Responsible use

Outputs must not be presented as official statements of the Church. Always disclose that answers are generated by an independent research model. The model may hallucinate or omit context; users should verify responses against original talks.

## Roadmap
	•	Quantised 4-bit QLoRA branch for laptop-class inference.
	•	Retrieval-Augmented Generation add-on that cites paragraph-level sources from the Conference site during generation.
	•	Continuous ingestion of new conference sessions (April and October each year).

## Acknowledgements

This work builds on Gemma open-weights and tutorials from Google DeepMind, community LoRA recipes, and open General Conference datasets maintained by the LDS community.