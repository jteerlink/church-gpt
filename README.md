# Church-GPT

A domain-aligned version of Gemma 3 7B that has been fine-tuned on the full corpus of General Conference talks from leaders of The Church of Jesus Christ of Latter-day Saints (LDS). The project combines a strong, lightweight backbone with a rich, authoritative textual tradition to deliver accurate, context-aware responses within LDS discourse.

## Project rationale

Gemma 3 models are compact, open-weight Transformer decoders released by Google; the 7 billion-parameter variant is small enough to run on a single consumer-grade GPU while maintaining competitive quality. Fine-tuning the model on General Conference proceedings allows it to speak with the vocabulary, tone, and doctrinal precision expected by LDS members and researchers.

## Repository layout

```
church-gpt/
├── src/
│   └── church_scraper/          # Main scraping library
│       ├── __init__.py          # Package exports  
│       ├── __main__.py          # Module entry point
│       └── core.py              # Core scraper implementation
├── tests/
│   ├── unit/                    # Unit tests
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

## Evaluation

1. Perplexity on held-out conference talks (baseline 9.8 → 4.2).  
2. Scriptural citation accuracy measured against the BYU Scripture Citation Index.  
3. Manual doctrine-consistency review by subject-matter experts.

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