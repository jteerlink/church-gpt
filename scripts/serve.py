#!/usr/bin/env python3
"""serve.py – Lightweight local inference wrapper for Church‑GPT

This script loads the fine‑tuned Gemma 3 7B checkpoint (optionally with LoRA
adapters) and exposes the model in two ways:
  • **CLI chat** – an interactive REPL in your terminal (default).
  • **FastAPI micro‑service** – JSON endpoint at `/generate` when `--api` flag is
    supplied.

Example usage – interactive shell:
    python serve.py --checkpoint ./checkpoints/gemma3-7b-church

Example usage – HTTP API:
    python serve.py --checkpoint ./checkpoints/gemma3-7b-church --api --port 8000

Requirements (see requirements.txt):
    transformers>=4.40.0
    peft>=0.10.0      # only if using LoRA adapters
    torch>=2.2.0
    fastapi>=0.111.0  # only for --api mode
    uvicorn>=0.29.0   # only for --api mode
"""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextGenerationPipeline,
)

try:
    from peft import PeftModel  # type: ignore
except ImportError:  # PEFT is optional unless LoRA adapters are used
    PeftModel = None  # type: ignore

logger = logging.getLogger("serve")
logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.INFO)

###############################################################################
# Model loading helpers
###############################################################################

def load_pipeline(
    checkpoint: str,
    lora_path: Optional[str] = None,
    fp16: bool = True,
    bf16: bool = False,
    device: str | int | torch.device | None = "auto",
    max_length: int = 4096,
) -> TextGenerationPipeline:
    """Load a Gemma checkpoint (optionally merged with LoRA adapters) and return a
    Hugging Face `TextGenerationPipeline` ready for inference.
    """
    dtype = torch.float32
    if fp16:
        dtype = torch.float16
    elif bf16:
        dtype = torch.bfloat16

    logger.info("Loading base model from %s …", checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    if lora_path:
        if PeftModel is None:
            logger.error("peft not installed – cannot load LoRA adapters.\n"
                         "Run `pip install peft` or omit --lora_path.")
            sys.exit(1)
        logger.info("Applying LoRA adapters from %s …", lora_path)
        model = PeftModel.from_pretrained(model, lora_path)
        # For inference it is more memory‑efficient to merge + unload adapter weights
        try:
            model = model.merge_and_unload()
            logger.info("LoRA adapters merged into base weights.")
        except Exception as exc:
            logger.warning("Could not merge adapters: %s – proceeding with separate weights.", exc)

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.model_max_length = max_length

    pipeline = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=model.device.index if torch.cuda.is_available() else -1,
    )
    return pipeline

###############################################################################
# CLI interactive chat
###############################################################################

def interactive_chat(pipeline: TextGenerationPipeline, max_new_tokens: int, temperature: float, top_p: float) -> None:
    """Simple REPL loop for local conversation."""
    print("\nChurch‑GPT interactive mode. Press Ctrl‑C to exit.\n")
    try:
        while True:
            prompt = input("You: ").strip()
            if not prompt:
                continue
            output = pipeline(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
            )
            generated = output[0]["generated_text"][len(prompt) :].strip()
            print(f"Church‑GPT: {generated}\n")
    except KeyboardInterrupt:
        print("\nExiting …")

###############################################################################
# FastAPI micro‑service
###############################################################################

app = None  # populated in main() if --api flag supplied


def start_api(pipeline: TextGenerationPipeline, host: str, port: int) -> None:
    """Launch a FastAPI server exposing /generate endpoint."""
    from fastapi import FastAPI
    from pydantic import BaseModel
    import uvicorn

    class GenRequest(BaseModel):
        prompt: str
        max_new_tokens: int = 256
        temperature: float = 0.7
        top_p: float = 0.95

    fast_app = FastAPI(title="Church‑GPT API", version="0.1.0")

    @fast_app.post("/generate")
    async def generate(req: GenRequest):  # noqa: D401 – FastAPI callback
        output = pipeline(
            req.prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            do_sample=req.temperature > 0,
        )
        completion = output[0]["generated_text"][len(req.prompt) :].strip()
        return {"response": completion}

    logger.info("Starting FastAPI server on http://%s:%d …", host, port)
    uvicorn.run(fast_app, host=host, port=port, log_level="info")

###############################################################################
# Main entry‑point and CLI
###############################################################################

def parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Local inference server for Church‑GPT (Gemma 3 7B)")
    parser.add_argument("--checkpoint", required=True, help="Path or HF hub ID of base model checkpoint")
    parser.add_argument("--lora_path", help="Optional path (local or HF hub) to LoRA adapters")
    parser.add_argument("--device", default="auto", help="Device identifier passed to device_map (default: auto)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16 weights (use fp32)")
    parser.add_argument("--bf16", action="store_true", help="Load weights in bfloat16 (overrides fp16)")
    # Generation hyper‑parameters
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate per request")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling top‑p")
    # API vs interactive
    parser.add_argument("--api", action="store_true", help="Launch FastAPI JSON server instead of REPL")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    return parser.parse_args()


def main() -> None:  # noqa: D401
    args = parse_args()

    pipeline = load_pipeline(
        checkpoint=args.checkpoint,
        lora_path=args.lora_path,
        fp16=not args.no_fp16,
        bf16=args.bf16,
        device=args.device,
    )

    if args.api:
        start_api(pipeline, host=args.host, port=args.port)
    else:
        interactive_chat(
            pipeline,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )


if __name__ == "__main__":
    main()
