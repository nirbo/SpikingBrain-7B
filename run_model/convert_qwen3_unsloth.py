#!/usr/bin/env python
"""Convert Qwen3-14B weights to the SpikingBrain HF hybrid model and optionally fine-tune with Unsloth."""

import unsloth
import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch
from safetensors.torch import load_file

from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

from hf_7B_model import GLAswaConfig, GLAswaForCausalLM
from unsloth import UnslothTrainer, UnslothTrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen3-path", required=True, help="Path to the downloaded Qwen3-14B checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save the converted SpikingBrain model")
    parser.add_argument("--dataset", default=None, help="Optional Hugging Face dataset name (e.g. 'tatsu-lab/alpaca') for immediate fine-tuning")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to use when fine-tuning")
    parser.add_argument("--text-column", default=None, help="Column containing plain text when not using chat-style data")
    parser.add_argument("--max-seq-length", type=int, default=4096, help="Sequence length for training")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps (Unsloth will fuse optimisations)")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Base learning rate for fine-tuning")
    parser.add_argument("--num-train-epochs", type=float, default=1.0, help="Number of epochs for fine-tuning")
    parser.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval during fine-tuning")
    parser.add_argument("--logging-steps", type=int, default=50, help="Logging interval during fine-tuning")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 during fine-tuning (recommended on recent GPUs)")
    parser.add_argument("--no-convert", action="store_true", help="Skip weight conversion if converted weights are already present in output-dir")
    parser.add_argument("--train", action="store_true", help="Run Unsloth fine-tuning after conversion")
    parser.add_argument("--device", default="cpu", help="Device to place converted model on (e.g. 'cuda', 'cpu')")
    parser.add_argument("--extra-dataset-args", default=None, help="JSON string of extra keyword arguments for datasets.load_dataset (e.g. data_files)")
    parser.add_argument("--optim", default="adamw_torch", help="Optimizer name for UnslothTrainingArguments (e.g. 'adamw_8bit')")
    parser.add_argument("--load-in-4bit", action="store_true", help="Enable 4-bit loading via bitsandbytes")
    parser.add_argument("--use-lora", action="store_true", help="Apply LoRA adapters instead of full fine-tuning")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--use-rslora", action="store_true", help="Enable rank-stabilized LoRA if supported")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=None, help="Evaluation batch size; defaults to train batch size if unset")
    return parser.parse_args()


def load_tokenizer(qwen_path: str, source_config: AutoConfig) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(qwen_path, use_fast=True)
    # Ensure special tokens exist as expected by our config
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = source_config.bos_token_id
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = source_config.eos_token_id

    current_size = len(tokenizer)
    target_size = source_config.vocab_size
    if current_size < target_size:
        extra_tokens = target_size - current_size
        tokenizer.add_tokens([f"<extra_token_{i}>" for i in range(extra_tokens)])

    return tokenizer


def build_config_from_qwen(source_config: AutoConfig) -> GLAswaConfig:
    return GLAswaConfig(
        vocab_size=source_config.vocab_size,
        hidden_size=source_config.hidden_size,
        num_hidden_layers=source_config.num_hidden_layers,
        num_attention_heads=source_config.num_attention_heads,
        num_key_value_heads=source_config.num_key_value_heads,
        intermediate_size=source_config.intermediate_size,
        max_position_embeddings=source_config.max_position_embeddings,
        bos_token_id=source_config.bos_token_id,
        eos_token_id=source_config.eos_token_id,
        enable_spike=False,
    )


def _group_keys_by_file(index_path: str) -> Dict[str, Iterable[str]]:
    with open(index_path, "r", encoding="utf-8") as handle:
        index_json = json.load(handle)
    file_groups: Dict[str, list[str]] = defaultdict(list)
    for param_name, shard_name in index_json["weight_map"].items():
        file_groups[shard_name].append(param_name)
    return file_groups


def _map_qwen_key_to_spikingbrain(key: str) -> Optional[str]:
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key == "model.embed_tokens.weight":
        return "model.embeddings.weight"
    if key == "model.norm.weight":
        return "model.norm.weight"

    if not key.startswith("model.layers."):
        return None

    parts = key.split(".")
    layer_idx = int(parts[2])
    suffix = parts[3]

    if suffix == "input_layernorm":
        return f"model.layers.{layer_idx}.attn_norm.weight"
    if suffix == "post_attention_layernorm":
        return f"model.layers.{layer_idx}.mlp_norm.weight"

    if suffix == "self_attn":
        attn_key = parts[4]
        mapping = {
            "q_proj": "attn.q_proj.weight",
            "k_proj": "attn.k_proj.weight",
            "v_proj": "attn.v_proj.weight",
            "o_proj": "attn.o_proj.weight",
        }
        if attn_key in ("q_norm", "k_norm", "rotary_emb"):
            return None
        if attn_key not in mapping:
            return None
        return f"model.layers.{layer_idx}.{mapping[attn_key]}"

    if suffix == "mlp":
        mlp_key = parts[4]
        mapping = {
            "gate_proj": "mlp.gate_proj.weight",
            "up_proj": "mlp.up_proj.weight",
            "down_proj": "mlp.down_proj.weight",
        }
        if mlp_key not in mapping:
            return None
        return f"model.layers.{layer_idx}.{mapping[mlp_key]}"

    return None


def copy_qwen_to_spikingbrain(qwen_dir: str, target_model: GLAswaForCausalLM) -> None:
    state_dict = target_model.state_dict()
    groups = _group_keys_by_file(os.path.join(qwen_dir, "model.safetensors.index.json"))

    for shard, keys in groups.items():
        shard_path = os.path.join(qwen_dir, shard)
        LOGGER.info("Loading shard %s", shard_path)
        shard_tensors = load_file(shard_path, device="cpu")

        for key in keys:
            mapped_key = _map_qwen_key_to_spikingbrain(key)
            if mapped_key is None:
                continue
            if mapped_key not in state_dict:
                LOGGER.warning("Target key %s not found in hybrid model; skipping", mapped_key)
                continue
            source_tensor = shard_tensors[key]
            dest_param = state_dict[mapped_key]
            if source_tensor.shape != dest_param.shape:
                LOGGER.warning(
                    "Shape mismatch for %s -> %s (%s vs %s); skipping",
                    key,
                    mapped_key,
                    tuple(source_tensor.shape),
                    tuple(dest_param.shape),
                )
                continue
            dest_param.copy_(source_tensor.to(dest_param.dtype))

    target_model.load_state_dict(state_dict, strict=False)


def save_converted_model(model: GLAswaForCausalLM, tokenizer: AutoTokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    LOGGER.info("Converted model saved to %s", output_dir)


def prepare_unsloth_trainer(
    model: GLAswaForCausalLM,
    tokenizer: AutoTokenizer,
    dataset_name: str,
    dataset_split: str,
    text_column: str,
    args: argparse.Namespace,
    extra_dataset_kwargs: Optional[dict],
    use_bf16: bool,
    use_fp16: bool,
    eval_batch_size: int,
) -> UnslothTrainer:
    dataset_kwargs = extra_dataset_kwargs or {}
    dataset = load_dataset(dataset_name, split=dataset_split, **dataset_kwargs)
    original_columns = dataset.column_names

    def format_example(example):
        if "conversations" in example:
            chunks = []
            for turn in example["conversations"]:
                role = turn.get("from", "")
                speaker = "User" if role == "human" else "Assistant"
                chunks.append(f"{speaker}: {turn.get('value', '').strip()}")
            text = "\n".join(chunks)
        else:
            if text_column is None:
                raise ValueError("text_column must be provided when dataset entries are not chat-style")
            text = example[text_column]
        if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
            text = text + tokenizer.eos_token
        return {"text": text}

    dataset = dataset.map(
        format_example,
        remove_columns=original_columns,
        num_proc=30,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = UnslothTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        per_device_eval_batch_size=eval_batch_size,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing="unsloth",
        optim=args.optim,
        report_to="none",
        max_steps=-1,
    )

    # Unsloth patching expects FastQwen3Model style modules for some features. We keep tokenizer consistent
    trainer = UnslothTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    source_config = AutoConfig.from_pretrained(args.qwen3_path)
    tokenizer = load_tokenizer(args.qwen3_path, source_config)
    extra_dataset_kwargs = json.loads(args.extra_dataset_args) if args.extra_dataset_args else None

    if not args.no_convert:
        LOGGER.info("Instantiating hybrid model with Qwen3-compatible configuration")
        config = build_config_from_qwen(source_config)
        config.vocab_size = len(tokenizer)
        device = torch.device(args.device)
        hybrid_model = GLAswaForCausalLM(config)
        hybrid_model.resize_token_embeddings(len(tokenizer))
        hybrid_model = hybrid_model.to(device)

        LOGGER.info("Copying Qwen3 weights into the hybrid model")
        copy_qwen_to_spikingbrain(args.qwen3_path, hybrid_model)

        save_converted_model(hybrid_model, tokenizer, args.output_dir)
        del hybrid_model
        torch.cuda.empty_cache()
    else:
        LOGGER.info("Skipping conversion because --no-convert was set")

    if args.train:
        if args.dataset is None:
            raise ValueError("--dataset must be provided when --train is specified")

        LOGGER.info("Loading converted model from %s for Unsloth fine-tuning", args.output_dir)

        if args.load_in_4bit:
            if args.device != "cuda":
                raise ValueError("4-bit loading requires --device cuda")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
            model = GLAswaForCausalLM.from_pretrained(
                args.output_dir,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            device = torch.device(args.device)
            model = GLAswaForCausalLM.from_pretrained(
                args.output_dir,
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
            ).to(device)
        use_bf16 = args.bf16
        use_fp16 = not args.bf16

        if args.use_lora:
            try:
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            except ImportError as exc:
                raise ImportError("peft is required for LoRA. Install with `pip install peft`." ) from exc

            target_modules = [
                "attn.q_proj",
                "attn.k_proj",
                "attn.v_proj",
                "attn.o_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ]

            if args.load_in_4bit:
                model = prepare_model_for_kbit_training(model)

            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
                use_rslora=args.use_rslora,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        eval_batch_size = args.per_device_eval_batch_size or args.per_device_train_batch_size
        trainer = prepare_unsloth_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset_name=args.dataset,
            dataset_split=args.dataset_split,
            text_column=args.text_column,
            args=args,
            extra_dataset_kwargs=extra_dataset_kwargs,
            use_bf16=use_bf16,
            use_fp16=use_fp16,
            eval_batch_size=eval_batch_size,
        )
        trainer.train()
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
