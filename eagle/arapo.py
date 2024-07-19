"""
# regular:
python eagle/arapo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-6 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-aligned-cpo" \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns

# peft:
python eagle/arapo.py \
    --model_name_or_path=gpt2 \
    --per_device_train_batch_size 4 \
    --max_steps 1000 \
    --learning_rate 8e-5 \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir="gpt2-lora-aligned-cpo" \
    --optim rmsprop \
    --warmup_steps 150 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r=16 \
    --lora_alpha=16
"""

import multiprocessing
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import ModelConfig, get_peft_config

from .arapo_trainer import AraPOTrainer, AraPOConfig

@dataclass
class ScriptArguments:
    dataset: str = field(
        default="trl-internal-testing/hh-rlhf-helpful-base-trl-style",
        metadata={"help": "The name of the dataset to use."},
    )


if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, AraPOConfig, ModelConfig))
    args, arapo_args, model_config = parser.parse_args_into_dataclasses()

    ################
    # Model & Tokenizer
    ################
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    ds = load_dataset(args.dataset)
    if arapo_args.debug:
        for key in ds:
            ds[key] = ds[key].select(range(50))
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = ds.map(
        process,
        num_proc=1 if arapo_args.debug else multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    ################
    # Training
    ################
    trainer = AraPOTrainer(
        model,
        args=arapo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
    )

    # train and save the model
    trainer.train()
    trainer.save_model(arapo_args.output_dir)