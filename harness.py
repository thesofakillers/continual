from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from custom_trainer import MyCustomTrainer
from rewards import reward_len

MODEL = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="auto"
)


training_args = GRPOConfig(
    output_dir=None,
    per_device_train_batch_size=2,
    max_steps=1,  # We train for one step per new data sample
    logging_steps=1,
    beta=0.1,
    num_generations=2,
)

trainer = MyCustomTrainer(
    model=model,
    args=training_args,
    train_dataset=load_dataset("trl-lib/tldr", split="train"),
    reward_funcs=[reward_len],
)


def model_harness(messages: list[dict[str, str]]) -> str:
    tokenized_chat: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    dataset = Dataset.from_dict({"prompt": [tokenized_chat]})

    trainer.train_dataset = dataset
    trainer.train()
    output = trainer.giulio_output
    output = output[:]

    print("TRAINER OUTPUT", output)

    return output
