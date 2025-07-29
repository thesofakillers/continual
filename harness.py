from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from custom_trainer import MyCustomTrainer, MyGRPOConfig
from rewards import reward_cumulative_logprob, reward_len, reward_surprisal_moments

MODEL = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, device_map="auto", torch_dtype="auto"
)
full_dataset = load_dataset("trl-lib/tldr", split="train")
initial_dataset = Dataset.from_dict({"prompt": [full_dataset[0]["prompt"]]})


training_args = MyGRPOConfig(
    output_dir=None,
    per_device_train_batch_size=2,
    max_steps=1,  # We train for one step per new data sample
    logging_steps=1,
    beta=0.1,
    num_generations=2,
)

reward_funcs = []
if training_args.use_logprob_reward:
    reward_funcs.append(reward_cumulative_logprob)
if training_args.surprisal_reward_moments:
    reward_funcs.append(reward_surprisal_moments)
if training_args.use_len_reward:
    reward_funcs.append(reward_len)

trainer = MyCustomTrainer(
    model=model,
    args=training_args,
    train_dataset=initial_dataset,
    reward_funcs=reward_funcs,
)


def model_harness(messages: list[dict[str, str]]) -> str:
    tokenized_chat: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    dataset = Dataset.from_dict({"prompt": [tokenized_chat]})

    trainer.train_dataset = dataset
    trainer.train()
    output = trainer.custom_cached_output
    output = output[:]
    reward = trainer.latest_train_metrics["reward"]
    print("TRAINER OUTPUT", output)

    return output, reward
