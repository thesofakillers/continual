import time
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


def reward_len(completions, **kwargs):
    """
    A simple reward function that rewards completions based on their length.
    It rewards completions that are close to 50 characters.
    """
    # The 'completions' argument is a list of strings
    return [-abs(50 - len(completion)) for completion in completions]


def main():
    """
    Main function to run the GRPO training loop, simulating a stream of data.
    """
    model_name = "HuggingFaceTB/SmolLM-135M"

    # Load the full dataset from Hugging Face Hub
    full_dataset = load_dataset("trl-lib/tldr", split="train")

    # We will iterate through the dataset to simulate a stream of data
    # but let's just take a small slice for this example to run faster.
    full_dataset = full_dataset.select(range(10))

    training_args = GRPOConfig(
        output_dir="./grpo_trainer_output",
        per_device_train_batch_size=2,
        max_steps=1,  # We train for one step per new data sample
        logging_steps=1,
        beta=0.1,
        num_generations=2,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name)

    # GRPOTrainer instantiation
    # We initialize it with a dummy dataset, as it will be replaced in the loop.
    # The trainer requires a train_dataset on init.
    initial_dataset = Dataset.from_dict({"prompt": [full_dataset[0]["prompt"]]})

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=initial_dataset,
        reward_funcs=[reward_len],
        # generation_kwargs=generation_kwargs,
    )

    print("Starting training loop, simulating a data stream...")
    # We start from the second sample since we used the first for initialization
    for i, sample in enumerate(full_dataset.select(range(1, len(full_dataset)))):
        print(f"\n--- Loop {i + 1} ---")

        # Create a new dataset with the single new sample
        # The trainer expects a dataset with a 'prompt' column
        new_data_dict = {"prompt": [sample["prompt"]]}
        new_dataset = Dataset.from_dict(new_data_dict)
        print("Generated new data from stream.")

        # Update the trainer's dataset
        trainer.train_dataset = new_dataset
        print("Updated trainer's dataset.")

        print("Calling trainer.train()...")
        trainer.train()
        print("trainer.train() finished.")

        print("Loop finished. Moving to next sample.")


if __name__ == "__main__":
    main()
