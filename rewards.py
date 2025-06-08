from transformers.trainer import Trainer
import numpy as np


def reward_len(completions, trainer_instance: Trainer, **kwargs):
    """
    A simple reward function that rewards completions based on their length.
    It rewards completions that are close to 50 characters.
    """
    # The 'completions' argument is a list of strings
    reward = [-abs(50 - len(completion)) for completion in completions]
    index = np.argmax(reward)
    trainer_instance.giulio_output = completions[index]
    return reward
