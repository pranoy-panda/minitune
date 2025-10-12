import torch
from torch.utils.data import Dataset

class RolloutBuffer(Dataset):
    """
    Buffer to store trajectories from the PPO rollout phase.
    """
    def __init__(self):
        self.prompts: list[torch.Tensor] = []
        self.responses: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.rewards: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def add(self, prompt, response, logprob, reward, value):
        self.prompts.append(prompt)
        self.responses.append(response)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.values.append(value)

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            self.prompts[idx],
            self.responses[idx],
            self.logprobs[idx],
            self.rewards[idx],
            self.values[idx],
        )

    def clear(self):
        self.__init__()