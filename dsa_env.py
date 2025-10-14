import random
import numpy as np

class DSAEnvironment:
    def __init__(self, num_agents, num_bands, packet_prob, alpha=0.1):
        self.num_agents = num_agents
        self.num_bands = num_bands
        self.packet_prob = packet_prob
        self.alpha = alpha
        self.idle_index = num_bands

        self.probs = np.ones((num_agents, num_bands + 1)) / (num_bands + 1)
        self.queue_lengths = [0] * num_agents
        self.last_actions = [self.idle_index] * num_agents

    def step(self):
        # --- Packet generation ---
        for i in range(self.num_agents):
            if random.random() < self.packet_prob:
                self.queue_lengths[i] += 1

        # --- Choose actions ---
        actions = []
        for i in range(self.num_agents):
            if self.queue_lengths[i] == 0:
                actions.append(self.idle_index)
                continue
            scaled = self.probs[i].copy()
            scaled[self.idle_index] *= max(0.1, 1.0 / (1 + self.queue_lengths[i]))
            scaled[:-1] *= (1 - scaled[self.idle_index]) / self.num_bands
            scaled /= scaled.sum()
            actions.append(np.random.choice(self.num_bands + 1, p=scaled))
        self.last_actions = actions

        # --- Compute rewards ---
        rewards = [0.0] * self.num_agents
        for b in range(self.num_bands):
            users = [i for i, a in enumerate(actions) if a == b]
            if len(users) == 1:
                i = users[0]
                rewards[i] = 1.0
                self.queue_lengths[i] = max(0, self.queue_lengths[i] - 1)

        # --- Update probabilities ---
        for i, a in enumerate(actions):
            if a == self.idle_index:
                continue
            if rewards[i] == 1:
                self.probs[i, a] += self.alpha * (1 - self.probs[i, a])
            else:
                self.probs[i, a] *= (1 - self.alpha)
            self.probs[i] /= self.probs[i].sum()

        return actions, rewards
