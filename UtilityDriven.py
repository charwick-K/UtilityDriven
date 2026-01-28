# utility_dqn.py
import random
import math
import time
import heapq
from collections import deque, namedtuple
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# ---------------------------
# Transition and utilities
# ---------------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ---------------------------
# Utility Replay Buffer
# ---------------------------
class UtilityReplayBuffer:
    """
    Maintains a min-heap keyed by utility Ut = alpha * |reward| + beta * |td_error|.
    Sampling is uniform over retained transitions.
    Logs retention/eviction events for interpretability.
    """
    def __init__(self, capacity: int, alpha: float = 0.5, beta: float = 0.5):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.heap = []  # list of (utility, idx) for min-heap
        self.data = {}  # idx -> (utility, transition)
        self.next_idx = 0
        self.eviction_log = []  # list of (step, evicted_utility)
        self.retention_log = []  # list of (step, retained_utility)

    def __len__(self):
        return len(self.data)

    def _push_heap(self, utility: float, idx: int):
        heapq.heappush(self.heap, (utility, idx))

    def _pop_heap(self):
        # pop until we find a valid entry (not stale)
        while self.heap:
            utility, idx = heapq.heappop(self.heap)
            if idx in self.data and math.isclose(self.data[idx][0], utility, rel_tol=1e-9):
                return utility, idx
        return None, None

    def add(self, transition: Transition, utility: float, step: int):
        """
        Insert transition with computed utility. If capacity exceeded, evict lowest-utility.
        """
        idx = self.next_idx
        self.next_idx += 1
        self.data[idx] = (utility, transition)
        self._push_heap(utility, idx)
        self.retention_log.append((step, float(utility)))
        if len(self.data) > self.capacity:
            ev_util, ev_idx = self._pop_heap()
            if ev_idx is not None:
                # remove evicted
                self.eviction_log.append((step, float(ev_util)))
                del self.data[ev_idx]

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Uniform sampling over retained transitions.
        """
        if len(self.data) == 0:
            return []
        idxs = random.sample(list(self.data.keys()), k=min(batch_size, len(self.data)))
        return [self.data[i][1] for i in idxs]

    def get_stats(self) -> Dict[str, Any]:
        utilities = [u for u, _ in self.data.values()]
        return {
            'size': len(self.data),
            'mean_utility': float(np.mean(utilities)) if utilities else 0.0,
            'min_utility': float(np.min(utilities)) if utilities else 0.0,
            'max_utility': float(np.max(utilities)) if utilities else 0.0,
            'evictions': len(self.eviction_log),
            'retentions': len(self.retention_log)
        }

    def clear_logs(self):
        self.eviction_log.clear()
        self.retention_log.clear()

# ---------------------------
# DQN network
# ---------------------------
class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------
# Agent
# ---------------------------
class DQNAgent:
    def __init__(self,
                 env: gym.Env,
                 buffer_capacity: int = 50000,
                 batch_size: int = 64,
                 gamma: float = 0.99,
                 lr: float = 1e-3,
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 device: str = 'cpu'):
        self.env = env
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n
        self.device = torch.device(device)
        self.policy_net = DQN(obs_dim, act_dim).to(self.device)
        self.target_net = DQN(obs_dim, act_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = UtilityReplayBuffer(capacity=buffer_capacity, alpha=alpha, beta=beta)
        self.batch_size = batch_size
        self.gamma = gamma
        self.steps_done = 0
        self.epsilon_start = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 100000  # steps
        self.update_target_every = 1000  # steps
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray) -> int:
        eps = self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
              math.exp(-1.0 * self.steps_done / self.epsilon_decay)
        self.steps_done += 1
        if random.random() < eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q = self.policy_net(s)
                return int(q.argmax(dim=1).item())

    def compute_td_error(self, transition: Transition) -> float:
        """
        Compute TD error for a single transition using current policy and target networks.
        """
        s = torch.tensor(transition.state, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = torch.tensor([transition.action], dtype=torch.long).to(self.device)
        r = torch.tensor([transition.reward], dtype=torch.float32).to(self.device)
        ns = torch.tensor(transition.next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
        done = transition.done

        with torch.no_grad():
            q_val = self.policy_net(s).gather(1, a.unsqueeze(1).long()).squeeze(1)
            q_next = self.target_net(ns).max(1)[0]
            q_target = r + (0.0 if done else self.gamma * q_next)
            td_error = (q_target - q_val).cpu().item()
        return float(td_error)

    def compute_utility(self, transition: Transition) -> float:
        td_err = abs(self.compute_td_error(transition))
        reward_mag = abs(transition.reward)
        return float(self.replay.alpha * reward_mag + self.replay.beta * td_err)

    def store_transition(self, transition: Transition, step: int):
        util = self.compute_utility(transition)
        self.replay.add(transition, util, step)

    def optimize_model(self):
        batch = self.replay.sample(self.batch_size)
        if len(batch) < 1:
            return None
        # convert to tensors
        states = torch.tensor(np.stack([t.state for t in batch]), dtype=torch.float32).to(self.device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.stack([t.next_state for t in batch]), dtype=torch.float32).to(self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32).unsqueeze(1).to(self.device)

        # current Q
        q_values = self.policy_net(states).gather(1, actions)
        # target Q
        with torch.no_grad():
            q_next = self.target_net(next_states).max(1)[0].unsqueeze(1)
            q_target = rewards + (1.0 - dones) * (self.gamma * q_next)

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# ---------------------------
# Training loop
# ---------------------------
def train(env_name: str = 'CartPole-v1',
          num_episodes: int = 500,
          buffer_capacity: int = 50000,
          batch_size: int = 64,
          gamma: float = 0.99,
          lr: float = 1e-3,
          alpha: float = 0.5,
          beta: float = 0.5,
          seed: int = 42,
          device: str = 'cpu'):
    set_seed(seed)
    env = gym.make(env_name)
    env.seed(seed)
    agent = DQNAgent(env, buffer_capacity=buffer_capacity, batch_size=batch_size,
                     gamma=gamma, lr=lr, alpha=alpha, beta=beta, device=device)

    rewards_history = []
    mean_rewards = []
    losses = []
    start_time = time.time()
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        ep_reward = 0.0
        done = False
        step_in_ep = 0
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            t = Transition(state, action, reward, next_state, done)
            # store with utility computed using current networks
            agent.store_transition(t, step=total_steps)
            # optimize
            loss = agent.optimize_model()
            if loss is not None:
                losses.append(loss)
            # periodic target update
            if total_steps % agent.update_target_every == 0:
                agent.update_target()
            state = next_state
            step_in_ep += 1
            total_steps += 1

        rewards_history.append(ep_reward)
        if ep % 10 == 0:
            mean_r = float(np.mean(rewards_history[-50:]))
            mean_rewards.append(mean_r)
            stats = agent.replay.get_stats()
            print(f"Ep {ep:03d} | MeanR(50): {mean_r:.2f} | BufferSize: {stats['size']} | MeanUtil: {stats['mean_utility']:.4f} | Evictions: {stats['evictions']} | Time: {time.time()-start_time:.1f}s")

    env.close()
    return {
        'agent': agent,
        'rewards': rewards_history,
        'losses': losses,
        'mean_rewards': mean_rewards,
        'duration_s': time.time() - start_time
    }

# ---------------------------
# Evaluation helper
# ---------------------------
def evaluate_agent(agent: DQNAgent, env_name: str, episodes: int = 20, seed: int = 0):
    env = gym.make(env_name)
    env.seed(seed)
    returns = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total = 0.0
        while not done:
            with torch.no_grad():
                st = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(agent.device)
                a = int(agent.policy_net(st).argmax(dim=1).item())
            s, r, done, _ = env.step(a)
            total += r
        returns.append(total)
    env.close()
    return {'mean_return': float(np.mean(returns)), 'std_return': float(np.std(returns)), 'returns': returns}

# ---------------------------
# Example run
# ---------------------------
if __name__ == "__main__":
    # Example: train on CartPole
    results = train(env_name='CartPole-v1',
                    num_episodes=500,
                    buffer_capacity=50000,
                    batch_size=64,
                    gamma=0.99,
                    lr=1e-3,
                    alpha=0.5,
                    beta=0.5,
                    seed=123,
                    device='cpu')
    agent = results['agent']
    eval_stats = evaluate_agent(agent, 'CartPole-v1', episodes=20, seed=123)
    print("Evaluation:", eval_stats)
    # Save model checkpoint
    torch.save(agent.policy_net.state_dict(), "dqn_utility_checkpoint.pth")
    print("Training finished. Model saved.")
