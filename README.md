# TrustGuardAI Utility Replay DQN

**Reference implementation** of two research contributions:

- **TrustGuardAI** — streaming anomaly detection (LSTM autoencoder with attention and Monte Carlo dropout for uncertainty).  
- **Utility Driven Replay** — a DQN replay buffer that uses a composite utility \(U_t = \alpha |r_t| + \beta |\delta_t|\) to selectively retain high‑value transitions and evict low‑utility ones.

This repository provides a production‑style PyTorch implementation of the Utility Replay Buffer and a runnable DQN agent for OpenAI Gym environments (CartPole-v1, Acrobot-v1). It logs retention/eviction events, supports uniform sampling over retained transitions, and includes utilities for evaluation and plotting.

---

## Key Features

- **Utility Replay Buffer** with min‑heap eviction keyed by composite utility.  
- **DQN Agent** with target network, epsilon schedule, and standard optimization loop.  
- **Uniform sampling** over retained transitions to preserve simple learning dynamics.  
- **Interpretability logs** for retention and eviction events (auditability).  
- **Evaluation utilities** for learning curves, final returns, and buffer statistics.  
- **Single‑file runnable example** for quick experiments and reproducibility.

---

## Repository Structure

```
.
├─ src
│  ├─ utility_dqn.py        # Main implementation and example run
│  ├─ trustguardai.py       # (Optional) LSTM autoencoder implementation
├─ notebooks
│  ├─ demo.ipynb            # Training, plotting, and analysis notebook
├─ models
│  ├─ checkpoints           # Saved model weights
├─ data
│  ├─ README.md             # Dataset instructions and download links
├─ docs
│  ├─ figures               # Plots and paper figures
├─ requirements.txt
├─ README.md                # This file
```

---

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows PowerShell
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Minimal `requirements.txt`**

```
torch>=2.0
numpy
gym
matplotlib
scikit-learn
```

---

## Quick Start

Run a short training session on CartPole (default settings):

```bash
python src/utility_dqn.py
```

This runs the example training loop in `utility_dqn.py` and saves a checkpoint `dqn_utility_checkpoint.pth`.

Run with custom settings from Python:

```python
from src.utility_dqn import train
train(env_name='Acrobot-v1', num_episodes=300, buffer_capacity=50000,
      batch_size=64, alpha=0.5, beta=0.5, seed=42)
```

---

## Configuration & Hyperparameters

Core hyperparameters (defaults shown in code):

- **buffer_capacity** — number of transitions retained (default: `50000`).  
- **alpha** — weight for reward magnitude in utility (default: `0.5`).  
- **beta** — weight for TD error in utility (default: `0.5`).  
- **batch_size** — training mini‑batch size (default: `64`).  
- **gamma** — discount factor (default: `0.99`).  
- **lr** — learning rate for Adam (default: `1e-3`).  
- **epsilon schedule** — start `1.0`, final `0.01`, decay controlled by `epsilon_decay`.  
- **update_target_every** — target network sync frequency in steps.

Change parameters by editing the `train(...)` call or by adding a small CLI wrapper.

---

## Logging & Evaluation

**Logged outputs**

- **Retention log**: `(step, retained_utility)` for every insertion.  
- **Eviction log**: `(step, evicted_utility)` for every eviction.  
- **Buffer stats**: mean/min/max utility and current size printed periodically.  
- **Training metrics**: episode rewards, moving averages, and loss history.

**Plotting**

Use `notebooks/demo.ipynb` to generate:

- Learning curves (episode reward vs episodes).  
- Utility distribution histograms for retained vs evicted transitions.  
- Eviction frequency over time.

Export logs to CSV for offline analysis and figure reproduction.

---

## Reproducibility Checklist

- Set seeds for `random`, `numpy`, and `torch` before experiments.  
- Record environment versions and PyTorch/CUDA versions.  
- Run multiple seeds and average metrics (paper used 5 seeds).  
- Save model checkpoints and the exact hyperparameter configuration for each run.  
- For single‑threaded CPU latency experiments: set `OMP_NUM_THREADS=1` and measure with `time.perf_counter()`.

---

## Practical Notes & Extensions

- **Utility computation**: current implementation computes utility at insertion using the current networks. If desired, add a maintenance pass to recompute utilities periodically and rebalance the heap.  
- **PER baseline**: implement Prioritized Experience Replay separately to reproduce baselines; keep sampling uniform for the utility buffer to match the paper.  
- **Scaling**: for larger environments or continuous control, increase buffer capacity and consider GPU training for the DQN.  
- **Logging**: export `eviction_log` and `retention_log` to CSV for offline analysis and figure reproduction.  
- **Visualization**: add attention/utility visualizations and calibration plots in the notebook for deeper analysis.

---

## How the Utility Buffer Works (brief)

- Each transition \(e_t = (s_t, a_t, r_t, s_{t+1})\) is assigned utility  
  \[
  U_t = \alpha |r_t| + \beta |\delta_t|
  \]
  where \(\delta_t\) is the TD error computed using the current policy/target networks.  
- The buffer stores transitions in a min‑heap keyed by \(U_t\). When capacity is exceeded, the lowest‑utility transition is evicted.  
- Sampling for training is uniform over the retained set. Retention/eviction events are logged for interpretability.

---

## Citation

If you use this code in research or projects, please cite:

```
K. Charwick Hamesh et al., "Utility Driven Selective Memory for Reinforcement Learning", IEEE conference paper, 2025.
```

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Suggested workflow:

1. Fork the repository.  
2. Create a feature branch.  
3. Add tests or a notebook demonstrating the change.  
4. Open a pull request with a clear description of the change.
