# Walkthrough 2: Training the 24-Block Transformer Model

This walkthrough covers dataset generation, model configuration, and training workflows for Noema's 24-block flagship neural theorem proving model.

## Model Configuration

Noema's core neural guide is a 24-block autoregressive Transformer designed specifically for formal mathematical reasoning:

| Hyperparameter | Value | Description |
|---|---|---|
| Number of Layers | 24 | Independently parameterized Transformer blocks |
| Hidden Dimension ($d_{\text{model}}$) | 512 | Representation width across attention and MLP |
| Attention Heads | 8 | Multi-head attention with head dimension 64 |
| Feedforward Dimension ($d_{\text{ff}}$) | 2048 | SwiGLU gated activation network |
| Vocabulary Size ($V$) | 16384 | BPE tokenized vocabulary for Lean 4 syntax |
| Parameter Count | 60,128,673 | Trainable weights (without tied embeddings) |

## Instantiating the Architecture in Python

```python
import torch
from noema.models.transformer import ProverTransformer, TransformerConfig

config = TransformerConfig(
    vocab_size=16384,
    d_model=512,
    num_layers=24,
    num_heads=8,
    d_ff=2048,
    max_seq_len=2048,
    dropout=0.1,
)

model = ProverTransformer(config)
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {param_count:,}")
```

## Synthetic and Formal Corpus Generation

Generate formal training data across abstract algebra, order theory, set theory, and topology:

```bash
# Generate synthetic corpus across domains
python3 -c "
from noema.corpus.generators.algebra import generate_algebra_dataset
from noema.corpus.generators.set_theory import generate_set_theory_dataset
from noema.corpus.generators.topology import generate_topology_dataset

alg_data = generate_algebra_dataset(samples=1000)
set_data = generate_set_theory_dataset(samples=1000)
top_data = generate_topology_dataset(samples=1000)
print(f'Generated {len(alg_data) + len(set_data) + len(top_data)} proof steps.')
"
```

## Multi-Task Training Loop

The model simultaneously optimizes three distinct objectives:
1. **Tactic Policy Loss ($\mathcal{L}_{\text{policy}}$)**: Cross-entropy over tactic token distributions.
2. **State Value Loss ($\mathcal{L}_{\text{value}}$)**: Mean squared error on scalar proving outcome ($1.0$ for solved, $0.0$ for failed).
3. **Censored Progress Loss ($\mathcal{L}_{\text{progress}}$)**: Negative log-likelihood on remaining proof steps with right-censoring.

```python
import torch.optim as optim
from noema.training.trainer import ProverTrainer, TrainingConfig

train_config = TrainingConfig(
    batch_size=16,
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=500,
    max_steps=10000,
    gradient_accumulation_steps=4,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

trainer = ProverTrainer(
    model=model,
    config=train_config,
    train_dataset=alg_data,
    val_dataset=set_data,
)

trainer.train()
trainer.save_checkpoint("checkpoints/model_24block_final.pt")
```

## Reinforcement Learning via PPO

After supervised pre-training, fine-tune using Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE):

```python
from noema.training.ppo import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    clip_epsilon=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    value_coeff=0.5,
    entropy_coeff=0.01,
)

ppo_agent = PPOTrainer(model=model, config=ppo_config)
# Perform policy optimization on self-play rollouts
```
