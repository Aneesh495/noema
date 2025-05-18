# ADR-004: Right-Censored Goal Progress and Joint Policy-Value Loss

## Context
Standard reinforcement learning and imitation learning formulations use binary outcome rewards: $+1$ for a complete proof and $0$ for failure. However, in automated theorem proving, most search trajectories terminate due to timeout or step budgets without finding a complete proof. Discarding all intermediate steps from unfinished proofs discards valuable signal regarding partial progress. Conversely, naively measuring goal count can reward pathological tactics that create trivial redundant subgoals.

## Decision
Noema formulates training loss as a joint multi-task objective with survival-analytic right-censored progress:
1. Joint Multi-Task Loss:
   $$\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{tactic}} + \lambda_{\text{premise}} \mathcal{L}_{\text{premise}} + \lambda_{\text{value}} \mathcal{L}_{\text{value}} + \lambda_{\text{progress}} \mathcal{L}_{\text{progress}}$$
2. Right-Censored Progress Model:
   - For proven theorems, progress $y_t \in [0, 1]$ is known exactly as the normalized fraction of remaining search distance to QED.
   - For unfinished trajectories halted by search budgets, progress is right-censored at the observed horizon: we know true progress $y^* \ge y_{\text{observed}}$.
   - Censored likelihood loss applies a asymmetric penalty that penalizes predicting less progress than observed while smoothly handling open-ended bounds.

## Consequences
- Provides dense gradient signals across both successful and unsuccessful proof attempts.
- Encourages neural value heads to recognize productive goal simplifications before full proof completion.
