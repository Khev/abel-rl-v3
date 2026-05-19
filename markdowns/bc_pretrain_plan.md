# BC pretraining plan (deferred — not implemented yet)

ChatGPT recommendation #3: bootstrap PPO with behavior cloning on
cheaply-generated solution traces. Drafted here so we can pick it up later
without re-deriving the design.

## Why

Two related observations:

- The value-beam result (Section `sec:value_beam` in paper) shows that even
  with our best baseline, **policy greedy = 0.01–0.22 while beam = 0.27–0.40**.
  Most of the policy's "knowledge" sits in lower-ranked actions; the argmax
  is unreliable.
- Anti-loop seed9100 achieved greedy = 0.22 (vs 0.01 baseline), suggesting
  that policies trained with action-diversity pressure produce sharper,
  more decisive argmaxes. Same insight as BC: when the policy sees
  end-to-end correct paths, it commits to them.

BC from existing solved traces should sharpen the policy in the same way —
without needing a custom reward shaping like anti-loop.

## Trace sources

Order by quality (best traces first):

1. **Current trained agents' solves.** We already log every solve to
   stdout in `train_abel.py:_on_step`. Could persist these to disk during
   training; for existing runs, replay by loading the model and rolling
   out greedy on solved-train-eqns until we capture the trace.
2. **Beam search successes** on the test set with value-beam — these are
   often *shorter* than the agent's own greedy path (since search finds
   better paths). Generate by running `eval_value_beam.py` on train
   equations and saving paths.
3. **BFS solver on tiny equations.** For `abel_level1/2` (linear, depth-1)
   we can BFS the action tree and find every legal short path. Cheap.
4. **Hand-coded teacher** for specific templates. E.g., for `a*x**2 + b*x + c = 0`,
   the canonical recipe is `COV → REL → SUB → DIV → SQRT → SQRT`. Write
   one teacher per CoV class. Most reliable but most work.

Start with **(1) + (2)**: they require no new infrastructure beyond a
trace-dumping pass.

## Trace storage format

```python
# bc_traces/<dataset>.jsonl, one trace per line
{
    "eqn": "a*x**2 + 2*b*x - 4*e",       # input eqn (sympy str)
    "actions": [12, 13, 0, 4, 2, 2],     # action indices in order
    "source": "seed8001_step3000000",    # provenance
    "solved": true,
    "length": 6
}
```

Keep them simple: only solved traces, only the action-index sequence. No
intermediate states (the env is deterministic, can replay).

## Pretraining pipeline

```
python pretrain_bc.py \
  --traces bc_traces/mixed_v2_easy.jsonl \
  --epochs 20 \
  --batch_size 256 \
  --gen mixed_v2_easy \
  --agent ppo-tree \
  --out checkpoints/bc_pretrain.zip
```

Inside:

1. Build env + initial MaskablePPO model (same hyperparams as our PPO setup).
2. Load traces. For each trace:
   - Reset env to `eqn`.
   - For each action `a_t` in the trace:
     - Get current `obs`, `action_mask`.
     - Compute policy distribution; take cross-entropy loss vs `a_t`.
     - Step env to advance state.
3. Standard SGD optimizer (Adam, lr 1e-4).
4. Periodically eval `greedy_accuracy` on test set to monitor progress.
5. Save final `pretrain.zip`.

## Fine-tune pipeline

```
python train_abel.py --gen mixed_v2_easy ... \
  --load_model_path checkpoints/bc_pretrain.zip \
  --ent_coef 0.005 \
  ...
```

Key tweaks vs from-scratch training:

- **Lower entropy coefficient** initially (e.g. 0.005 vs 0.01). The BC
  policy is already sharp; high entropy bonus would erase the BC effect.
- **Optionally**: KL anchor to BC policy for first N rollouts to prevent
  policy drift.

## Ablation table (target)

| Config | greedy | beam | beam+value |
|---|---|---|---|
| PPO from-scratch | 0.01 | 0.28 | 0.40 |
| PPO + anti-loop | 0.22 | 0.34 | 0.34 |
| BC pretrain only | ? | ? | ? |
| BC pretrain → PPO finetune | ? | ? | ? |
| BC pretrain → PPO finetune + value-beam | ? | ? | ? |

The interesting question: does BC pretrain → finetune narrow the
greedy/beam gap (i.e., does it match anti-loop's sharper argmax effect)?
If yes, we get a much cleaner methods story:

> "Pretraining on agent-discovered traces sharpens the policy; value-guided
> beam compensates for residual ranking errors."

## Risks / things to watch

- **Mode collapse**: if all our traces use the same quartic recipe (the
  baseline's REL-loop story), the BC policy will inherit that bias.
  Mitigation: include traces from seed9100 (anti-loop, more diverse).
- **Trace quality**: greedy traces from seed8001 are only 3% of test eqns
  — sample size is tiny. Train traces are bigger (coverage=1.0 = 916 train
  solves) but trained on relabel-canonicalized forms.
- **Distribution shift**: BC is trained on solved-trace distribution; PPO
  may immediately drift back to its own distribution. Hence the KL anchor
  suggestion.

## Estimated effort

- Trace dump: 0.5 day (modify `_on_step` to write JSON, or write a separate
  collect script).
- pretrain_bc.py: 1 day (most of it is the env-reset + sequence loss; we
  already have the dict-obs handling from success-replay).
- First full ablation: 1 day compute + writeup.

**Total: ~2.5 days**, including paper writeup. Defer until value-beam
result + anti-loop variance results stabilize, then decide whether BC
pretrain adds to the story or duplicates anti-loop's contribution.
