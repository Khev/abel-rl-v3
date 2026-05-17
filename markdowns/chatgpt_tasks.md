We have an RL symbolic equation-solving codebase using PPO + TreeMLP.
The environment has deterministic SymPy transitions, legal action masks, and successful solution traces.

Goal: implement value-guided beam search first. Do not refactor the environment or action space yet.

Tasks:
1. Locate the evaluation code for greedy and beam search.
2. Add canonical equation-state hashing using SymPy simplify/srepr or an existing canonicalizer.
3. Implement beam expansion over legal actions only.
4. Score partial paths by:
   score = log_policy_sum + lambda_value * value_estimate - alpha_len * path_length - beta_complexity * complexity
5. Add CLI args:
   --beam-width
   --beam-depth
   --lambda-value
   --alpha-len
   --beta-complexity
6. Save per-equation results:
   solved, path length, final equation, action trace, score components.
7. Add tests:
   - beam width 1 matches greedy behavior approximately
   - solved traces replay deterministically
   - no duplicate canonical states within one beam search
   - illegal actions are never selected
8. Do not change training code in this pass.

Return a summary of modified files and exact commands to run.
