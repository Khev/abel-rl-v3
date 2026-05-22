import sys, time, glob
sys.set_int_max_str_digits(0)
import warnings
warnings.filterwarnings("ignore")

t0 = time.time()
from sb3_contrib import MaskablePPO
from envs.env_multi_eqn import multiEqn
from train_abel import greedy_accuracy, beam_accuracy
print(f"imports:        {time.time()-t0:6.1f}s", flush=True)

t = time.time()
env = multiEqn(gen="abel_level4", state_rep="graph_integer_1d",
               use_cov=False, use_relabel_constants=True,
               use_success_replay=True)
print(f"env build:      {time.time()-t:6.1f}s   test_eqns={len(env.test_eqns)}",
      flush=True)

ckdir = ("data/dynamic_actions/use_relabel_constants/use_buffer/"
         "abel_level4_hidden_dim256_nenvs1/ppo-tree/seed7006/checkpoints/")
ck = sorted(glob.glob(ckdir + "model_step*.zip"))[0]
t = time.time()
model = MaskablePPO.load(ck, env=env, device="cpu")
print(f"model load:     {time.time()-t:6.1f}s", flush=True)

eqns = env.test_eqns[:5]
t = time.time()
g = greedy_accuracy(model, env, eqns, max_steps=60, per_eqn_seconds=0.75)
print(f"greedy 5 eqns:  {time.time()-t:6.1f}s   acc={g}", flush=True)

t = time.time()
pb = beam_accuracy(model, env, eqns, beam_width=5, topk_per_node=5,
                   max_steps=60, per_eqn_seconds=0.75, beam_lambda=0.0)
print(f"beam 5 eqns:    {time.time()-t:6.1f}s   acc={pb}", flush=True)
print("DONE", flush=True)
