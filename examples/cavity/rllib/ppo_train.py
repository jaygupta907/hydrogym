# Modified from https://github.com/ray-project/ray/blob/master/rllib/examples/custom_env.py
import os
import ray
from common import *
from common import TorchCustomModel, parser
from firedrake import logging
from ray import tune
from ray.rllib.algorithms import ppo 
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
import hydrogym as hgym

precomputed_data = "../output"

env_config = {
    "flow": hgym.firedrake.Cavity,
    "flow_config": {
        "restart": f"{precomputed_data}/checkpoint.h5"
    },
    "solver_config":{
        "dt":0.01,
    },
    "solver": hgym.firedrake.SemiImplicitBDF,
}

logging.set_log_level(logging.DEBUG)

if __name__ == "__main__":
  args = parser.parse_args()
  print(f"Running with following CLI options: {args}")

  tune.register_env("cavity", lambda config: hgym.FlowEnv(env_config))
  ray.init(local_mode=args.local_mode)
  


  # Can also register the env creator function explicitly with:
  # register_env("corridor", lambda config: SimpleCorridor(config))
  ModelCatalog.register_custom_model("cav_actor", TorchCustomModel)

  # Set up the printing callback
  log = hgym.firedrake.io.LogCallback(
      postprocess=lambda flow: flow.collect_observations(),
      nvals=1,
      interval=1,
      print_fmt="t: {0:0.2f},\t\t m: {1:0.3f}",
      filename=None,
  )

  config = {
      "log_level": "DEBUG",
      "env": "cavity",
      "env_config": {
          "Re": 5000,
          "checkpoint": "./output/checkpoint.h5",
          "mesh": "coarse",
          "callbacks": [log],
          "max_steps": 1000,
      },
      # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
      "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
      "model": {
          "custom_model": "cav_actor",
          "vf_share_layers": True,
      },
      "num_workers": 1, 
      "sample_timeout_s" : 200.0
  }

  stop = {
      "training_iteration": args.stop_iters,
      "timesteps_total": args.stop_timesteps,
      "episode_reward_mean": args.stop_reward,
  }

  if not args.tune:
    # manual training with train loop using PPO and fixed learning rate
    if args.run != "PPO":
      raise ValueError("Only support --run PPO with --no-tune.")
    print("Running manual train loop without Ray Tune.")
    ppo_config = ppo.PPOConfig()
    ppo_config = ppo_config.update_from_dict(config_dict=config)
    ppo_config["lr"] = 1e-3
    trainer = ppo.PPO(config=ppo_config, env="cavity")
    # run manual training loop and print results after each iteration
    for _ in range(args.stop_iters):
      result = trainer.train()
      print(pretty_print(result))
      trainer.save("./rllib_checkpoint")
      # stop training of the target train steps or reward are reached
      if (result["timesteps_total"] >= args.stop_timesteps or
          result["episode_reward_mean"] >= args.stop_reward):
        break
  else:
    # automated run with Tune and grid search and TensorBoard
    print("Training automatically with Ray Tune")
    results = tune.run(args.run, config=config, stop=stop)

    if args.as_test:
      print("Checking if learning goals were achieved")
      check_learning_achieved(results, args.stop_reward)

  ray.shutdown()
