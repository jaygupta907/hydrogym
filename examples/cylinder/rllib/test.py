import os
import ray
from firedrake import logging
from ray import tune
from ray.rllib.algorithms import ppo  # ray.rllib.algorithms in latest version
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
import hydrogym as hgym
import numpy as np
import psutil
import argparse

import torch
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2


logging.set_log_level(logging.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run",
    type=str,
    default="PPO",
    help="The RLlib-registered algorithm to use.")
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters",
    type=int,
    default=10000,
    help="Number of iterations to train.")
parser.add_argument(
    "--stop-timesteps",
    type=int,
    default=1000000,
    help="Max number of timesteps to train.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=1e8,
    help="Reward at which we stop training.")
parser.add_argument(
    "--tune",
    action="store_true",
    help="Run using Tune with grid search and TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)

precomputed_data = "../output"
env_config = {
    "flow": hgym.firedrake.RotaryCylinder,
    "flow_config": {
        "Re" :100,
        "restart": f"{precomputed_data}/checkpoint.h5",
        "mesh":'medium',
        "velocity_order":1
    },
    "solver_config":{
        "dt":0.01,
    },
    "solver": hgym.firedrake.SemiImplicitBDF,
}

class CustomModel(TFModelV2):
  """Example of a keras custom model that just delegates to an fc-net."""

  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    super(CustomModel, self).__init__(obs_space, action_space, num_outputs,
                                      model_config, name)
    self.model = FullyConnectedNetwork(obs_space, action_space, num_outputs,
                                       model_config, name)

  def forward(self, input_dict, state, seq_lens):
    return self.model.forward(input_dict, state, seq_lens)

  def value_function(self):
    return self.model.value_function()
  

class TorchCustomModel(TorchModelV2, torch.nn.Module):
  """Example of a PyTorch custom model that just delegates to a fc-net."""

  def __init__(self, obs_space, action_space, num_outputs, model_config, name):
    TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                          model_config, name)
    torch.nn.Module.__init__(self)

    self.torch_sub_model = TorchFC(obs_space, action_space, num_outputs,
                                   model_config, name)

  def forward(self, input_dict, state, seq_lens):
    if isinstance(input_dict["obs"], tuple):
      input_dict["obs"] = torch.stack(input_dict["obs"], dim=1)
      input_dict["obs_flat"] = input_dict["obs"]
    fc_out, _ = self.torch_sub_model(input_dict, state, seq_lens)
    return fc_out, []

  def value_function(self):
    return torch.reshape(self.torch_sub_model.value_function(), [-1])


if __name__ == "__main__":

    args = parser.parse_args()
    print(f"Running with following CLI options: {args}")

    tune.register_env("cylinder", lambda config: hgym.FlowEnv(env_config))
    ray.init()

    # Can also register the env creator function explicitly with:
    # register_env("corridor", lambda config: SimpleCorridor(config))
    ModelCatalog.register_custom_model("cav_actor", TorchCustomModel)

    def log_postprocess(flow):
        mem_usage = psutil.virtual_memory().percent
        CL, CD = flow.get_observations()
        return CL, CD, mem_usage
    
    # Set up the printing callback
    print_fmt = "t: {0:0.2f},\t\t CL: {1:0.3f},\t\t CD: {2:0.03f}\t\t Mem: {3:0.1f}"  # This will format the output
    log = hgym.firedrake.utils.io.LogCallback(
        postprocess=log_postprocess,
        nvals=3,
        interval=1,
        print_fmt=print_fmt,
        filename=None,
    )

    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    config = {
        "log_level": "DEBUG",
        "env": "cylinder",
        "env_config": {
            "Re": 100,
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
    }

#    if not args.tune:
    # manual training with train loop using PPO and fixed learning rate
    print("Running manual train loop without Ray Tune.")
    ppo_config = ppo.PPOConfig()
    ppo_config = ppo_config.update_from_dict(config_dict=config)
    ppo_config["lr"] = 1e-3
    trainer = ppo.PPO(config=ppo_config, env="cylinder")
    # run manual training loop and print results after each iteration
    for i in range(20):
        result = trainer.train()
        print(pretty_print(result))
        trainer.save("./rllib_checkpoint")
        
        # Stop training of the target train steps or reward are reached
        if (result["timesteps_total"] >= args.stop_timesteps or
            result["episode_reward_mean"] >= args.stop_reward):
            break
#else:
#    # automated run with Tune and grid search and TensorBoard
#    print("Training automatically with Ray Tune")
#    results = tune.run(args.run, config=config, stop=stop)
#    if args.as_test:
#        print("Checking if learning goals were achieved")
#        check_learning_achieved(results, args.stop_reward)