import time
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import get_overcooked_multi_class
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from Agents import AlwaysStationaryRLM, RandomRLM
import os


def define_env():
    reward_config = {
        "metatask failed": 0,
        "goodtask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1.,
    }
    env_params = {
        "centralized": False,
        "grid_dim": [5, 5],
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "mode": "vector",
        "debug": False,
        "agents": ['ai1', 'ai2'],
    }

    register_env(
        "Overcooked",
        lambda _: get_overcooked_multi_class(env_params),
    )

def define_agents(args):
    '''
    Define the human agent policy and the policies to train.
    Can easily be extended to also define the AI policy
    :param args:
    :return: RLModuleSpec for the human agent, list for policies to train
    '''
    if args.rl_module == 'stationary':
        human_policy = RLModuleSpec(module_class=AlwaysStationaryRLM)
        policies_to_train = ['ai1', 'ai2']
    elif args.rl_module == 'random':
        human_policy = RLModuleSpec(module_class=RandomRLM)
        policies_to_train = ['ai1', 'ai2']
    elif args.rl_module == 'learned':
        human_policy = RLModuleSpec()
        policies_to_train = ['ai1', 'ai2,' 'human']
    else:
        raise NotImplementedError(f"{args.rl_module} not a valid human agent")
    return human_policy, policies_to_train


def define_training(human_policy, policies_to_train):
    gpuenabled = False
    if gpuenabled:
        config = (
            PPOConfig()
            .resources(num_gpus=1)
            .api_stack( #reduce some warning.
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment("Overcooked")
            .env_runners( # define how many envs to run in parallel and resources per env
                num_env_runners=1 # number of runners
                num_envs_per_env_runner=1, # number of envs per runner (RAM/VRAM capped)
                num_cpus_per_env_runner=1, # number of cpus used per runner (aim 75% cores = num_env_runners * num_cpus_per_env_runner)
                num_gpus_per_env_runner=0.5 # number of gpus used per env runner (aim 100% gpus (0.5*2=1 gpu) = num_env_runners * num_gpus_per_env_runner)
            )
            .multi_agent(
                policies={"ai1", "ai2"}, # old: policies={"ai1", "ai2", "human"},
                policy_mapping_fn=lambda aid, *a, **kw: aid,
                policies_to_train=policies_to_train

            )
            .rl_module( # define what kind of policy each agent is
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        #"human": human_policy,
                        "ai1": RLModuleSpec(),
                        "ai2": RLModuleSpec(),
                    }
                ),
            )
            .training( # these are hyper paramters for PPO
                lr=1e-3,
                lambda_=0.98,
                gamma=0.99,
                clip_param=0.05,
                entropy_coeff=0.1,
                vf_loss_coeff=0.1,
                grad_clip=0.1,
                num_epochs=10,
                minibatch_size=64,
            )
    else:
        config = (
            PPOConfig()
            .api_stack( #reduce some warning.
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment("Overcooked")
            .env_runners( # define how many envs to run in parallel and resources per env
                num_envs_per_env_runner=1,
                num_cpus_per_env_runner=1,
                num_gpus_per_env_runner=0
            )
            .multi_agent(
                policies={"ai1", "ai2"}, # old: policies={"ai1", "ai2", "human"},
                policy_mapping_fn=lambda aid, *a, **kw: aid,
                policies_to_train=policies_to_train

            )
            .rl_module( # define what kind of policy each agent is
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        #"human": human_policy,
                        "ai1": RLModuleSpec(),
                        "ai2": RLModuleSpec(),
                    }
                ),
            )
            .training( # these are hyper paramters for PPO
                lr=1e-3,
                lambda_=0.98,
                gamma=0.99,
                clip_param=0.05,
                entropy_coeff=0.1,
                vf_loss_coeff=0.1,
                grad_clip=0.1,
                num_epochs=10,
                minibatch_size=64,
            )
    )
    return config


def train(args, config):
    ray.init()
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir) # save the results in the runs folder
    experiment_name = f"{args.name}_{args.rl_module}_{int(time.time() * 1000)}" # add a timestamp to the name to make it unique
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop={"training_iteration": 200}, # stop after 200 iterations (fairly arbitrary, and many more options if you look at the docs)
            checkpoint_config=CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True, num_to_keep=2), # save a checkpoint every 10 iterations
        )
    )
    tuner.fit()

def main(args):
    define_env()
    human_policy, policies_to_train = define_agents(args)
    config = define_training(human_policy, policies_to_train)
    train(args, config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="stationary", help = "Set the policy of the human, can be stationary, random, or learned")

    args = parser.parse_args()
    ip = main(args)