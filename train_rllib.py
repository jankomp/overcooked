import time
import glob
import ray
from ray.train import RunConfig, CheckpointConfig
from environment.Overcooked import Overcooked_multi
from ray.tune.registry import register_env
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec, RLModule
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from Agents import AlwaysStationaryRLM, RandomRLM, HybridRandomRLM
import os
from functools import partial
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_POLICY_ID
)

def define_env(args):
    reward_config = {
        "metatask failed": -1,
        "pretask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.5,
        "right step": 0.5,
    }
    env_config = {
        "centralized": args.centralized,
        "grid_dim": [5, 5],
        "task": "lettuce-onion-tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "mode": "vector",
        "debug": False,
        "agents": ['ai', 'human'] if args.centralized else ['ai1', 'ai2', 'human'],
        "n_players": 3,
        "max_episode_length": 100,
        "rotate_map": False,
        "randomized_items": 0,
        "randomized_agents": 0,
        "ind_reward": args.ind_reward,  
        "ind_distance": args.ind_distance,
        "reward_distance": args.reward_distance,
    }

    register_env(
        "Overcooked",
        lambda env_config: Overcooked_multi(**env_config),
    )

    return env_config

def load_trained_human(checkpoint_path):
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, checkpoint_path)
    p = f"{storage_path}*"
    experiment_name = glob.glob(p)[-1]
    print(f"Loading results from {experiment_name}...")
    restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint

    human_module = RLModule.from_checkpoint(os.path.join(
        best_checkpoint.path,
        COMPONENT_LEARNER_GROUP,
        COMPONENT_LEARNER,
        COMPONENT_RL_MODULE,
        'human'
    ))
    human_module_spec = RLModuleSpec(
        module_class=type(human_module),
        observation_space=human_module.observation_space,
        action_space=human_module.action_space,
        model_config={'base_module': human_module}
    )
    return human_module_spec


def define_agents(args):
    '''
    Define the human agent policy and the policies to train.
    Can easily be extended to also define the AI policy
    :param args:
    :return: RLModuleSpec for the human agent, list for policies to train
    '''
    if args.rl_module == 'stationary':
        human_policy = RLModuleSpec(module_class=AlwaysStationaryRLM)
    elif args.rl_module == 'random':
        human_policy = RLModuleSpec(module_class=RandomRLM)
    elif args.rl_module == 'learn':
        human_policy = RLModuleSpec(model_config=custom_model_config())
        policies_to_train = ['ai', 'human'] if args.centralized else ['ai1', 'ai2', 'human']
        return human_policy, policies_to_train
    elif args.rl_module == 'hybrid':
        assert args.checkpoint_path, "If you want to train the AI with a hybrid human module, you have to specify the checkpoint path to the module!"
        human_policy = load_trained_human(args.checkpoint_path)
    else:
        raise NotImplementedError(f"{args.rl_module} not a valid human agent")
    
    policies_to_train = ['ai'] if args.centralized else ['ai1', 'ai2']
    return human_policy, policies_to_train

def _remote_fn(env_runner, rotate_map, randomize_items, randomize_agents):
    # We recreate the entire env object by changing the env_config on the worker,
    # then calling its `make_env()` method.
    new_env_config = {
        **env_runner.config.env_config,
        "rotate_map": rotate_map,
        "randomized_items": randomize_items,
        "randomized_agents": randomize_agents,
    }
    env_runner.config.environment(env_config=new_env_config)
    env_runner.make_env()

def _stage_params(stage):
    if stage == 0:
        rotate_map = True
        randomized_items = 0
        randomized_agents = 0
    if stage > 0 and stage <= 3:
        rotate_map = True
        randomized_items = 0
        randomized_agents = stage
    elif stage > 3 and stage <= 11:
        rotate_map = False
        randomized_items = stage - 3
        randomized_agents = 0
    elif stage > 11 and stage <= 19: 
        rotate_map = False
        randomized_items = stage - 12 if stage -12 < 8 else 8
        randomized_agents = 3
    elif stage == 20:
        rotate_map = True
        randomized_items = 8
        randomized_agents = 3
    return rotate_map, randomized_items, randomized_agents

class EnvRandomizationCallback(RLlibCallback):
    """Custom callback implementing `on_train_result()` for changing the envs' maps."""

    def on_train_result(
        self,
        *,
        algorithm: Algorithm,
        metrics_logger=None,
        result: dict,
        **kwargs,
    ) -> None:
        current_stage = algorithm._counters.get("current_stage", 0)
        new_stage = current_stage

        result["current_stage"] = current_stage
        current_return = result[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]
        #print("current_return", current_return)
        thresholds = [0.5 + 0.125 * i for i in range(21)]
        if current_return > thresholds[new_stage]:
            new_stage = current_stage + 1
            result["current_stage"] = new_stage
            if new_stage == 21:
                return
            else:
                rotate_map, randomized_items, randomized_agents = _stage_params(new_stage)

            print(
                f"Switching randomization on all EnvRunners to rotate_map={rotate_map}, randomize_items={randomized_items}, randomize_agents={randomized_agents} [(False, 0, 0)=easiest, "
                f"(True, 8, 3)=hardest], b/c R={current_return} on current stage."
            )
            algorithm.env_runner_group.foreach_env_runner(
                func=partial(_remote_fn, rotate_map=rotate_map, randomize_items=randomized_items, randomize_agents=randomized_agents)
            )
        # Emergency brake: If return is smaller than -0.5 AND we are already at a harder stage than 0, we go back one stage
        elif current_return < -0.5 and new_stage > 0:
            print(
                "Emergency brake: Our policy seems to have collapsed -> Setting stage "
                f"back to {current_stage - 1}, b/c R={current_return} on current stage, stage {current_stage}."
            )
            new_stage = current_stage - 1
            rotate_map, randomized_items, randomized_agents = _stage_params(new_stage)
            algorithm.env_runner_group.foreach_env_runner(
                func=partial(_remote_fn, rotate_map=rotate_map, randomize_items=randomized_items, randomize_agents=randomized_agents)
            )
        algorithm._counters["current_stage"] = new_stage


def custom_model_config():
    model_config = DefaultModelConfig()
    model_config.fcnet_hiddens = [256, 256] # hidden layers
    model_config.fcnet_activation = 'relu' # relu activation instead of default (tanh)
    #model_config.use_lstm = True # use LSTM so we have memory
    #model_config.lstm_cell_size = 128 
    #model_config.lstm_use_prev_action = True
    #model_config.lstm_use_prev_reward = True
    return model_config

def define_training(centralized, human_policy, policies_to_train, env_config):
    config = (
        PPOConfig()
        #.callbacks(EnvRandomizationCallback)
        .api_stack( #reduce some warning
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment("Overcooked", env_config=env_config)
        .env_runners( # define how many envs to run in parallel and resources per env
            num_envs_per_env_runner=2,
            num_cpus_per_env_runner=4,
            num_gpus_per_env_runner=0
        )
        .training( # these are hyper paramters for PPO
            use_critic=True,
            use_gae=True,
            lr=5e-3,
            lambda_=0.95,
            gamma=0.99,
            clip_param=0.2,
            entropy_coeff=0.03,
            vf_loss_coeff=0.2,
            grad_clip=0.5,
            num_epochs=10,
            minibatch_size=2048,
        )
    )

    rl_module_spec = RLModuleSpec(model_config=custom_model_config())


    if centralized:
        config = (config
            .multi_agent(
                policies={"ai", "human"}, # one critic controlling all the ai players
                policy_mapping_fn=lambda aid, *a, **kw: aid,
                policies_to_train=policies_to_train

            )
            .rl_module( # define what kind of policy each agent is
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "ai": rl_module_spec,
                        "human": human_policy,
                    }
                ),
            )
        )
    else:
        config = (config
            .multi_agent(
                policies={"ai1", "ai2", "human"},
                policy_mapping_fn=lambda aid, *a, **kw: aid,
                policies_to_train=policies_to_train

            )
            .rl_module( # define what kind of policy each agent is
                rl_module_spec=MultiRLModuleSpec(
                    rl_module_specs={
                        "ai1": rl_module_spec,
                        "ai2": rl_module_spec,
                        "human": human_policy,
                    }
                ),
            )
        )

    return config


def train(args, config):
    ray.init()
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir) # save the results in the runs folder
    experiment_name = f"{'centralized' if args.centralized else 'decentralized'}_{args.name}_{args.rl_module}_{int(time.time() * 1000)}" # add a timestamp to the name to make it unique
    tuner = tune.Tuner(
        "PPO",
        param_space=config,
        #tune_config=tune.TuneConfig(
        #    metric="env_runners/episode_return_mean",
        #    mode="max",
        #    num_samples=3
        #),
        run_config=RunConfig(
            storage_path=storage_path,
            name=experiment_name,
            stop={
                "training_iteration": 1_000,
                "env_runners/episode_return_mean": 10.0,
            },
            checkpoint_config=CheckpointConfig(checkpoint_frequency=10, checkpoint_at_end=True, num_to_keep=2), # save a checkpoint every 10 iterations
        )
    )
    tuner.fit()

def main(args):
    env_config = define_env(args)
    human_policy, policies_to_train = define_agents(args)
    config = define_training(args.centralized, human_policy, policies_to_train, env_config)
    train(args, config)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="random", help = "Set the policy of the human, can be stationary, random, hybrid or learn")
    parser.add_argument("--centralized", action="store_true", help="True for centralized training, False for decentralized training")
    parser.add_argument("--ind_reward", action="store_true", help="True for individual reward, False for shared reward")
    parser.add_argument("--ind_distance", action="store_true", help="True for individual distance, False for shared distance")
    parser.add_argument("--reward_distance", action="store_true", help="True for incorporating item distance into rewards, False for not")
    parser.add_argument("--checkpoint_path", help="The path to the trained policy the human module is taken from if we are using rl_module hybrid.")
    args = parser.parse_args()

    ip = main(args)