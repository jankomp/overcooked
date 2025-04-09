import glob
import time
from environment.Overcooked import get_overcooked_multi_class
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core import (
    COMPONENT_LEARNER_GROUP,
    COMPONENT_LEARNER,
    COMPONENT_RL_MODULE,
    DEFAULT_POLICY_ID
)
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
)
from ray.rllib.core.columns import Columns
import torch
import os
from ray.rllib.utils.numpy import convert_to_numpy, softmax
import numpy as np

def define_environment(args):
    reward_config = {
        "metatask failed": 0,
        "goodtask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -1.,
    }
    env_params = {
        "centralized": args.centralized,
        "grid_dim": [5, 5],
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "mode": "vector",
        "debug": False,
        "agents": ['ai1', 'ai2'],
    }

    env = get_overcooked_multi_class(env_params)
    return env


def sample_action(mdl, obs, action_space_shape):
    mdl_out = mdl.forward_inference({Columns.OBS: obs})
    if Columns.ACTION_DIST_INPUTS in mdl_out: #our custom policies might return the actions directly, while learned policies might return logits.
        logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])

        if action_space_shape is not None: # centralized control means one action per agent
            logits = np.reshape(logits, action_space_shape)
            action = [np.random.choice(list(range(len(agent_logits))), p=softmax(agent_logits)) for agent_logits in logits]
        else:
            action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
        return action
    elif 'actions' in mdl_out:
        return mdl_out['actions'][0]

    else:
        raise NotImplementedError("Something weird is going on when sampling acitons")

def load_modules(args):
    current_dir = os.getcwd()
    storage_path = os.path.join(current_dir, args.save_dir)
    p = f"{storage_path}/{"centralized" if args.centralized else "decentralized"}_{args.name}_{args.rl_module}_*"
    experiment_name = glob.glob(p)[-1]
    print(f"Loading results from {experiment_name}...")
    restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint

    if args.centralized:
        rl_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            DEFAULT_POLICY_ID
            )
        )
        return rl_module
    else:
        ai1_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'ai1',
        ))
        ai2_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'ai2',
        ))
        return ai1_module, ai2_module


def main(args):
    env = define_environment(args)

    if args.centralized:
        ai_module = load_modules(args)
    else:
        ai1_module, ai2_module = load_modules(args)
    env.game.on_init()
    obs, info = env.reset()
    env.render()

    action_space_shape = np.array([env.action_space.shape[0], env.action_space[0].n]) if args.centralized else None

    while True:
        if args.centralized:
            action = sample_action(ai_module, torch.from_numpy(obs).unsqueeze(0).float(), action_space_shape)
        else:
            ai1_action = sample_action(ai1_module, torch.from_numpy(obs['ai1']).unsqueeze(0).float(), action_space_shape)
            ai2_action = sample_action(ai1_module, torch.from_numpy(obs['ai2']).unsqueeze(0).float(), action_space_shape)
            action = {'ai1': ai1_action,
                    'ai2': ai2_action}

        obs, rewards, terminateds, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

        if args.centralized:
            if terminateds:
                break
        else:
            if terminateds['__all__']:
                break

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="stationary", type=str)
    parser.add_argument("--centralized", action="store_true", help="True for centralized training, False for decentralized training")

    args = parser.parse_args()
    main(args)
