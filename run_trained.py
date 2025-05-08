import glob
import time
from environment.Overcooked import Overcooked_multi
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

def define_environment(centralized):
    reward_config = {
        "metatask failed": -1,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -50,
        "step penalty": -0.5,
        "right step": 0.5,
    }
    env_params = {
        "centralized": centralized,
        "grid_dim": [7, 7],
        "task": "tomato salad",
        "rewardList": reward_config,
        "map_type": "A",
        "mode": "vector",
        "debug": False,
        "agents": ['ai', 'human'] if centralized else ['ai1', 'ai2', 'human'],
        "n_players": 3,
        "max_episode_length": 100,
    }

    env = Overcooked_multi(**env_params)
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
    p = f"{storage_path}/{'centralized' if args.centralized else 'decentralized'}_{args.name}_{args.rl_module}_*"
    experiment_name = glob.glob(p)[-1]
    print(f"Loading results from {experiment_name}...")
    restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
    result_grid = restored_tuner.get_results()
    best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
    print(best_result.config)
    best_checkpoint = best_result.checkpoint

    if args.centralized:
        ai_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'ai'
            )
        )
        human_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'human'
            )
        )

        return ai_module, human_module
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
        human_module = RLModule.from_checkpoint(os.path.join(
            best_checkpoint.path,
            COMPONENT_LEARNER_GROUP,
            COMPONENT_LEARNER,
            COMPONENT_RL_MODULE,
            'human'
            )
        )
        return ai1_module, ai2_module, human_module


def main(args):
    env = define_environment(args.centralized)

    if args.centralized:
        ai_module, human_module = load_modules(args)
    else:
        ai1_module, ai2_module, human_module = load_modules(args)
    env.game.on_init()
    obs, info = env.reset()
    env.render()

    if args.centralized:
        ai_action_space_shape = np.array([env.action_spaces["ai"].shape[0], env.action_spaces["ai"][0].n])

    while True:
        if args.centralized:
            ai_action = sample_action(ai_module, torch.from_numpy(obs['ai']).unsqueeze(0).float(), ai_action_space_shape)
            human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float(), None)
            action = {'ai': ai_action,
                    'human': human_action}
        else:
            ai1_action = sample_action(ai1_module, torch.from_numpy(obs['ai1']).unsqueeze(0).float(), None)
            ai2_action = sample_action(ai2_module, torch.from_numpy(obs['ai2']).unsqueeze(0).float(), None)
            human_action = sample_action(human_module, torch.from_numpy(obs['human']).unsqueeze(0).float(), None)
            action = {'ai1': ai1_action,
                    'ai2': ai2_action,
                    'human': human_action}

        obs, rewards, terminateds, _, _ = env.step(action)
        env.render()
        time.sleep(0.1)

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
