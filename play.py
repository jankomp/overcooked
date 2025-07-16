import argparse
import copy
import glob
import os

from ray import tune
from environment.Overcooked import Overcooked_multi
from Agents import *
import pandas as pd
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

from ray.rllib.utils.numpy import convert_to_numpy, softmax
import torch

TASKLIST = [
    "tomato salad", "lettuce salad", "onion salad",
    "lettuce-tomato salad", "onion-tomato salad",
    "lettuce-onion salad", "lettuce-onion-tomato salad"
]


class Player:
    ACTION_MAPPING = {
        "w": 3,
        "d": 0,
        "a": 2,
        "s": 1,
        "q": 4
    }

    REWARD_LIST = {
        "metatask failed": -1,
        "pretask finished": 5,
        "subtask finished": 10,
        "correct delivery": 200,
        "wrong delivery": -5,
        "step penalty": -0.5,
        "right step": 0.5,
    }

    def __init__(self, args):
        self.env_params = {
            'grid_dim': args['grid_dim'],
            'task': TASKLIST[args['task']],
            'rewardList': self.REWARD_LIST,
            'map_type': args['map_type'],
            'mode': args['mode'],
            "agents": ['ai', 'human'] if args['centralized'] else ['ai1', 'ai2', 'human'],
            "n_players": 3,
            'debug': args['debug'],
            'centralized': args['centralized'],
            "max_episode_length": 100,
            "randomized_items": 0,
            "randomized_agents": 0,
            "rotate_map": False
        }
        self.env = Overcooked_multi(**self.env_params)
        self.centralized = args['centralized']
        self.deterministic = args['deterministic']

        if args['agent'] == 'stationary':
            assert(not args['centralized'])
            self.agent = [AlwaysStationaryRLM(
                                observation_space=self.env.observation_spaces['ai1'],
                                action_space=self.env.action_spaces['ai1'],
                                inference_only=True
                            ),
                            AlwaysStationaryRLM(
                                    observation_space=self.env.observation_spaces['ai2'],
                                    action_space=self.env.action_spaces['ai2'],
                                    inference_only=True
                                )]

        elif args['agent'] == 'random':
            self.agent = RandomRLM(
                observation_space=self.env.observation_spaces['ai'],
                action_space=self.env.action_spaces['ai'],
                inference_only=True
            )

        elif args['agent'] == 'human':
            self.agent = 'human'

        elif args['agent'] == 'learned':
            self.agent = self.load_ai_modules(args['save_dir'], args['name'], args['rl_module'], args['centralized'])
        else:
            raise NotImplementedError(f'{args['agent']} is unknonw')


        self.rewards = 0
        self.discount = 1
        self.step = 0

    def sample_action(self, mdl, obs, action_space_shape):
        mdl_out = mdl.forward_inference({Columns.OBS: obs})
        if Columns.ACTION_DIST_INPUTS in mdl_out: #our custom policies might return the actions directly, while learned policies might return logits.
            logits = convert_to_numpy(mdl_out[Columns.ACTION_DIST_INPUTS])

            if action_space_shape is not None: # centralized control means one action per agent
                logits = np.reshape(logits, action_space_shape)
                if self.deterministic:
                    action = [np.argmax(agent_logits) for agent_logits in logits]
                else:
                    action = [np.random.choice(list(range(len(agent_logits))), p=softmax(agent_logits)) for agent_logits in logits]
            else:
                if self.deterministic:
                    action = np.argmax(logits[0])
                else:
                    action = np.random.choice(list(range(len(logits[0]))), p=softmax(logits[0]))
            return action
        elif 'actions' in mdl_out:
            return mdl_out['actions'][0]

        else:
            raise NotImplementedError("Something weird is going on when sampling acitons")

    def load_ai_modules(self, save_dir, name, rl_module, centralized):
        current_dir = os.getcwd()
        storage_path = os.path.join(current_dir, save_dir)
        p = f"{storage_path}/{'centralized' if centralized else 'decentralized'}_{name}_{rl_module}_*"        
        experiment_name = glob.glob(p)[-1]
        print(f"Loading results from {experiment_name}...")
        restored_tuner = tune.Tuner.restore(experiment_name, trainable="PPO")
        result_grid = restored_tuner.get_results()
        best_result = result_grid.get_best_result(metric=f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}", mode="max")
        print(best_result.config)
        best_checkpoint = best_result.checkpoint

        if centralized:
            ai_module = RLModule.from_checkpoint(os.path.join(
                best_checkpoint.path,
                COMPONENT_LEARNER_GROUP,
                COMPONENT_LEARNER,
                COMPONENT_RL_MODULE,
                'ai'
                )
            )

            return ai_module
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
            return [ai1_module, ai2_module]

    def run(self):
        self.env.game.on_init()
        new_obs, _ = self.env.reset()
        self.env.render()
        data = [["observation", "action_human", "action_ai1", "action_ai2", "new_obs", "reward_human", "reward_ai1", "reward_ai2", "done"]]


        if self.centralized:
            ai_action_space_shape = np.array([self.env.action_spaces["ai"].shape[0], self.env.action_spaces["ai"][0].n])

        while True:
            obs=new_obs
            row = [obs['human']]
            self.step += 1
            input_human = ""
            while input_human not in ['w', 'a', 's', 'd', 'q']:
                input_human = input("Input Human: ").strip().split(" ")[0]

            if input_human == ['p']:
                self.save_data(data)
                continue


            if self.agent == 'human':
                input_ai1 = input("Input AI1: ").strip().split(" ")[0]
                input_ai1 = self.ACTION_MAPPING[input_ai1]

                input_ai2 = input("Input AI2: ").strip().split(" ")[0]
                input_ai2 = self.ACTION_MAPPING[input_ai2]
            else:
                if self.centralized:
                    ai_action = self.sample_action(self.agent, torch.from_numpy(obs['ai']).unsqueeze(0).float(), ai_action_space_shape)
                else:
                    ai1_action = self.sample_action(self.agent[0], torch.from_numpy(obs['ai1']).unsqueeze(0).float(), None)
                    ai2_action = self.sample_action(self.agent[1], torch.from_numpy(obs['ai2']).unsqueeze(0).float(), None)
                    input_ai1 = ai1_action
                    input_ai2 = ai2_action

            if self.centralized:
                action = {
                    "ai": ai_action,
                    "human": self.ACTION_MAPPING[input_human],
                }
            else:
                action = {
                    "ai1": input_ai1,
                    "ai2": input_ai2,
                    "human": self.ACTION_MAPPING[input_human],
                }

            new_obs, reward, done, _, _ = self.env.step(action)

            log_action_ai1 = ai_action[0] if self.centralized else action['ai1']
            log_action_ai2 = ai_action[1] if self.centralized else action['ai2']
            log_reward_ai1 = reward['ai'] if self.centralized else action['ai1']
            log_reward_ai2 = reward['ai'] if self.centralized else action['ai2']

            row.append(action['human'])
            row.append(log_action_ai1)
            row.append(log_action_ai2)


            row.append(new_obs['human'])
            row.append(reward['human'])
            row.append(log_reward_ai1)
            row.append(log_reward_ai2)
            row.append(done['__all__'])

            data.append(copy.deepcopy(row))

            self.env.render()

            if done['__all__']:
                self.save_data(data)
                break

    def save_data(self, data):
        columns = data[0]
        # Extract data
        data = data[1:]
        print(f"steps: {len(data)}")
        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)
        # Save to CSV
        csv_filename = "output.csv"
        df.to_csv(csv_filename, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_dim', type=int, nargs=2, default=[5, 5], help='Grid world size')
    parser.add_argument('--task', type=int, default=6, help='The recipe agent cooks')
    parser.add_argument('--map_type', type=str, default="A", help='The type of map')
    parser.add_argument('--mode', type=str, default="vector", help='The type of observation (vector/image)')
    parser.add_argument('--debug', type=bool, default=True, help='Whether to print debug information and render')

    parser.add_argument('--agent', type=str, default='learned', help='Human, stationary, random, or learned')
    parser.add_argument("--save_dir", default="runs", type=str)
    parser.add_argument("--name", default="run", type=str)
    parser.add_argument("--rl_module", default="random", type=str)
    parser.add_argument("--centralized", action="store_true", help="True for centralized training, False for decentralized training")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic actions?")

    params = vars(parser.parse_args())

    player = Player(params)
    player.run()