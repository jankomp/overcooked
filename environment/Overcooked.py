import numpy as np
from environment.render.game import Game
from gymnasium import spaces
from .items import Tomato, Lettuce, Onion, Plate, Knife, Delivery, Agent, Food
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
from collections import Counter

DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ITEMNAME = ["space", "counter", "agent", "tomato", "lettuce", "plate", "knife", "delivery", "onion"]
ITEMIDX = {"space": 0, "counter": 1, "agent": 2, "tomato": 3, "lettuce": 4, "plate": 5, "knife": 6, "delivery": 7, "onion": 8}
AGENTCOLOR = ["robot-black", "robot-blue", "robot-green", "robot-orange", "robot-pink", "robot-red", "robot-yellow", "yellow"]
TASKLIST = ["tomato salad", "lettuce salad", "onion salad", "lettuce-tomato salad", "onion-tomato salad", "lettuce-onion salad", "lettuce-onion-tomato salad"]

class Overcooked_multi(MultiAgentEnv):

    def __init__(self, centralized, grid_dim, task, rewardList, map_type="A", mode="vector", debug=False, agents=["ai", "human"], n_players=2, max_episode_length=80):
        super().__init__()
        self.step_count = 0
        self.centralized = centralized
        self.agents = agents
        self.n_agents = n_players
        self.max_episode_length = max_episode_length

        self.players = [f"ai{i}" for i in range(n_players - 1)] + ["human"]
        self.obs_radius = 0 # full observability
        self.xlen, self.ylen = grid_dim

        self.task = task
        self.rewardList = rewardList
        self.mapType = map_type
        self.debug = debug
        self.mode = mode

        self.reward = None

        self.initMap = self._initialize_map()
        self.map = copy.deepcopy(self.initMap)

        self.oneHotTask = [1 if t in self.task else 0 for t in TASKLIST]

        self.taskCompletionStatus = [1 if element in [self.task] else 0 for element in TASKLIST]

        self._createItems()
        n_ais = n_players - 1 # there should always be 1 human agent and the rest ais
        
        if self.centralized:
            centralized_ai_action_space = spaces.MultiDiscrete([5]*n_ais)
            human_action_space = spaces.Discrete(5)
            self.action_spaces = {"ai": centralized_ai_action_space, "human": human_action_space} #TODO: find cleaner way to define action_spaces (not hard coded agent names)
            
            observations = self._get_obs()
            centralized_ai_observation_space = spaces.Box(low=0, high=1, shape=(len(observations["ai"]),), dtype=np.float64)
            human_observation_space = spaces.Box(low=0, high=1, shape=(len(observations["human"]),), dtype=np.float64)
            self.observation_spaces = {"ai": centralized_ai_observation_space, "human": human_observation_space} #TODO: find cleaner way to define observation_spaces (not hard coded agent names)
        else:
            self.action_spaces = {agent: spaces.Discrete(5) for agent in self.agents}
            self._initObs()
            self.observation_spaces = {agent: spaces.Box(low=0, high=1, shape=(len(self._get_obs()[agent]),), dtype=np.float64) for agent in self.agents}

        self.game = Game(self)

    def _initialize_map(self):
        if self.xlen == 3 and self.ylen == 3:
            return self._map_3x3()
        elif self.xlen == 5 and self.ylen == 5:
            return self._map_5x5()
        elif self.xlen == 3 and self.ylen == 5:
            return self._map_3x5()
        elif self.xlen == 7 and self.ylen == 7:
            return self._map_7x7()
        elif self.xlen == 9 and self.ylen == 9:
            return self._map_9x9()
        else:
            return []
    
    def _map_3x3(self):
        if self.n_agents == 2:
            if self.mapType == "A":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]] 
            elif self.mapType == "B":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]] 
            elif self.mapType == "C":
                map =  [[1, 3, 1],
                        [7, 2, 6],
                        [1, 5, 2]]
        elif self.n_agents == 3:
            if self.mapType == "A":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
            elif self.mapType == "B":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
            elif self.mapType == "C":
                map =  [[1, 3, 2],
                        [7, 2, 6],
                        [1, 5, 2]]
        return map
    
    def _map_5x5(self):
        if self.n_agents == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1],
                        [6, 0, 0, 2, 1],
                        [3, 0, 0, 0, 1],
                        [7, 0, 0, 2, 1],
                        [1, 5, 1, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 8, 1, 1, 1],
                        [6, 2, 1, 0, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 4, 1, 1, 1]] 
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        elif self.n_agents == 3:
            if self.mapType == "A":
                map =  [[1, 1, 5, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 6],
                        [7, 0, 2, 0, 1],
                        [1, 1, 5, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 1, 1, 1, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 2, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        return map
    
    def _map_3x5(self):
        if self.n_agents == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 1],
                        [7, 0, 0, 0, 1],
                        [1, 5, 1, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 8, 1, 1, 1],
                        [6, 2, 1, 0, 1],
                        [3, 0, 5, 2, 6]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
        elif self.n_agents == 3:
            if self.mapType == "A":
                map =  [[1, 1, 5, 1, 1],
                        [6, 2, 0, 2, 1],
                        [3, 0, 0, 0, 6],
                        [7, 0, 2, 0, 1],
                        [1, 1, 5, 1, 1]] 
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 2, 6],
                        [7, 0, 5, 0, 1],
                        [1, 1, 1, 1, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 5, 1],
                        [6, 2, 1, 2, 1],
                        [3, 0, 5, 0, 6],
                        [7, 2, 0, 0, 1],
                        [1, 1, 1, 1, 1]] 
            return map
        
    def _map_7x7(self):
        if self.n_agents == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 4],
                        [6, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 5, 1, 0, 1, 4, 1],
                        [1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 1, 0, 1, 0, 1],
                        [1, 0, 4, 1, 1, 0, 1],
                        [1, 0, 2, 1, 2, 0, 1],
                        [7, 0, 0, 1, 0, 0, 5],
                        [1, 6, 1, 1, 7, 6, 1]]  
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "D":
                map =  [[1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 2, 1, 2, 1, 1],
                        [1, 6, 0, 1, 0, 3, 1],
                        [1, 6, 0, 1, 0, 4, 1],
                        [1, 7, 0, 0, 0, 8, 1],
                        [1, 1, 1, 5, 5, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1]]
        elif self.n_agents == 3:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 2, 1, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]] 
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 1, 2, 0, 4],
                        [6, 0, 0, 1, 0, 0, 8],
                        [7, 0, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 5, 1]]
        return map

    def _map_9x9(self):
        if self.n_agents == 2:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 5, 1, 0, 1, 1, 4, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 1, 0, 1, 0, 0, 1],
                        [1, 0, 0, 4, 1, 7, 0, 0, 1],
                        [1, 0, 2, 0, 1, 0, 2, 0, 1],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 6, 1, 1, 1, 6, 5, 1, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        elif self.n_agents == 3:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 1, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 0, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
        elif self.n_agents == 4:
            if self.mapType == "A":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 0, 0, 2, 0, 4],
                        [6, 0, 0, 0, 0, 0, 0, 0, 8],
                        [7, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 0, 0, 0, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 2, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "B":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 1, 0, 2, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "C":
                map =  [[1, 1, 1, 1, 1, 1, 1, 3, 1],
                        [6, 0, 2, 0, 1, 0, 2, 0, 4],
                        [6, 0, 0, 0, 1, 0, 0, 0, 8],
                        [7, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 0, 0, 1, 0, 0, 0, 1],
                        [1, 0, 2, 0, 0, 0, 2, 0, 5],
                        [1, 1, 1, 1, 1, 1, 1, 5, 1]]
            elif self.mapType == "D":
                map =  [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 6, 2, 0, 1, 0, 2, 4, 1],
                        [1, 6, 0, 0, 1, 0, 0, 8, 1],
                        [1, 7, 0, 0, 1, 0, 0, 1, 1],
                        [1, 1, 0, 0, 1, 0, 0, 1, 1],
                        [1, 1, 2, 0, 0, 0, 2, 5, 1],
                        [1, 1, 1, 1, 1, 1, 5, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1]]
        return map

    def _createItems(self):
        """
        Initialize the items in the environment based on the map configuration.

        This function iterates through the map grid and creates instances of various items
        (e.g., agents, knives, delivery points, tomatoes, lettuce, onions, plates) at their
        respective positions. It populates the item dictionary and item list with these instances.

        The item dictionary (itemDic) maps item names to lists of item instances, excluding "space" and "counter".
        The item list (itemList) is a flattened list of all item instances.
        The agent list (agent) is a list of all agent instances.

        The function also assigns colors to agents based on their index.

        Returns:
            None
        """
        self.itemDic = {name: [] for name in ITEMNAME if name != "space" and name != "counter"}
        agent_idx = 0

        self.knife = []
        self.delivery = []
        self.tomato = []
        self.lettuce = []
        self.onion = []
        self.plate = []

        for x in range(self.xlen):
            for y in range(self.ylen):
                item_type = ITEMNAME[self.map[x][y]]
                if item_type == "agent":
                    self.itemDic[item_type].append(Agent(agent_idx, x, y, color=AGENTCOLOR[agent_idx - self.n_agents]))
                    agent_idx += 1
                elif item_type == "knife":
                    new_knife = Knife(x, y)
                    self.itemDic[item_type].append(new_knife)
                    self.knife.append(new_knife)
                elif item_type == "delivery":
                    new_delivery = Delivery(x, y)
                    self.itemDic[item_type].append(new_delivery)
                    self.delivery.append(new_delivery)
                elif item_type == "tomato":
                    new_tomato = Tomato(x, y)
                    self.itemDic[item_type].append(new_tomato)
                    self.tomato.append(new_tomato)
                elif item_type == "lettuce":
                    new_lettuce = Lettuce(x, y)
                    self.itemDic[item_type].append(new_lettuce)
                    self.lettuce.append(new_lettuce)
                elif item_type == "onion":
                    new_onion = Onion(x, y)
                    self.itemDic[item_type].append(new_onion)
                    self.onion.append(new_onion)
                elif item_type == "plate":
                    new_plate = Plate(x, y)
                    self.itemDic[item_type].append(new_plate)
                    self.plate.append(new_plate)

        self.itemList = [item for sublist in self.itemDic.values() for item in sublist]
        self.agent = self.itemDic["agent"]

    def _initObs(self):
        """
        Initialize the observations for the agents.

        This function creates an observation list by normalizing the positions of items
        and appending additional information if the item is of type Food. It then extends
        the observation list with one-hot encoded task information. Finally, it assigns
        the observation list to each agent and returns a list of observations for all agents.

        Returns:
            list: A list containing the observation arrays for each agent.
        """
        obs = []
        for item in self.itemList:
            obs.extend([item.x / self.xlen, item.y / self.ylen])
            if isinstance(item, Food):
                obs.append(item.cur_chopped_times / item.required_chopped_times)
        obs.extend(self.oneHotTask)

        for agent in self.agent:
            agent.obs = obs
        return [np.array(obs)] * self.n_agents

    def _get_vector_state(self):
        """
        Get the global state of the environment.

        This function creates a global state representation by normalizing the positions of items
        and appending additional information based on the item type. It includes the positions of
        all items, their chopped status if they are food, whether plates contain food, and whether
        knives or agents are holding items. The state is extended with one-hot encoded task information.

        Returns:
            list: A list containing the state arrays for each agent.
        """
        state = []
        for item in self.itemList:
            x = item.x / self.xlen
            y = item.y / self.ylen
            state.extend([x, y])
            
            if isinstance(item, Food):
                state.append(item.cur_chopped_times / item.required_chopped_times)
            elif isinstance(item, Plate):
                state.append(1 if item.containing else 0)
            elif isinstance(item, Knife):
                state.append(1 if item.holding else 0)
            elif isinstance(item, Agent):
                state.append(1 if item.holding else 0)

        state.extend(self.oneHotTask)
        return [np.array(state)] * self.n_agents

    def _get_image_state(self):
        """
        Retrieve the current image state for each agent.

        This method returns a list containing the current image observation 
        of the game state, repeated for each agent in the environment.

        Returns:
            list: A list where each element is the current image observation 
            of the game state, repeated for the number of agents.
        """
        return [self.game.get_image_obs()] * self.n_agents

    def _get_obs(self):
        """
        Returns
        -------
        obs : dict
            observation for each agent.
        """
        vec_obs = self._get_vector_obs()

        # If observation radius is greater than 0, agents have partial observability
        if self.obs_radius > 0:
            if self.mode == "vector":
                if self.centralized:
                	#TODO: take obs radius into account
                    return {"ai": np.array([agent_obs for agent_obs in vec_obs[:-1]]), "human": np.array([vec_obs[-1]]).squeeze()}
                else:                    
                    return {agent: np.asarray(vec_obs[i], dtype=np.float64) for i, agent in enumerate(self.agents)}
            elif self.mode == "image":
                return self._get_image_obs()
        else:
            # If observation radius is 0, agents have full observability
            if self.mode == "vector":
                if self.centralized:
                    obs = {"ai": np.array(vec_obs[:-1]).flatten(), "human": np.array(vec_obs[-1])}
                    return obs
                else:                    
                    return {agent: np.asarray(self._get_vector_state()[i], dtype=np.float64) for i, agent in enumerate(self.agents)}
            elif self.mode == "image":
                return self._get_image_state()

    def _get_vector_obs(self):

        """
        Returns
        -------
        vector_obs : list
            vector observation for each agent.
        """

        po_obs = []
        for agent in self.agent:
            obs = []
            idx = 0


            if self.xlen == 3 and self.ylen == 3:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1],
                                    [1, 0, 1],
                                    [1, 1, 1]]
            elif self.xlen == 5 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 3 and self.ylen == 5:
                if self.mapType == "A":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1]]
                elif self.mapType == "C":
                    agent.pomap =  [[1, 1, 1, 1, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 1, 0, 1],
                                    [1, 0, 0, 0, 1],
                                    [1, 1, 1, 1, 1]]
            elif self.xlen == 7 and self.ylen == 7:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    agent.pomap= [[1, 1, 1, 0, 1, 1, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 0, 1, 0, 1],
                                [1, 0, 1, 1, 1, 0, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1]]
            elif self.xlen == 9 and self.ylen == 9:
                if self.mapType == "A":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "B":
                    
                    agent.pomap= [[1, 1, 1, 1, 0, 1, 1, 1, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 0, 1, 0, 0, 1],
                                [1, 0, 0, 1, 1, 1, 0, 0, 1],
                                [1, 0, 2, 0, 1, 0, 2, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1]]
                elif self.mapType == "C":
                    agent.pomap= [[1, 1, 1, 1, 1, 1, 1, 1, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 1, 0, 0, 0, 1],
                                [1, 0, 0, 0, 0, 0, 0, 0, 1],
                                [1, 1, 1, 1, 1, 1, 1, 1, 1]]

            for item in self.itemList:
                # Check if the item is within the agent's observation radius or if the radius is 0 (full observability)
                if (agent.x - self.obs_radius <= item.x <= agent.x + self.obs_radius and
                    agent.y - self.obs_radius <= item.y <= agent.y + self.obs_radius) or self.obs_radius == 0:
                    # Normalize item position and add to observation
                    x = item.x / self.xlen
                    y = item.y / self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status to observation
                    if isinstance(item, Food):
                        obs.append(item.cur_chopped_times / item.required_chopped_times)
                        idx += 1
                else:
                    # If the item is outside the observation radius, use its initial position
                    x = agent.obs[idx] * self.xlen
                    y = agent.obs[idx + 1] * self.ylen
                    if not (agent.x - self.obs_radius <= x <= agent.x + self.obs_radius and
                            agent.y - self.obs_radius <= y <= agent.y + self.obs_radius):
                        x = item.initial_x
                        y = item.initial_y
                    x /= self.xlen
                    y /= self.ylen
                    obs.extend([x, y])
                    idx += 2
                    # If the item is food, add its chopped status from the agent's previous observation
                    if isinstance(item, Food):
                        obs.append(agent.obs[idx] / item.required_chopped_times)
                        idx += 1
                # Update the agent's partial observability map
                agent.pomap[int(x * self.xlen)][int(y * self.ylen)] = ITEMIDX[item.rawName]
            # Mark the agent's position on the partial observability map
            agent.pomap[agent.x][agent.y] = ITEMIDX["agent"]
            # Add one-hot encoded task information to the observation
            obs.extend(self.oneHotTask)
            agent.obs = obs
            po_obs.append(np.array(obs))
        return po_obs

    def _get_image_obs(self):

        """
        Returns
        -------
        image_obs : list
            image observation for each agent.
        """

        po_obs = []
        frame = self.game.get_image_obs()
        old_image_width, old_image_height, channels = frame.shape
        new_image_width = int((old_image_width / self.xlen) * (self.xlen + 2 * (self.obs_radius - 1)))
        new_image_height =  int((old_image_height / self.ylen) * (self.ylen + 2 * (self.obs_radius - 1)))
        color = (0,0,0)
        obs = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        obs[x_center:x_center+old_image_width, y_center:y_center+old_image_height] = frame

        for idx, agent in enumerate(self.agent):
            agent_obs = self._get_PO_obs(obs, agent.x, agent.y, old_image_width, old_image_height)
            po_obs.append(agent_obs)
        return po_obs

    def _get_PO_obs(self, obs, x, y, ori_width, ori_height):
        x1 = (x - 1) * int(ori_width / self.xlen)
        x2 = (x + self.obs_radius * 2) * int(ori_width / self.xlen)
        y1 = (y - 1) * int(ori_height / self.ylen)
        y2 = (y + self.obs_radius * 2) * int(ori_height / self.ylen)
        return obs[x1:x2, y1:y2]

    def _findItem(self, x, y, itemName):
        for item in self.itemDic[itemName]:
            if item.x == x and item.y == y:
                return item
        return None

    @property
    def state_size(self):
        return self.get_state().shape[0]

    @property
    def obs_size(self):
        return [self.observation_space.shape[0]] * self.n_agents

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    def reset(self, *, seed=None, options=None):
        """
        Returns
        -------
        obs : list
            observation for each agent.
        """
        #TODO: change to task index and randomize locations of items
        self.map = copy.deepcopy(self.initMap)
        self._createItems()
        self.step_count = 0

        # Reset task completion status
        self.taskCompletionStatus = [1 if element in [self.task] else 0 for element in TASKLIST]

        self._initObs()

        return self._get_obs(), {}
    
    def step(self, action):
        if self.centralized:
            action = self._split_centralized_actions(action)

        action_list = [action[key] for key in action]
        self.step_count += 1
        self.reward = self.rewardList["step penalty"]
        done = False

        self._reset_agent_movement_flags()
        info = self._initialize_info(action_list)

        if self.debug:
            print("in overcooked primitive actions:", action_list)

        while not self._all_agents_moved():
            for idx, agent in enumerate(self.agent):
                if agent.moved:
                    continue
                self._execute_agent_action(agent, action_list, idx)

        done = done or self._check_task_completion()
        terminateds = {"__all__": done or self.step_count >= self.max_episode_length}
        rewards = {agent: self.reward for agent in self.agents}
        infos = {agent: info for agent in self.agents}
        truncated = False

        if self.debug:
            print("rewards:", rewards)

        return self._get_obs(), rewards, terminateds, {'__all__': truncated}, infos
    
    def _split_centralized_actions(self, action):
        player_actions = {player: action["ai"][idx] for idx, player in enumerate(self.players[:-1])}
        player_actions["human"] = action["human"]
        return player_actions
    
    def _execute_agent_action(self, agent, action_list, idx):
        agent.moved = True

        if action_list[idx] < 4:
            self._attempt_move(agent, action_list, idx)
        else:
            for x, y in DIRECTION:
                self._handle_interaction(agent, agent.x + x, agent.y + y)

    def _attempt_move(self, agent, action_list, idx):
        direction = action_list[idx]
        target_x = agent.x + DIRECTION[direction][0]
        target_y = agent.y + DIRECTION[direction][1]
        target_name = ITEMNAME[self.map[target_x][target_y]]

        if target_name == "space":
            self._move_agent(agent, target_x, target_y)
        elif target_name == "agent":
            self._handle_agent_collision(agent, action_list, target_x, target_y)
        else:
            self._handle_interaction(agent, target_x, target_y)


    def _reset_agent_movement_flags(self):
        for agent in self.agent:
            agent.moved = False

    def _initialize_info(self, action):
        return {
            'cur_mac': action,
            'mac_done': [True] * self.n_agents,
            'collision': []
        }

    def _all_agents_moved(self):
        return all(agent.moved for agent in self.agent)
    
    def _check_task_completion(self):
        if self.debug:
            print("task completion status:", self.taskCompletionStatus)
        return all(value == 0 for value in self.taskCompletionStatus)


    def _move_agent(self, agent, tx, ty):
        self.map[agent.x][agent.y] = ITEMIDX["space"]
        agent.move(tx, ty)
        self.map[tx][ty] = ITEMIDX["agent"]

    def _handle_agent_collision(self, agent, action_list, tx, ty):
        target_player = self._findItem(tx, ty, 'agent')
        if not target_player.moved:
            agent.moved = False
            target_player_action = action_list[target_player.idx]
            if target_player_action < 4:
                new_target_player_x = target_player.x + DIRECTION[target_player_action][0]
                new_target_player_y = target_player.y + DIRECTION[target_player_action][1]
                if new_target_player_x == agent.x and new_target_player_y == agent.y:
                    target_player.move(new_target_player_x, new_target_player_y)
                    agent.move(tx, ty)
                    agent.moved = True
                    target_player.moved = True

    def _handle_interaction(self, agent, tx, ty):
        target_name = ITEMNAME[self.map[tx][ty]]

        if not agent.holding:
            self._try_pickup(agent, tx, ty, target_name)
        else:
            self._try_putdown(agent, tx, ty, target_name)

    def _try_pickup(self, agent, tx, ty, target_name):
        if target_name in ["tomato", "lettuce", "plate", "onion"]:
            item = self._findItem(tx, ty, target_name)
            agent.pickup(item)
            self.map[tx][ty] = ITEMIDX["counter"]

        elif target_name == "knife":
            knife = self._findItem(tx, ty, target_name)
            if isinstance(knife.holding, Plate):
                item = knife.holding
                knife.release()
                agent.pickup(item)
            elif isinstance(knife.holding, Food):
                if knife.holding.chopped:
                    item = knife.holding
                    knife.release()
                    agent.pickup(item)
                else:
                    knife.holding.chop()
                    self.reward += self.rewardList["subtask finished"]
                    if knife.holding.chopped:
                        if any(knife.holding.rawName in task for task in self.task):
                            self.reward += self.rewardList["subtask finished"]

    def _try_putdown(self, agent, tx, ty, target_name):
        item = agent.holding

        if target_name == "counter":
            if item.rawName in ["tomato", "lettuce", "onion", "plate"]:
                self.map[tx][ty] = ITEMIDX[item.rawName]
            agent.putdown(tx, ty)
            self.reward += self.rewardList["metatask failed"]

        elif target_name == "plate":
            if isinstance(item, Food) and item.chopped:
                plate = self._findItem(tx, ty, target_name)
                agent.putdown(tx, ty)
                plate.contain(item)
                self.reward += self.rewardList["subtask finished"]
            else:
                self.reward += self.rewardList["metatask failed"]

        elif target_name == "knife":
            self._handle_putdown_on_knife(agent, tx, ty, target_name)

        elif target_name == "delivery":
            self._handle_delivery(agent, tx, ty)

        elif target_name in ["tomato", "lettuce", "onion"]:
            food_item = self._findItem(tx, ty, target_name)
            if food_item.chopped and isinstance(item, Plate):
                item.contain(food_item)
                self.map[tx][ty] = ITEMIDX["counter"]
                self.reward += self.rewardList["subtask finished"]
            else:
                self.reward += self.rewardList["metatask failed"]

    def _handle_putdown_on_knife(self, agent, tx, ty, target_name):
        knife = self._findItem(tx, ty, target_name)
        # If the knife is empty, place the item on the knife
        if not knife.holding:
            item = agent.holding
            agent.putdown(tx, ty)
            knife.hold(item)
            if isinstance(item, Food):
                if item.chopped:
                    # Penalty for placing chopped food back on the knife
                    self.reward += self.rewardList["metatask failed"]
                else:
                    # Reward for placing unchopped food on the knife
                    self.reward += self.rewardList["subtask finished"]
            else:
                self.reward += self.rewardList["metatask failed"]
        # If the knife is holding food and the agent is holding a plate, place the food on the plate
        elif isinstance(knife.holding, Food) and isinstance(agent.holding, Plate):
            item = knife.holding
            if item.chopped:
                self.reward += self.rewardList["subtask finished"]
                knife.release()
                agent.holding.contain(item)
            else:
                # Penalty for placing unchopped food on a plate
                self.reward += self.rewardList["metatask failed"]
        elif isinstance(knife.holding, Food) and not isinstance(agent.holding, Plate):
            # Penalty for holding food while the knife is holding food
            self.reward += self.rewardList["metatask failed"]
        # If the knife is holding a plate and the agent is holding food, place the food on the plate
        elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Food):
            plate_item = knife.holding
            food_item = agent.holding
            if food_item.chopped:
                self.reward += self.rewardList["subtask finished"]
                knife.release()
                agent.pickup(plate_item)
                agent.holding.contain(food_item)
            else:
                # Penalty for placing unchopped food on a plate
                self.reward += self.rewardList["metatask failed"]
        elif isinstance(knife.holding, Plate) and isinstance(agent.holding, Plate):
            # Penalty for holding a plate while the knife is holding a plate
            self.reward += self.rewardList["metatask failed"]

    def _handle_delivery(self, agent, tx, ty):
        item = agent.holding

        if isinstance(item, Plate):
            if item.containing:
                dish_name = self._compose_dish_name(item.containing)
                if dish_name in self.task:
                    self._finalize_delivery(agent, item, dish_name, tx, ty)
                else:
                    self._fail_delivery(agent, item, tx, ty)
            else:
                self._fail_delivery(agent, item, tx, ty)

        else:
            self._fail_delivery(agent, item, tx, ty)

    def _compose_dish_name(self, contents):
        food_types = [Lettuce, Onion, Tomato]
        found = [-1] * len(food_types)

        for idx, f in enumerate(contents):
            for i, t in enumerate(food_types):
                if isinstance(f, t):
                    found[i] = idx

        names = [contents[i].rawName for i in found if i >= 0]
        return "-".join(names) + " salad"

    def _finalize_delivery(self, agent, plate, dish_name, tx, ty):
        idx = TASKLIST.index(dish_name)
        if self.taskCompletionStatus[idx] > 0:
            self.taskCompletionStatus[idx] -= 1
            self.reward += self.rewardList["correct delivery"]
        else:
            self.reward += self.rewardList["wrong delivery"]

        agent.putdown(tx, ty)
        for f in plate.containing:
            f.refresh()
            self.map[f.x][f.y] = ITEMIDX[f.rawName]
        plate.release()
        plate.refresh()
        self.map[plate.x][plate.y] = ITEMIDX[plate.name]

        if self._check_task_completion():
            self.reward += self.rewardList["correct delivery"]
            self.done = True

    def _fail_delivery(self, agent, item, tx, ty):
        self.reward += self.rewardList["wrong delivery"]
        agent.putdown(tx, ty)
        if isinstance(item, Plate):
            item.refresh()
            self.map[item.x][item.y] = ITEMIDX[item.name]
        elif isinstance(item, Food):
            item.refresh()
            self.map[item.x][item.y] = ITEMIDX[item.rawName]

    def render(self, mode='human'):
        return self.game.on_render()

