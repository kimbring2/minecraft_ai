import os
from collections import defaultdict
import numpy as np
import minerl
import time
from typing import List
import collections
import os
import cv2
import gym
from minerl.data import DataPipeline
import sys
import time
from collections import deque, defaultdict
from enum import Enum

mapping = dict()

def register(name):
    def _thunk(func):
        mapping[name] = func
        return func
    return _thunk


def get_discretizer(name):
    if callable(name):
        return name
    elif name in mapping:
        return mapping[name]
    else:
        raise ValueError('Registered wrappers:', ', '.join(mapping.keys()))


class LoopCraftingAgent:
    """
    Agent that acts according to the chain
    """
    def __init__(self, crafting_actions):
        """
        :param crafting_actions: list of crafting actions list({},...)
        """
        self.crafting_actions = crafting_actions
        self.current_action_index = 0

    def get_crafting_action(self):
        """
        :return: action to be taken
        """
        if len(self.crafting_actions) == 0:
            return {}

        result = self.crafting_actions[self.current_action_index]

        # move the pointer to the next action in the list
        self.current_action_index = (self.current_action_index + 1) % len(self.crafting_actions)

        return result

    def reset_index(self):
        self.current_action_index = 0


class CraftInnerWrapper(gym.Wrapper):
    """
    Wrapper for crafting actions
    """
    def __init__(self, env, crafts_agent):
        """
        :param env: env to wrap
        :param crafts_agent: instance of LoopCraftingAgent
        """
        super().__init__(env)
        self.crafts_agent = crafts_agent

    def step(self, action):
        """
        mix craft action with POV action
        :param action: POV action
        :return:
        """
        craft_action = self.crafts_agent.get_crafting_action()
        action = {**action, **craft_action}
        observation, reward, done, info = self.env.step(action)

        return observation, reward, done, info
        

class ObtainPoVWrapper(gym.ObservationWrapper):
    """Obtain 'pov' value (current game display) of the original observation."""
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = self.env.observation_space.spaces['pov']

    def observation(self, observation):
        #return observation['pov']
        return observation

   
class DiscreteBase(gym.Wrapper):
    def __init__(self, env):
        super(DiscreteBase, self).__init__(env)
        self.action_dict = {}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))

    def step(self, action):
        #print("self.action_dict[action]: " + str(self.action_dict[action]))
        s, r, done, info = self.env.step(self.action_dict[action])
        return s, r, done, info

    def sample_action(self):
        return self.action_space.sample()


class ItemAgentNode:
    """
    combined info about each agent
    """
    def __init__(self, node_name, count_, pov_agent, crafting_agent):
        self.name = node_name
        self.count = count_
        self.pov_agent = pov_agent
        self.crafting_agent = crafting_agent
        self.success = deque([0], maxlen=10)
        self.eps_to_save = 0
        self.model_dir = 'train/' + self.name
        self.exploration_force = True
        self.fixed = False

    def load_agent(self, load_dir=None):
        if load_dir is None:
            load_dir = self.model_dir

        self.pov_agent.load_agent(load_dir)


# "craft": "Enum(crafting_table,none,planks,stick,torch)",
# "equip": "Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe)",
# "nearbyCraft": "Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe)",
# "nearbySmelt": "Enum(coal,iron_ingot,none)",
# "place": "Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch)",
class craft(Enum):
    crafting_table = 0
    none = 1
    planks = 2
    stick = 3
    torch = 4


class equip(Enum):
    air = 0
    iron_axe = 1
    iron_pickaxe = 2 
    none = 3
    stone_axe = 4
    stone_pickaxe = 5
    wooden_axe = 6
    wooden_pickaxe = 7


class nearbyCraft(Enum):
    furnace = 0
    iron_axe = 1
    iron_pickaxe = 2
    none = 3
    stone_axe = 4 
    stone_pickaxe = 5
    wooden_axe = 6
    wooden_pickaxe = 7


class nearbySmelt(Enum):
    coal = 0
    iron_ingot = 1
    none = 2


class place(Enum):
    cobblestone = 0
    crafting_table = 1
    dirt = 2
    furnace = 3
    none = 4
    stone = 5
    torch = 6


def is_item(name):
    """
    method to differ actions and items
    :param name:
    :return:
    """
    return len(name.split(":")) == 2    


def get_crafting_actions_from_chain(chain_, node_name_):
    """
    getting crafting actions from chain for node_name_ item
    :param chain_:
    :param node_name_: item
    :return:
    """
    previous_actions = []
    for vertex in chain_:
        #print("vertex: ", vertex)
        #print("")
        
        if vertex == node_name_:
            break

        if not is_item(vertex):
            previous_actions.append(vertex)
        else:
            previous_actions = []
            
    #print("len(previous_actions): ", len(previous_actions))
    #print("")
            
    return [str_to_action_dict(action_) for action_ in previous_actions]


def str_to_action_dict(action_):
    """
    str -> dict
    :param action_:
    :return:
    """
    a_, _, value = action_.split(":")

    if a_ == 'craft':      
        value = craft[value].value
    elif a_ == 'equip':
        value = equip[value].value
    elif a_ == 'nearbyCraft':
        value = nearbyCraft[value].value
    elif a_ == 'nearbySmelt':
        value = nearbySmelt[value].value
    elif a_ == 'place':
        value = place[value].value

    #if a_ == 'craft': 
    #    if value == 0:
    #        print("value: ", value)
            #print("int(craft[value].value): ", int(craft[value].value))
    #        print("")
    
    return {a_: int(value)}    

    
def create_nodes(chain):
    nodes_names = [item for item in chain if is_item(item)]
    #print("nodes_names: " + str(nodes_names))

    craft_agents = []
    for node_name in nodes_names:
        craft_agents.append(LoopCraftingAgent(get_crafting_actions_from_chain(chain, node_name)))

    nodes_dict = {}
    nodes = []
    for index, (name, count) in enumerate([_.split(":") for _ in nodes_names]):
        if name not in nodes_dict.keys():
            nodes_dict[name] = ItemAgentNode(node_name=name,
                                             count_=int(count),
                                             pov_agent=None,
                                             crafting_agent=craft_agents[index])

        nodes.append(nodes_dict[name])

    return nodes
    

class DummyDataLoader:
     def __init__(self, data, items_to_add):
         self.data = data
         self.items_to_add = items_to_add
            
     def batch_iter(self, *args, **kwargs):
         for item in self.items_to_add:
             #print("item: " + str(item))
             for slice_ in self.data[item]:
                 #print("slice_: " + str(slice_))
                 yield slice_


class AbstractItemOrAction(dict):
    def __init__(self, name, value):
        super().__init__()
        self.name = name
        self.value = value

    @property
    def name(self):
        return self.__getitem__('name')

    @name.setter
    def name(self, value):
        self.__setitem__('name', value)

    @property
    def value(self):
        return self.__getitem__('value')

    @value.setter
    def value(self, value):
        self.__setitem__('value', value)

    def is_item(self):
        return self.get('type') == 'item'

    def is_action(self):
        return self.get('type') == 'action'


class Action(AbstractItemOrAction):
    def __init__(self, name, value):
        super().__init__(name, value)
        self.__setitem__('type', 'action')

    def is_noop(self):
        return not self.value


class Item(AbstractItemOrAction):
    def __init__(self, name: str, value: int, begin: int, end: int, actions: List[Action] = ()):
        super().__init__(name, value)
        self.actions = actions
        self.begin = begin
        self.end = end
        self.__setitem__('type', 'item')

    @property
    def actions(self) -> List[Action]:
        return self.__getitem__('actions')

    @actions.setter
    def actions(self, value: List[Action]):
        self.__setitem__('actions', value)

    def get_last_action(self) -> Action:
        actions = self.__getitem__('actions')
        if actions:
            return actions[-1]
        else:
            return None

    def add_action(self, action: Action):
        self.actions = (*self.actions, action)

    @property
    def begin(self):
        return self.__getitem__('begin')

    @begin.setter
    def begin(self, value):
        self.__setitem__('begin', value)

    @property
    def end(self):
        return self.__getitem__('end')

    @end.setter
    def end(self, value):
        self.__setitem__('end', value)


class ChainInfo:
    def __init__(self, chain, reward, env_name, trajectory_name, id_, length, time_indexes):
        self.chain = chain
        self.reward = reward
        self.env = env_name
        self.trajectory_name = trajectory_name
        self.id = id_
        self.length = length
        self.time_indexes = time_indexes

    def __str__(self):
        return str(self.reward) + "\n" + str(self.chain)


class TrajectoryDataPipeline:
    """
    number of tools to load trajectory
    """
    @staticmethod
    def get_trajectory_names(data_dir):
        # noinspection PyProtectedMember
        result = [os.path.basename(x) for x in DataPipeline._get_all_valid_recordings(data_dir)]

        return sorted(result)

    @staticmethod
    def map_to_dict(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):
        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
            if isinstance(gym_space, gym.spaces.Dict):
                dont_count = False
                inner_dict = collections.OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):

                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)

                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]

                return i + 1

        result = collections.OrderedDict()
        index = 0

        inventory_key_list = ['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 
        'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 'torch', 'wooden_axe', 'wooden_pickaxe', 'pov']

        #for inventory_key in inventory_key_list:
        #    result['inventory'][inventory_key] = 0

        equipped_items_key_list = ['equipped_items.mainhand.damage', 'equipped_items.mainhand.maxDamage', 'equipped_items.mainhand.type']
        key_index = 0
        for key, space in target_space.spaces.items():
            if key in ignore_keys:
                continue
            
            if key == 'pov':
                index = _map_to_dict(index, handler_list, key, space, result)

            if key == 'inventory':
                for inventory_key, inventory_space in space.spaces.items():
                    if key in ignore_keys:
                        continue

                    index = _map_to_dict(index, handler_list, inventory_key, inventory_space, result)

        return result

    @staticmethod
    def map_to_dict_act(handler_list: list, target_space: gym.spaces.space, ignore_keys=()):
        def _map_to_dict(i: int, src: list, key: str, gym_space: gym.spaces.space, dst: dict):
            if isinstance(gym_space, gym.spaces.Dict):
                dont_count = False
                inner_dict = collections.OrderedDict()
                for idx, (k, s) in enumerate(gym_space.spaces.items()):
                    if key in ['equipped_items', 'mainhand']:
                        dont_count = True
                        i = _map_to_dict(i, src, k, s, inner_dict)
                    else:
                        _map_to_dict(idx, src[i].T, k, s, inner_dict)

                dst[key] = inner_dict
                if dont_count:
                    return i
                else:
                    return i + 1
            else:
                dst[key] = src[i]

                return i + 1

        result = collections.OrderedDict()
        index = 0

        '''
        actions: ['action$forward', 'action$left', 'action$back', 'action$right', 'action$jump', 'action$sneak', 'action$sprint', 
        'action$attack', 'action$camera', 'action$place', 'action$equip', 'action$craft', 'action$nearbyCraft', 'action$nearbySmelt']

        target_space.spaces.items(): odict_items([('attack', Discrete(2)), ('back', Discrete(2)), ('camera', Box(low=-180.0, high=180.0, shape=(2,))), 
        ('craft', Discrete(5)), ('equip', Discrete(8)), ('forward', Discrete(2)), ('jump', Discrete(2)), ('left', Discrete(2)), ('nearbyCraft', Discrete(8)), 
        ('nearbySmelt', Discrete(3)), ('place', Discrete(7)), ('right', Discrete(2)), ('sneak', Discrete(2)), ('sprint', Discrete(2))])
        '''
        key_list = ['forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack', 'camera', 'place', 'equip', 'craft', 'nearbyCraft', 'nearbySmelt']
        key_index = 0
        for key, space in target_space.spaces.items():
            key = key_list[key_index]
            key_index += 1

            if key in ignore_keys:
                continue

            index = _map_to_dict(index, handler_list, key, space, result)

        return result

    @staticmethod
    def load_video_frames(video_path, suffix_size):
        cap = cv2.VideoCapture(video_path)
        ret, frame_num = True, 0
        while ret:
            ret, _ = DataPipeline.read_frame(cap)
            if ret:
                frame_num += 1

        num_states = suffix_size
        frames = []
        max_frame_num = frame_num
        frame_num = 0

        # Advance video capture past first i-frame to start of experiment
        cap = cv2.VideoCapture(video_path)
        for _ in range(max_frame_num - num_states):
            ret, _ = DataPipeline.read_frame(cap)
            frame_num += 1
            if not ret:
                return None

        while ret and frame_num < max_frame_num:
            ret, frame = DataPipeline.read_frame(cap)
            frames.append(frame)
            frame_num += 1

        return frames

    # noinspection PyProtectedMember
    @classmethod
    def load_data(cls, file_dir, ignore_keys=()):
        numpy_path = str(os.path.join(file_dir, 'rendered.npz'))
        video_path = str(os.path.join(file_dir, 'recording.mp4'))

        state = np.load(numpy_path, allow_pickle=True)

        reward_vec = state['reward']
        frames = cls.load_video_frames(video_path=video_path, suffix_size=len(reward_vec) + 1)

        action_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('action')])

        actions = list(action_dict.keys())

        action_data = [None for _ in actions]
        for i, key in enumerate(actions):
            action_data[i] = np.asanyarray(action_dict[key])

        obs_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('observation$inventory$')])

        obs = list(obs_dict.keys())

        current_observation_data = [None for _ in obs]

        next_observation_data = [None for _ in obs]

        reward_vec = state['reward']
        reward_data = np.asanyarray(reward_vec, dtype=np.float32)
        done_data = [False for _ in range(len(reward_data))]
        done_data[-1] = True

        info_dict = collections.OrderedDict([(key, state[key]) for key in state if key.startswith('observation$inventory$')])

        observables = list(info_dict.keys()).copy()
        if 'pov' not in ignore_keys:
            observables.append('pov')
            current_observation_data.append(None)
            next_observation_data.append(None)

        #print("observables: " + str(observables))
        #observables.append('pov')
        if 'pov' not in ignore_keys:
            frames = cls.load_video_frames(video_path=video_path, suffix_size=len(reward_vec) + 1)
        else:
            frames = None

        for i, key in enumerate(observables):
            if key in ignore_keys:
                continue

            if key == 'pov':
                current_observation_data[i] = np.asanyarray(frames[:-1])
                next_observation_data[i] = np.asanyarray(frames[1:])
            #else:
                #print("current_observation_data" + "[" + str(i) + "]" + ": " + str(current_observation_data[i]))
                #print("info_dict[key][:-1]: " + str(info_dict[key][:-1]))
            else:
                current_observation_data[i] = np.asanyarray(info_dict[key][:-1])
                next_observation_data[i] = np.asanyarray(info_dict[key][1:])

        gym_spec = gym.envs.registration.spec('MineRLObtainDiamond-v0')
        observation_dict = cls.map_to_dict(current_observation_data, gym_spec._kwargs['observation_space'], ignore_keys)
        action_dict = cls.map_to_dict_act(action_data, gym_spec._kwargs['action_space'])
        next_observation_dict = cls.map_to_dict(next_observation_data, gym_spec._kwargs['observation_space'], ignore_keys)

        #print("action_dict: " + str(action_dict))
        return [observation_dict, action_dict, reward_data, next_observation_dict, done_data]

    @classmethod
    def load_data_no_pov(cls, file_dir):
        return cls.load_data(file_dir, ignore_keys=('pov',))


class VisTools:
    """
    number of methods to draw chains with pyGraphviz
    """

    @staticmethod
    def get_all_vertexes_from_edges(edges):
        """
        determines all vertex of a graph
        :param edges: list of edges
        :return: list of vertexes
        """
        vertexes = []
        for left, right in edges:
            if left not in vertexes:
                vertexes.append(left)
            if right not in vertexes:
                vertexes.append(right)

        return vertexes

    @staticmethod
    def get_colored_vertexes(chain):
        """
        determines the color for each vertex
        item + its actions have the same color
        :param chain:
        :return: {vertex: color}
        """
        vertexes = VisTools.get_all_vertexes_from_edges(chain)
        result = {}
        colors = ["#ffe6cc", "#ccffe6"]
        current_color = 0

        for vertex in vertexes:
            result[vertex] = colors[current_color]

            bool_ = True
            for action in ["equip", "craft", "nearbyCraft", "nearbySmelt", 'place']:
                if action + ":" in vertex:
                    bool_ = False
            if bool_:
                current_color = (current_color + 1) % len(colors)

        return result

    @staticmethod
    def replace_with_name(name):
        """
        replace all names with human readable variants
        :param name: crafting action or item (item string will be skipped)
        :return: human readable name
        """
        if len(name.split(":")) == 3:
            name, order, digit = name.split(":")
            name = name + ":" + order
            
            # "craft": "Enum(crafting_table,none,planks,stick,torch)"
            # "equip": "Enum(air,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe)"
            # "nearbyCraft": "Enum(furnace,iron_axe,iron_pickaxe,none,stone_axe,stone_pickaxe,wooden_axe,wooden_pickaxe)"
            # "nearbySmelt": "Enum(coal,iron_ingot,none)"
            # "place": "Enum(cobblestone,crafting_table,dirt,furnace,none,stone,torch)"
            
            translate = {"place": ["cobblestone", "crafting_table", "dirt", "furnace", "none", "stone", "torch"],
                         "nearbySmelt": ["coal", "iron_ingot", "none"],
                         "nearbyCraft": ["furnace", "iron_axe", "iron_pickaxe", "none", "stone_axe",
                                         "stone_pickaxe", "wooden_axe", "wooden_pickaxe"],
                         "equip": ["air", "iron_axe", "iron_pickaxe", "none", "stone_axe", "stone_pickaxe",
                                   "wooden_axe", "wooden_pickaxe"],
                         "craft": ["crafting_table", "none", "planks", "stick", "torch"],
                         }
            name_without_digits = name
            while name_without_digits not in translate:
                name_without_digits = name_without_digits[:-1]

            return name + " -> " + translate[name_without_digits][int(digit)]
        else:
            return name

    @staticmethod
    def draw_graph(file_name, graph, format_="svg", vertex_colors=None):
        """
        drawing png graph from the list of edges
        :param vertex_colors:
        :param format_: resulted file format
        :param file_name: file_name
        :param graph: graph file with format: (left_edge, right_edge) or (left_edge, right_edge, label)
        :return: None
        """
        import pygraphviz as pgv
        g_out = pgv.AGraph(strict=False, directed=True)
        for i in graph:
            g_out.add_edge(i[0], i[1], color='black')
            edge = g_out.get_edge(i[0], i[1])
            if len(i) > 2:
                edge.attr['label'] = i[2]

        g_out.node_attr['style'] = 'filled'
        if vertex_colors:
            for vertex, color in vertex_colors.items():
                g_out.get_node(vertex).attr['fillcolor'] = color

        g_out.layout(prog='dot')
        g_out.draw(path="{file_name}.{format_}".format(**locals()))

    @staticmethod
    def save_chain_in_graph(chain_to_save, name="out", format_="png"):
        """
        saving image of a graph using draw_graph method
        :param chain_to_save:
        :param name: filename
        :param format_: file type e.g. ".png" or ".svg"
        :return:
        """
        graph = []

        for c_index, item in enumerate(chain_to_save):
            if c_index:
                graph.append(
                    [str(c_index) + '\n' + VisTools.replace_with_name((chain_to_save[c_index - 1])),
                     str(c_index + 1) + '\n' + VisTools.replace_with_name(item)])

        VisTools.draw_graph(name, graph=graph, format_=format_,
                            vertex_colors=VisTools.get_colored_vertexes(graph))
    

class TrajectoryInformation:
    def __init__(self, path_to_trajectory, trajectory=None):
        self.path_to_trajectory = path_to_trajectory
        self.trajectory_name = os.path.basename(path_to_trajectory)
        if trajectory is None:
            trajectory = TrajectoryDataPipeline.load_data_no_pov(self.path_to_trajectory)

        state, action, reward, next_state, done = trajectory
        self.chain = self.extract_subtasks(trajectory)
        self.reward = int(sum(reward))
        self.length = len(reward)

    def __str__(self):
        return self.path_to_trajectory + '\n' + str(self.chain)

    @classmethod
    def extract_from_dict(cls, dictionary, left, right):
        result = dict()
        for key, value in dictionary.items():
            if isinstance(value, dict):
                result[key] = cls.extract_from_dict(value, left, right)
            else:
                result[key] = value[left:right]

        return result

    def slice_trajectory_by_item(self, trajectory):
        if trajectory is None:
            trajectory = TrajectoryDataPipeline.load_data(self.path_to_trajectory)

        state, action, reward, next_state, done = trajectory
        #print("action: ", action)
        if self.length != len(reward):
            print(self.length, len(reward))
            raise NameError("Please, double check trajectory")

        result = defaultdict(list)
        for item in self.chain:
            # skip short ones
            if item.end - item.begin < 4:
                continue

            sliced_state = self.extract_from_dict(state, item.begin, item.end)
            sliced_action = self.extract_from_dict(action, item.begin, item.end)
            #print("sliced_action.keys(): ", sliced_action.keys()) 
        
            sliced_reward = reward[item.begin:item.end]
            sliced_next_state = self.extract_from_dict(next_state, item.begin, item.end)
            sliced_done = done[item.begin:item.end]
            result[item.name].append([sliced_state, sliced_action, sliced_reward, sliced_next_state, sliced_done])

        return result

    @staticmethod
    def to_old_chain_format(items: List[Item], return_time_indexes: bool):
        result = []
        used_actions = defaultdict(int)
        for item in items:
            for action in item.actions:
                full_action = f"{action.name}{action.value}"
                result.append(f"{action.name}:{used_actions[full_action]}:{action.value}")
                used_actions[full_action] += 1

            result.append(f"{item.name}:{item.value}")

        time_indexes = [(f"{item.name}+{item.value}", item.begin, item.end) for item in items]
        if return_time_indexes:
            return result, time_indexes

        return result

    @classmethod
    def compute_item_order(cls, trajectory, return_time_indexes=False, ):
        return cls.to_old_chain_format(cls.extract_subtasks(trajectory), return_time_indexes=return_time_indexes)

    @classmethod
    def extract_subtasks(cls, trajectory,
                         excluded_actions=("attack", "back", "camera",
                                           "forward", "jump", "left",
                                           "right", "sneak", "sprint"),
                         item_appear_limit=4) -> List[Item]:
        """
        computes item and actions order in time order
        :param trajectory:
        :param excluded_actions: by default all POV actions is excluded
        :param item_appear_limit: filter item vertexes appeared more then item_appear_limit times
        :return:
        """
        states, actions, rewards, next_states, _ = trajectory
        for index in range(len(rewards)):
            for action in actions:
                if action not in excluded_actions:
                    a = Action(name=action, value=actions[action][index])
        
        items = states.keys()
        
        # add auxiliary items to deal with crafting actions
        empty_item = Item(name='empty', value=0, begin=-1, end=0)
        result: List[Item] = [empty_item]

        for index in range(len(rewards)):
            for action in actions:
                #print("action: ", action)
                '''
                action:  'forward', 'left', 'back', 'right', 'jump', 'sneak', 'sprint', 'attack', 'camera', 'place', 'equip', 
                         'craft', 'nearbyCraft', 'nearbySmelt'
                '''
                
                if action not in excluded_actions:
                    #print("actions[action][index]: ", actions[action][index])
                    a = Action(name=action, value=actions[action][index])
                    last_item = result[-1]
                    #print("type(last_item): ", type(last_item))
                    #print("len(last_item): ", len(last_item))
                    #print("last_item['actions']: ", last_item['actions'])
                    #print("result[-1]: ", result[-1])
                    #time.sleep(1.0)
                    if not a.is_noop():
                        if last_item.get_last_action() != a:
                            #print("action: ", action)
                            #print("index: ", index)
                            #print("a: ", a)
                            #print("")
                            
                            last_item.add_action(a)
            #print("")
                     
            #print("result: ", result)
            #time.sleep(1.0)
            #print("len(last_item): ", len(last_item))
            #print("last_item: ", last_item)
            #print("")
                                     
            for item in items:
                '''
                items:  odict_keys(['coal', 'cobblestone', 'crafting_table', 'dirt', 'furnace', 'iron_axe', 
                                    'iron_ingot', 'iron_ore', 'iron_pickaxe', 'log', 'planks', 'stick', 'stone', 
                                    'stone_axe', 'stone_pickaxe', 'torch', 'wooden_axe', 'wooden_pickaxe'])
                '''
                
                if next_states[item][index] > states[item][index]:
                    #print("next_states[item][index]: ", next_states[item][index])
                    #print("states[item][index]: ", states[item][index])
                    
                    i = Item(item, next_states[item][index], begin=result[-1].end, end=index)
                    last_item = result[-1]
                    
                    #print("result: ", result)
                    #print("i.name: ", i.name)
                    ##print("last_item.name: ", last_item.name)
                    #print("")
                    #time.sleep(1.0)
                    
                    if i.name == last_item.name:
                        # update the required number of items
                        last_item.value = i.value
                        last_item.end = index
                    else:
                        pass
                        # add new item in chain
                        #print("i: ", i)
                        result.append(i)
                        
        result.append(empty_item)
        for item, next_item in zip(reversed(result[:-1]), reversed(result[1:])):
            item.actions, next_item.actions = next_item.actions, item.actions

        #print("len(result): ", len(result))
            
        # trying to remove bugs with putting and getting items on the crafting table and furnace
        to_remove = set()
        for index, item in enumerate(result):
            #print("index: ", index)
            #print("item: ", item)
            #print("")
            
            if item.begin == item.end:
                to_remove.add(index)
                if index - 1 >= 0:
                    to_remove.add(index - 1)

            if sum([1 for _ in result[:index + 1] if _.name == item.name]) >= item_appear_limit:
                to_remove.add(index)

        for index in reversed(sorted(list(to_remove))):
            if result[index].actions:
                # saving useful actions of wrong items
                result[index + 1].actions = (*result[index].actions, *result[index + 1].actions)

            result.pop(index)

        # remove empty items
        result = [item for item in result if item != empty_item]

        return result


def all_chains_info(envs, data_dir):
    chains = []

    def get_reward(trajectory_):
        return int(sum(trajectory_[2]))

    for env_name in envs:
        data = minerl.data.make(env_name, data_dir=data_dir)

        for index, trajectory_name in enumerate(sorted(data.get_trajectory_names())):
            print(trajectory_name)
            trajectory = TrajectoryDataPipeline.load_data_no_pov(
                os.path.join(data_dir, env_name, trajectory_name))
            # trajectory = load_data_without_pov(
            #     os.path.join(data_dir, env_name, trajectory_name))

            chain, time_indexes = TrajectoryInformation.compute_item_order(trajectory, return_time_indexes=True)
            chains.append(ChainInfo(chain=chain, reward=get_reward(trajectory), env_name=env_name,
                                    trajectory_name=trajectory_name, id_=index, length=len(trajectory[2]),
                                    time_indexes=time_indexes))

    return chains


def generate_best_chains(envs=("MineRLObtainIronPickaxe-v0",), data_dir="../data/"):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    chains = all_chains_info(envs=envs, data_dir=data_dir)
    filtered = [c for c in chains if c.reward == max([_.reward for _ in chains])]
    filtered = [c for c in sorted(filtered, key=lambda x: x.length)][:60]
    filtered = [c for c in sorted(filtered, key=lambda x: len(x.chain)) if 25 < len(c.chain) <= 31]
    filtered_chains = []
    for chain in filtered:
        filtered_chains.append(chain.chain)
        
    return filtered_chains


def generate_final_chain(envs=("MineRLObtainIronPickaxe-v0",), data_dir="../data/"):
    """
    generates final chain
    it may sampled randomly, but be careful short chains give poor results
    :param envs: number of envs
    :param data_dir:
    :return:
    """
    return generate_best_chains(envs=envs, data_dir=data_dir)[-1]


