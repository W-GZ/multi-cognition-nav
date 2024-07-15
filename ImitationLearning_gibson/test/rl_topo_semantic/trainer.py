import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from collections import namedtuple, OrderedDict
import logging

# Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
Transition = namedtuple('Transition', ['action', 'a_log_prob', 'reward', 'detect_features',
                                       'current_visual_feature', 'target_visual_feature', 'locate_vector',
                                       # 'number_of_nodes', 'all_edges'
                                       ])


# Transition_graph = namedtuple('Transition', ['topological_graph'])


class Trainer(object):
    def __init__(self, action_space,
                 model,
                 use_greedy=False,
                 device='cpu',
                 clip_param=0.2,
                 max_grad_norm=0.4,
                 ppo_update_iters=5,
                 batch_size=128,
                 gamma=0.99,
                 lr_decay_step=5e6,
                 epsilon_greedy_decay_step=5e6,
                 initial_lr=1e-6,
                 initial_epsilon_greedy=0.6  # 0.5
                 ):
        super(Trainer, self).__init__()

        self.initial_lr = initial_lr
        self.initial_epsilon_greedy = initial_epsilon_greedy
        self.clip_param = clip_param
        self.max_grad_norm = max_grad_norm
        self.ppo_update_iters = ppo_update_iters
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr_decay_step = lr_decay_step
        self.epsilon_greedy_decay_step = epsilon_greedy_decay_step

        self.device = device
        self.use_greedy = use_greedy
        self.model = model
        self.action_space = action_space
        self.optimizer = optim.Adam(self.model.parameters(), self.initial_lr)

        self.criterion = nn.CrossEntropyLoss()

        # Training stats
        self.global_step = 0
        self.buffer = []
        # self.buffer_graph = []
        self.stats = {'scene_index': [],
                      'cumulative_reward': [],
                      'episode_length': [],
                      # 'search_rate': [],
                      # 're_search_rate': [],
                      'target_dis': [],
                      'value_loss': [],
                      'policy_loss': [],
                      'learning_rate': [],
                      'success_rate': []
                      }

        self.init_flag = True

        self.MSELoss = nn.MSELoss()

    def _adjust_learning_rate(self):
        if self.lr_decay_step > 0:
            learning_rate = 0.9 * self.initial_lr * (
                    self.lr_decay_step - self.global_step) / self.lr_decay_step + 0.1 * self.initial_lr
            if self.global_step > self.lr_decay_step:
                learning_rate = 0.1 * self.initial_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            learning_rate = self.initial_lr
        self.stats['learning_rate'].append(learning_rate)

    def work(self, current_bgr, target_bgr, type_="selectAction", current_position_yaw=None, target_position_yaw=None, scene_id=None):
        """
        Forward pass of the PPO agent. Depending on the type_ argument, it either explores by sampling its actor's
        PPO代理的正向传递。根据类型参数，它要么通过采样参与者的softmax输出进行探索，要么通过选择具有最大概率（argmax）的动作来消除探索。 
        softmax output, or eliminates exploring by selecting the action with the maximum probability (argmax).
        """
        # rgb: up-->tensor
        # observation = from_numpy(np.array(observation)).float().unsqueeze(0).to(self.device)
        current_bgr = np.expand_dims(current_bgr, axis=0)  # 1*144*192*3, 0-1
        target_bgr = np.expand_dims(target_bgr, axis=0)  # 1*144*192*3, 0-1

        # print(current_bgr.shape)
        # print(target_bgr.shape)

        with no_grad():
            # action_prob, detect_features, current_visual_feature, target_visual_feature, locate_vector\
            #     = self.model(current_bgr=current_bgr, target_bgr=target_bgr, position=position, name='gathering')
            action_prob, current_obj_predict, dir_predict = self.model(current_bgr=current_bgr, target_bgr=target_bgr,
                                                                       current_position_yaw=current_position_yaw,
                                                                       target_position_yaw=target_position_yaw,
                                                                       scene_id=scene_id,
                                                                       name='gathering')

        # print(action_prob)

        # adjust epsilon_greedy
        epsilon_greedy = self.initial_epsilon_greedy * (
                self.epsilon_greedy_decay_step - self.global_step) / self.epsilon_greedy_decay_step
        if self.global_step > self.epsilon_greedy_decay_step:
            epsilon_greedy = 0

        # select action
        if type_ == "selectAction":
            if self.use_greedy and np.random.rand() < epsilon_greedy:  # 小于epsilon_greedy，随机动作
                return np.random.choice(range(self.action_space))
                # selectedAction:0、1、2, actionProb
            else:  # 大于，听网络的
                c = Categorical(action_prob)
                action = c.sample()
                return action.item()
        elif type_ == "selectActionMax":
            return np.argmax(action_prob.cpu()).item()

    def save(self, path, episode):
        """
        Save actor and critic models in the path provided.

        :param path: path to save the models
        :type path: str
        :param episode:
        """
        # save(self.model.state_dict(), path + str(self.global_step) + '.pkl')
        save(self.model.state_dict(), path + str(episode) + '_' + str(self.global_step) + '.pkl')

    def load(self, path):
        """
        Load actor and critic models from the path provided.

        :param path: path where the models are saved
        :type path: str
        """

        model_state_dict = load(path)
        self.model.load_state_dict(model_state_dict, strict=False)

    def writeSummary(self, writer, episodeCount):
        """
        Write training metrics and data into tensorboard.将训练指标和数据写入张量板

        :param writer: pre-defined summary writer
        :type writer: TensorBoard summary writer
        :param episodeCount:
        """
        for key in self.stats:
            if len(self.stats[key]) > 0:
                stat_mean = float(np.mean(self.stats[key]))
                # writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=self.global_step)
                writer.add_scalar(tag='Info/{}'.format(key), scalar_value=stat_mean, global_step=episodeCount)
                self.stats[key] = []
        writer.flush()

    def storeTransition(self, transition):
        """
        Stores a transition in the buffer to be used later.将转换存储在缓冲区中以供以后使用。

        :param transition: contains state, action, action_prob, reward, next_state
        :type transition: namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])
        """
        self.buffer.append(transition)

    def store_graph(self, transition_graph):
        self.buffer_graph.append(transition_graph)

    def trainStep(self, topological_graph=None):
        """
        Performs a training step for the actor and critic models, based on transitions gathered in the
        buffer. It then resets the buffer.  根据缓冲区进行training，然后重置缓冲区

        :param batchSize: Overrides agent set batch size, defaults to None
        :type batchSize: int, optional
        """
        # # Default behaviour waits for buffer to collect at least one batch_size of transitions
        # if batchSize is None:
        #     if len(self.buffer) < self.batch_size:
        #         return
        #     batchSize = self.batch_size

        # Transition = namedtuple('Transition', ['action', 'a_log_prob', 'reward',
        #                                        'detect_features', 'number_of_nodes', 'all_edges'])
        # Transition_graph = namedtuple('Transition', ['topological_graph'])

        # Extract actions, rewards and action probabilities from transitions in buffer
        reward = [t.reward for t in self.buffer]  # buffer_size
        action = tensor([t.action for t in self.buffer], dtype=torch_long).view(-1, 1)  # buffer_size*1
        old_action_log_prob = tensor([t.a_log_prob for t in self.buffer], dtype=torch_float).view(-1,
                                                                                                  1)  # buffer_size*1
        detect_features = tensor([t.detect_features for t in self.buffer], dtype=torch_float)  # b*42*5
        current_visual_feature = tensor([t.current_visual_feature.cpu().detach().numpy() for t in self.buffer],
                                        dtype=torch_float)  # b*1*1000
        target_visual_feature = tensor([t.target_visual_feature.cpu().detach().numpy() for t in self.buffer],
                                       dtype=torch_float)  # b*1*1000
        locate_vector = tensor([t.locate_vector.cpu().detach().numpy() for t in self.buffer],
                               dtype=torch_float)  # b*N*2

        # number_of_nodes = tensor([t.number_of_nodes for t in self.buffer], dtype=torch_float)  # b*1
        # # all_edges = tensor([t.all_edges for t in self.buffer], dtype=torch.int64)  # b*([], [])
        # all_edges = [t.all_edges for t in self.buffer]  # b*([], [])

        # print(detect_features.shape)
        # print(current_visual_feature.shape)
        # print(number_of_nodes.shape)
        # print(all_edges.shape)

        # learning rate decay
        self._adjust_learning_rate()

        # Unroll rewards
        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = tensor(Gt, dtype=torch_float).to(self.device)

        total_v, total_p = 0, 0
        # Repeat the update procedure for ppo_update_iters
        for i in range(self.ppo_update_iters):
            # Create randomly ordered batches of size batchSize from buffer
            print(i)
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), 1, False):
                # Calculate the advantage at each step
                Gt_index = Gt[index].view(-1, 1)

                V, action_prob = self.model(obj_detect_results=detect_features[index],
                                            current_visual_feature_=current_visual_feature[index],
                                            target_visual_feature_=target_visual_feature[index],
                                            locate_vector_=locate_vector[index],
                                            # number_of_nodes=number_of_nodes[index],
                                            # all_edges=all_edges[index[0]],
                                            # topological_graph=topological_graph,
                                            name='training')
                delta = Gt_index - V
                advantage = delta.detach()

                # Get the current probabilities
                # Apply past actions with .gather()
                action_prob = action_prob.gather(1, action[index].cuda())  # new policy

                # PPO
                ratio = (action_prob / old_action_log_prob[
                    index].cuda())  # Ratio between current and old policy probabilities
                surr1 = ratio * advantage
                surr2 = clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update main network
                action_loss = -torch_min(surr1, surr2).mean()  # MAX->MIN descent
                value_loss = F.mse_loss(Gt_index, V)

                # if np.isnan(action_loss.item()):
                #     print('action_loss:', action_loss)
                #     print('value_loss:', value_loss)
                loss = action_loss + value_loss  # + position_loss + theta_loss

                self.optimizer.zero_grad()  # Delete old gradients

                loss.backward(retain_graph=True)  # Perform backward step to compute new gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)  # Clip gradients
                self.optimizer.step()  # Perform training step based on gradients

                total_v += value_loss
                total_p += action_loss

        self.stats['value_loss'].append(total_v.cpu().item())
        self.stats['policy_loss'].append(total_p.cpu().item())
        # After each training step, the buffer is cleared
        self.buffer = []
        # self.buffer_graph = []
