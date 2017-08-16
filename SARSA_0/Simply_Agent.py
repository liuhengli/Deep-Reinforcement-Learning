"""
Using SARSA(0) Algorithm implementation a simply Agent
"""

from Gridworld import *
from random import random  # 随机策略
import gym
from gym import Env


class Agent(object):
    def __init__(self, env=Env):
        self.env = env  # 个体持有环境的引用
        self.Q = {}  # 个体维护一张行为价值表Q
        # self.state = None  # 个体当前的观测
        self.reset_Agent()

    def reset_Agent(self):
        self.state = self.env.reset()
        state_name = self._get_state_name(self.state)
        self._assert_state_in_Q(state_name, randomized=False)

    # 执行一次策略
    """
    为了能够使得个体随着训练次数的增多而减少产生不确定行为的几率epsilon-greedy，进而收敛至
    最优策略，可以将其就改为衰减的epsilon-greedy ，这里采用的办法是将当前训练的Episode次
    数作为参数传递给策略函数
    我们为执行策略方法增加了一个use_epsilon参数，使得我们可以随时切换是否使用 epsilon 。
    通过这样设置，今后可以很容易将SARSA算法修改为Q学习算法。最后我们来实现SARSA算法的核心。
    """

    def Preform_Policy(self, state, episode_num, use_epsilon):
        epsilon = 1.0 / (episode_num + 1)
        print("===", state)
        Q_s = self.Q[state]
        action_name = 'unknow'
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:
            action = self.env.action_space.sample()
        else:
            action_name = max(Q_s, key=Q_s.get)
            action = int(action_name)
        return action

    # 执行一个动作
    def Action_Step(self, action):
        return self.env.step(action)

    #learning方法是SARSA的核心
    def Learning(self, gamma, alpha, max_episode_num):
        total_steps = 0
        # 设置训练次数
        for num_episode in range(max_episode_num):
            # 初始化环境
            self.state = self.env.reset()
            # 获取个体对于观测的命名
            state_prior_name = self._get_state_name(self.state)
            # 界面显示
            self.env.render()
            action_prior = self.Preform_Policy(
                state_prior_name, num_episode, use_epsilon=True)
            step_in_epidode = 0
            is_done = False
            # 针对一个Episode
            while not is_done:
                # 执行行为
                state_next, reward, is_done, info = self.Action_Step(
                    action_prior)
                # 更新界面
                self.env.render()
                # 获取新的状态名S’
                state_next_name = self._get_state_name(state_next)
                self._assert_state_in_Q(state_next_name, randomized=True)
                # 得到A‘
                action_next = self.Preform_Policy(
                    state_next_name, num_episode, use_epsilon=True)
                old_Q = self._get_Q(state_prior_name, action_prior)
                Q_prime = self._get_Q(state_next_name, action_next)
                # TD算法
                TD_target = reward + gamma * Q_prime
                # 更新Q值
                new_Q = old_Q + alpha * (TD_target - old_Q)
                self._set_Q(state_prior_name, action_prior, new_Q)

                # 显示最后一个Episode的信息
                if num_episode == max_episode_num:
                    print(
                        "step: {0:>2}: state: {1}, action: {2: 2}, state_next:{3}".
                        format(step_in_epidode, state_prior_name, action_prior,
                               state_next_name))

                # 开始下一step
                action_prior = action_next
                state_prior_name = state_next_name
                step_in_epidode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, step_in_epidode))
            total_steps += step_in_epidode

    def _get_state_name(self, state_name):
        return str(state_name)  # 仅适用离散观测空间的环境。

    # 判断s的初值是否存在
    def _is_state_in_Q(self, state):
        return self.Q.get(state) is not None

    # 初始化某状态的Q值
    def _init_state_value(self, s_name, randomized=True):
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            # 针对所有可能行为
            for action in range(self.env.action_space.n):
                default_value = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_value

    # 确保某状态的Q值存在
    def _assert_state_in_Q(self, state_name, randomized=True):
        if not self._is_state_in_Q(state_name):
            self._init_state_value(state_name, randomized)

    # 获取动作行为价值Q(s, a)
    def _get_Q(self, state, action):
        self._assert_state_in_Q(state, randomized=True)
        return self.Q[state][action]

    # 设置Q(s, a)
    def _set_Q(self, state, action, value):
        self._assert_state_in_Q(state, randomized=True)
        self.Q[state][action] = value


def main():
    env = SimpleGridWorld()
    agent = Agent(env)
    agent.Learning(gamma=0.9, alpha=0.1, max_episode_num=800)


if __name__ == '__main__':
    main()
