import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import copy
import seaborn as sns
import os
from datetime import datetime
from numpy import sum, mean, size, sqrt
from scipy.stats import invgamma


def draw_mus_and_sigmas(data):
    # number of samples
    N = size(data)
    # find the mean of the data
    the_mean = mean(data)
    # sum of squared differences between data and mean
    SSD = sum((data - the_mean) ** 2)

    # combining the prior with the data - page 79 of Gelman et al.
    # to make sense of this note that
    # inv-chi-sq(v,s^2) = inv-gamma(v/2,(v*s^2)/2)
    kN = float(k0 + N)
    mN = (k0 / kN) * m0 + (N / kN) * the_mean
    vN = v0 + N
    vN_times_s_sqN = v0 * s_sq0 + SSD + (N * k0 * (m0 - the_mean) ** 2) / kN

    # 1) draw the variances from an inverse gamma
    # (params: alpha, beta)
    alpha = vN / 2
    beta = vN_times_s_sqN / 2
    # thanks to wikipedia, we know that:
    # if X ~ inv-gamma(a,1) then b*X ~ inv-gamma(a,b)
    sig_sq_samples = beta * invgamma.rvs(alpha)

    # 2) draw means from a normal conditioned on the drawn sigmas
    # (params: mean_norm, var_norm)
    mean_norm = mN
    var_norm = sqrt(sig_sq_samples / kN)
    mu_samples = np.random.normal(mean_norm, var_norm)

    # 3) return the mu_samples and std_devs
    return mu_samples, sqrt(sig_sq_samples)


class Item:
    def __init__(self):
        self.tasks = []
        for i in range(nr_buckets):
            self.tasks.append(np.random.randint(0, process_types, size=bucket_size).tolist())
        self.value = np.random.uniform()
        self.enqueued = False

    def process(self, process_type):
        if tasks_ordered and self.tasks:
            if process_type in self.tasks[0]:
                self.tasks[0].remove(process_type)
            if not self.tasks[0]:
                self.tasks.pop(0)
        else:
            if process_type in self.tasks:
                self.tasks.remove(process_type)
        self.enqueued = False

    def enqueue(self, machine):
        if self.enqueued:
            return
        self.enqueued = True
        machine.items.append(self)


class Machine:
    def __init__(self, process_type=None):
        if process_type is not None:
            self.process_type = process_type
        else:
            self.process_type = np.random.randint(0, process_types)
        self.process_cost = np.random.uniform()
        self.failure_p = np.random.uniform(0, 0.2)
        self.items = []
        self.total_cost = 0

    def process(self):
        if noise and np.random.uniform() < self.failure_p:
            return
        if self.items:
            self.items[0].process(self.process_type)
            self.items.pop(0)
            self.total_cost += self.process_cost


class Factory:
    def __init__(self):
        self.machines = []
        for i in range(0, nr_machines):
            if i < process_types:
                self.machines.append(Machine(i))
            else:
                self.machines.append(Machine())
        self.items = []
        self.produced_value = 0
        self.time = 0

    def init_items(self):
        self.items = [Item() for _ in range(0, nr_items)]

    def update(self, action):
        self.time += 1
        score_pre = self.get_score()
        for i, a in zip(self.items, action):
            if 0 <= a < len(self.machines):
                i.enqueue(self.machines[a])
        for m in self.machines:
            m.process()
        for i in self.items:
            if not i.tasks:
                self.produced_value += i.value
                self.items.remove(i)
        return self.get_score() - score_pre

    def report(self):
        print('time', self.time)
        for i in self.items:
            print(i.__dict__)
        for m in self.machines:
            print(m.__dict__)
        print(self.get_score())

    def get_score(self):
        score = - self.get_process_requests()
        if use_cost:
            score -= self.get_total_cost()
        if use_value:
            score += self.produced_value
        return score

    def get_process_requests(self):
        requests = 0
        for i in self.items:
            for bucket in i.tasks:
                requests += len(bucket)
        return requests

    def get_total_cost(self):
        return sum([m.total_cost for m in self.machines])


class Planner:
    @staticmethod
    def random(factory):
        return np.random.choice(range(0, len(factory.machines)), size=len(factory.items) + 1).tolist()

    @staticmethod
    def vanilla_mc(factory):
        best_action = None
        best_score = float('-inf')
        for _ in range(0, nr_planning_samples):
            virtual_factory = copy.deepcopy(factory)
            action = Planner.random(virtual_factory)
            reward = virtual_factory.update(action)
            if reward > best_score:
                best_score = reward
                best_action = action
        return best_action

    @staticmethod
    def seq_vmc(factory):
        best_action = None
        best_score = float('-inf')
        for _ in range(0, int(nr_planning_samples / seq_steps)):
            virtual_factory = copy.deepcopy(factory)
            actions = []
            reward = 0
            for _ in range(0, seq_steps):
                action = Planner.random(virtual_factory)
                actions.append(action)
                reward += virtual_factory.update(action)

            if reward > best_score:
                best_score = reward
                best_action = actions[0]
        return best_action

    @staticmethod
    def bandit(factory, sample_method):
        bandits = [Bandit(factory) for i in factory.items]

        for _ in range(0, nr_planning_samples):
            virtual_factory = copy.deepcopy(factory)
            action = [sample_method(b) for b in bandits]
            reward = virtual_factory.update(action)
            for b in bandits:
                b.update(reward)

        action = [b.play() for b in bandits]
        return action

    @staticmethod
    def thompson(factory):
        return Planner.bandit(factory, Bandit.sample)

    @staticmethod
    def e_greedy(factory):
        return Planner.bandit(factory, Bandit.sample_epsilon_greedy)

    @staticmethod
    def seq_bandit(factory, sample_method):
        bandits = []
        for step in range(0, seq_steps):
            bandits.append([Bandit(factory) for i in factory.items])

        for _ in range(0, int(nr_planning_samples / seq_steps)):
            virtual_factory = copy.deepcopy(factory)
            rewards = []
            actions = []
            drop_mask = np.random.choice([-1, 0], p=[drop_rate, 1. - drop_rate], size=len(bandits[0]))
            for step in range(0, seq_steps):
                action = [sample_method(b) for b in bandits[step]]
                for i, a in enumerate(action):
                    if drop_mask[i] == -1:
                        action[i] = -1
                actions.append(action)
                rewards.append(virtual_factory.update(action))

            rewards = np.cumsum(rewards[::-1])[::-1]
            for step in range(0, seq_steps):
                for b, a in zip(bandits[step], actions[step]):
                    if a >= 0:
                        b.update(rewards[step])

        action = [b.play() for b in bandits[0]]
        return action

    @staticmethod
    def seq_thompson(factory):
        return Planner.seq_bandit(factory, Bandit.sample)

    @staticmethod
    def seq_e_greedy(factory):
        return Planner.seq_bandit(factory, Bandit.sample_epsilon_greedy)

    @staticmethod
    def seq_ucb(factory):
        return Planner.seq_bandit(factory, Bandit.sample_ucb)


class Bandit:
    def __init__(self, factory):
        # additional arm for noop
        self.arms = [Arm() for _ in range(len(factory.machines) + 1)]
        self.play_index = None

    def sample(self):
        values = [arm.sample() for arm in self.arms]
        self.play_index = np.argmax(values)
        return self.play_index

    def sample_epsilon_greedy(self):
        if np.random.uniform() <= epsilon:
            self.play_index = np.random.randint(0, len(self.arms))
        else:
            values = [np.mean(arm.data[-bandit_buffer_size:]) for arm in self.arms]
            self.play_index = np.argmax(values)
        return self.play_index

    def sample_ucb(self):
        log_n = np.log(np.sum([len(arm.data[-bandit_buffer_size:]) for arm in self.arms]))
        values = [np.mean(arm.data[-bandit_buffer_size:]) + np.sqrt(log_n / len(arm.data[-bandit_buffer_size:])) for arm
                  in self.arms]
        self.play_index = np.argmax(values)
        return self.play_index

    def play(self):
        values = [np.mean(arm.data[-bandit_buffer_size:]) for arm in self.arms]
        self.play_index = np.argmax(values)
        return self.play_index

    def update(self, value):
        self.arms[self.play_index].data.append(value)
        self.play_index = None


class Arm:
    def __init__(self):
        self.data = []

    def sample(self):
        mu, std = draw_mus_and_sigmas(self.data[-bandit_buffer_size:])
        return mu


def run_experiment(method, label, factory):
    print('running', label)
    for i in range(0, 40):
        print('step', i, end='\r')
        action = method(factory)
        factory.update(action)
        log.loc[len(log.index)] = {'time': factory.time, 'score': factory.get_score(),
                                   'requests': factory.get_process_requests(), 'cost': factory.get_total_cost(),
                                   'value': factory.produced_value, 'planner': label, 'nr_samples': nr_planning_samples,
                                   'drop_rate': drop_rate, 'experiment': experiment}


np.random.seed(111222333)

directory = "plots/bayes/{}".format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
if not os.path.exists(directory):
    os.makedirs(directory)

process_types = 3
nr_buckets = 2
bucket_size = 2
nr_machines = 4
nr_planning_samples = 256
seq_steps = 4
bandit_buffer_size = 10
epsilon = 0.1
nr_items = 8
noise = True
tasks_ordered = True
use_cost = True
use_value = False
drop_rate = 0.5
# prior hyperparameters
m0 = 0.
k0 = 1.
s_sq0 = 100.
v0 = 1.

log = pd.DataFrame(
    columns=['time', 'score', 'requests', 'cost', 'value', 'planner', 'nr_samples', 'drop_rate', 'experiment'])
print(log.index)
sns.set_style("white")

for i in range(0, 100):
    experiment = i
    print('running experiment', i)
    factory = Factory()
    factory.init_items()
    for fig_i, nrps in enumerate([64, 128, 256, 512]):
        nr_planning_samples = nrps
        for dr in [0., 0.25, 0.5, 0.75]:
            drop_rate = dr
            comm_rate = 1 - drop_rate
            run_experiment(Planner.seq_thompson, 'thompson {}'.format(comm_rate), copy.deepcopy(factory))
            run_experiment(Planner.seq_e_greedy, 'epsilon {}'.format(comm_rate), copy.deepcopy(factory))
            run_experiment(Planner.seq_ucb, 'ucb {}'.format(comm_rate), copy.deepcopy(factory))
            run_experiment(Planner.vanilla_mc, 'vmc {}'.format(comm_rate), copy.deepcopy(factory))

            if i > 0:
                exp = []
                for e in range(len(log)):
                    exp.append(int(e / (40 * 16)))
                log['experiment'] = pd.Series(exp)
                log.to_csv('{}/data.csv'.format(directory), sep='\t', encoding='utf-8')

                plt.clf()
                filtered_log = log
                filtered_log['time'] = filtered_log['time'].astype(int)
                filtered_log['nr_samples'] = filtered_log['nr_samples'].astype(int)
                filtered_log = filtered_log[filtered_log['nr_samples'] == nr_planning_samples]
                filtered_log = filtered_log[filtered_log['drop_rate'] == drop_rate]
                sns.tsplot(data=filtered_log, time='time', condition='planner', value='score', unit='experiment')
                sns.despine(trim=True)
                plt.savefig('{}/reward_{}_{}.png'.format(directory, nr_planning_samples, drop_rate),
                            bbox_inches='tight')
