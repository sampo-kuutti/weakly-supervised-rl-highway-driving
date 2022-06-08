# DDPG with Safety Cages
import numpy as np
import tensorflow as tf
import json, sys, os
from os import path
import random
from collections import deque
import ipg_proxy
import csv
import safety_val as sv

# hyperparameters
gamma = 0.99  # reward discount factor
h1_actor = 50  # hidden layer 1 size for the actor
h2_actor = 50  # hidden layer 2 size for the actor
h3_actor = 50  # hidden layer 3 size for the actor
h1_critic = 50  # hidden layer 1 size for the critic
h2_critic = 50  # hidden layer 2 size for the critic
h3_critic = 50  # hidden layer 3 size for the critic
lstm_actor = 16  # lstm units for actor
lr_actor = 1e-4  # learning rate for the actor
lr_critic = 1e-3  # learning rate for the critic
lr_decay = 1  # learning rate decay (per episode)
l2_reg_actor = 1e-6  # L2 regularization factor for the actor
l2_reg_critic = 1e-6  # L2 regularization factor for the critic
dropout_actor = 0  # dropout rate for actor (0 = no dropout)
dropout_critic = 0  # dropout rate for critic (0 = no dropout)
num_episodes = 5000  # number of episodes
max_steps_ep = 10000  # default max number of steps per episode (unless env has a lower hardcoded limit)
tau = 1e-3  # soft target update rate
train_every = 100  # number of steps to run the policy (and collect experience) before updating network weights
replay_memory_capacity = int(1e4)  # capacity of experience replay memory
minibatch_size = 64  # size of minibatch from experience replay memory for updates
initial_noise_scale = 1.0  # scale of the exploration noise process (1.0 is the range of each action dimension)
noise_decay = 0.997  # decay rate (per episode) of the scale of the exploration noise process
exploration_mu = 0.0  # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_theta = 0.15  # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_sigma = 0.2  # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt
restore_from = None
restore_ep = 0

# game parameters
#env = gym.make(env_to_use)
# Action Space Shape
N_S = 4  # number of states
N_A = 1  # number of actions
A_BOUND = [-1, 1]  # action bounds

np.random.seed(0)

outdir = '/vol/research/safeav/Sampo/condor-a2c/test/log_ddpg/ddpg_lstm_sc/'


def writefile(fname, s):
    with open(path.join(outdir, fname), 'w') as fh: fh.write(s)


info = {}
info['params'] = dict(
    gamma=gamma,
    h1_actor=h1_actor,
    h2_actor=h2_actor,
    h3_actor=h3_actor,
    lstm_actor=lstm_actor,
    h1_critic=h1_critic,
    h2_critic=h2_critic,
    h3_critic=h3_critic,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    lr_decay=lr_decay,
    l2_reg_actor=l2_reg_actor,
    l2_reg_critic=l2_reg_critic,
    dropout_actor=dropout_actor,
    dropout_critic=dropout_critic,
    num_episodes=num_episodes,
    max_steps_ep=max_steps_ep,
    tau=tau,
    train_every=train_every,
    replay_memory_capacity=replay_memory_capacity,
    minibatch_size=minibatch_size,
    initial_noise_scale=initial_noise_scale,
    noise_decay=noise_decay,
    exploration_mu=exploration_mu,
    exploration_theta=exploration_theta,
    exploration_sigma=exploration_sigma,
)

#np.set_printoptions(threshold=np.nan)

replay_memory = deque(maxlen=replay_memory_capacity)  # used for O(1) popleft() operation
# trauma memory
trauma_buffer = deque(maxlen=minibatch_size)


def add_to_memory(experience):
    replay_memory.append(experience)


def sample_from_memory(minibatch_size):
    return random.sample(replay_memory, minibatch_size)


def calculate_reward(th, delta_th, x_rel, sc):

    if 0 <= th < 0.50:                              # crash imminent
        reward = -10
    elif 0.50 <= th < 1.75 and delta_th <= 0:       # too close
        reward = -0.5
    elif 0.50 <= th < 1.75 and delta_th > 0:        # closing up
        reward = 0.1
    elif 1.75 <= th < 1.90:                         # goal range large
        reward = 0.5
    elif 1.90 <= th < 2.10:                         # goal range small
        reward = 5
    elif 2.10 <= th < 2.25:                         # goal range large
        reward = 0.5
    elif 2.25 <= th < 10 and delta_th <= 0:         # closing up
        reward = 0.1
    elif 2.25 <= th < 10 and delta_th > 0:          # too far
        reward = -0.1
    elif th >= 10 and delta_th <= 0:                # closing up
        reward = 0.05
    elif th >= 10 and delta_th > 0:                 # way too far
        reward = -10
    elif x_rel <= 0:
        reward = -100                               # crash occurred
    else:
        print('no reward statement requirements met (th = %f, delta_th = %f, x_rel = %f), reward = 0'
              % (th, delta_th, x_rel))
        reward = 0

    if sc > 0:
        reward += -1

    return reward


def calculate_reward2(th, delta_th, x_rel):

    if 0 <= th < 0.50:                              # crash imminent
        reward = -0.5
    elif 0.50 <= th < 1.75 and delta_th <= 0:       # too close
        reward = -0.1
    elif 0.50 <= th < 1.75 and delta_th > 0:        # closing up
        reward = 0.1
    elif 1.75 <= th < 1.90:                         # goal range large
        reward = 0.5
    elif 1.90 <= th < 2.10:                         # goal range small
        reward = 1
    elif 2.10 <= th < 2.25:                         # goal range large
        reward = 0.5
    elif 2.25 <= th < 10 and delta_th <= 0:         # closing up
        reward = 0.1
    elif 2.25 <= th < 10 and delta_th > 0:          # too far
        reward = -0.01
    elif th >= 10 and delta_th <= 0:                # closing up
        reward = 0.05
    elif th >= 10 and delta_th > 0:                 # way too far
        reward = -0.5
    elif x_rel <= 0:
        reward = -1                               # crash occurred
    else:
        print('no reward statement requirements met (th = %f, delta_th = %f, x_rel = %f), reward = 0'
              % (th, delta_th, x_rel))
        reward = 0

    return reward


class DdpgAgent(object):
    def __init__(self, sess):
        self.sess = sess

        # placeholders
        self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='state')
        self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_A], name='action')
        self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
        self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='next_state')
        self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='is_not_terminal')  # indicators (go into target computation)
        self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')  # for dropout

        # episode counter
        self.episodes = tf.Variable(float(restore_ep), trainable=False, name='episodes')
        self.episode_inc_op = self.episodes.assign_add(1)

        # actor network
        with tf.variable_scope('actor'):
            # Policy's outputted action for each state_ph (for generating actions and training the critic)
            self.actions = self.generate_actor_network(self.state_ph, trainable=True, reuse=False)

        # slow target actor network
        with tf.variable_scope('slow_target_actor', reuse=False):
            # Slow target policy's outputted action for each next_state_ph (for training the critic)
            # use stop_gradient to treat the output values as constant targets when doing backprop
            slow_target_next_actions = tf.stop_gradient(
                self.generate_actor_network(self.next_state_ph, trainable=False, reuse=False))

        with tf.variable_scope('critic') as scope:
            # Critic applied to state_ph and a given action (for training critic)
            q_values_of_given_actions = self.generate_critic_network(self.state_ph, self.action_ph, trainable=True, reuse=False)
            # Critic applied to state_ph and the current policy's outputted actions for state_ph (for training actor via deterministic policy gradient)
            q_values_of_suggested_actions = self.generate_critic_network(self.state_ph, self.actions, trainable=True, reuse=True)

        # slow target critic network
        with tf.variable_scope('slow_target_critic', reuse=False):
            # Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
            slow_q_values_next = tf.stop_gradient(
                self.generate_critic_network(self.next_state_ph, slow_target_next_actions, trainable=False, reuse=False))

        # isolate vars for each network
        actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
        slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
        critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
        slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

        # update values for slowly-changing targets towards current actor and critic
        update_slow_target_ops = []
        for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
            update_slow_target_actor_op = slow_target_actor_var.assign(
                tau * actor_vars[i] + (1 - tau) * slow_target_actor_var)
            update_slow_target_ops.append(update_slow_target_actor_op)

        for i, slow_target_var in enumerate(slow_target_critic_vars):
            update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
            update_slow_target_ops.append(update_slow_target_critic_op)

        self.update_slow_targets_op = tf.group(*update_slow_target_ops, name='update_slow_targets')

        # One step TD targets y_i for (s,a) from experience replay
        # = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
        # = r_i if s' terminal
        targets = tf.expand_dims(self.reward_ph, 1) + tf.expand_dims(self.is_not_terminal_ph, 1) * gamma * slow_q_values_next

        # 1-step temporal difference errors
        td_errors = targets - q_values_of_given_actions

        # critic loss function (mean-square value error with regularization)
        self.critic_loss = tf.reduce_mean(tf.square(td_errors))
        for var in critic_vars:
            if not 'bias' in var.name:
                self.critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

        # critic optimizer
        self.critic_train_op = tf.train.AdamOptimizer(lr_critic * lr_decay ** self.episodes).minimize(self.critic_loss)

        # actor loss function (mean Q-values under current policy with regularization)
        self.actor_loss = -1 * tf.reduce_mean(q_values_of_suggested_actions)
        for var in actor_vars:
            if not 'bias' in var.name:
                self.actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

        # actor optimizer
        # the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
        self.actor_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay ** self.episodes).minimize(self.actor_loss,
                                                                                          var_list=actor_vars)


    # will use this to initialize both the actor network its slowly-changing target network with same structure
    def generate_actor_network(self, s, trainable, reuse):
        hidden = tf.layers.dense(s, h1_actor, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
        hidden_drop = tf.layers.dropout(hidden, rate=dropout_actor, training=trainable & self.is_training_ph)
        hidden_2 = tf.layers.dense(hidden_drop, h2_actor, activation=tf.nn.relu, trainable=trainable, name='dense_1',
                                   reuse=reuse)
        hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout_actor, training=trainable & self.is_training_ph)
        hidden_3 = tf.layers.dense(hidden_drop_2, h3_actor, activation=tf.nn.relu, trainable=trainable, name='dense_2',
                                   reuse=reuse)
        hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout_actor, training=trainable & self.is_training_ph)

        # Recurrent network for temporal dependencies
        self.lstm_layer = tf.keras.layers.LSTM(lstm_actor, stateful=True, return_sequences=True)
        rnn_out = self.lstm_layer(tf.expand_dims(hidden_drop_3, [0]))
        rnn_out = tf.reshape(rnn_out, [-1, lstm_actor])

        actions_unscaled = tf.layers.dense(rnn_out, N_A, trainable=trainable, name='dense_3', reuse=reuse)
        actions = A_BOUND[0] + tf.nn.sigmoid(actions_unscaled) * (
                A_BOUND[1] - A_BOUND[0])  # bound the actions to the valid range
        return actions


    # will use this to initialize both the critic network its slowly-changing target network with same structure
    def generate_critic_network(self, s, a, trainable, reuse):
        state_action = tf.concat([s, a], axis=1)
        hidden = tf.layers.dense(state_action, h1_critic, activation=tf.nn.relu, trainable=trainable, name='dense',
                                 reuse=reuse)
        hidden_drop = tf.layers.dropout(hidden, rate=dropout_critic, training=trainable & self.is_training_ph)
        hidden_2 = tf.layers.dense(hidden_drop, h2_critic, activation=tf.nn.relu, trainable=trainable, name='dense_1',
                                   reuse=reuse)
        hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout_critic, training=trainable & self.is_training_ph)
        hidden_3 = tf.layers.dense(hidden_drop_2, h3_critic, activation=tf.nn.relu, trainable=trainable, name='dense_2',
                                   reuse=reuse)
        hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout_critic, training=trainable & self.is_training_ph)
        q_values = tf.layers.dense(hidden_drop_3, 1, trainable=trainable, name='dense_3', reuse=reuse)
        return q_values

    def reset_lstm(self):
        self.lstm_layer.reset_states()


tf.reset_default_graph()
# initialise noise process
noise_process = np.zeros(N_A)

# initialize session
# create agent and environment
graph = tf.Graph()
sess = tf.Session(graph=graph)

with sess.as_default():
    with graph.as_default():
        agent = DdpgAgent(sess)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        if restore_from is not None:
            saver.restore(sess, restore_from)


# Tensorboard summaries
tf.summary.scalar('loss/policy_loss', agent.actor_loss)
tf.summary.scalar('loss/value_loss', agent.critic_loss)
tf.summary.histogram('act_out', agent.actions)
#tf.summary.histogram('q_values', q_values_of_suggested_actions)
tf.summary.histogram('noise_process', noise_process)

with sess.as_default():
    with graph.as_default():
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(outdir, sess.graph)

# Define proxy environment
proxy = ipg_proxy.IpgProxy()

crash_count = 0  # count number of crashes in training run

total_steps = 0
arr_scen = []
# Run training episodes
for ep in range(restore_ep, num_episodes):
    # reset lstm
    with graph.as_default():
        agent.reset_lstm()

    total_reward = 0
    steps_in_ep = 0

    # clear traj buffers
    ER_buffer = []  # experience replay buffer
    trauma_buffer.clear()  # clear trauma buffer

    # Initialize exploration noise process
    noise_process = np.zeros(N_A)
    noise_scale = (initial_noise_scale * noise_decay ** ep) * (A_BOUND[1] - A_BOUND[0])

    # empty arrays
    arr_a = []  # acceleration array
    arr_j = []  # jerk array
    arr_t = []  # time array
    arr_x = []  # x_rel array
    arr_v = []  # velocity array
    arr_dv = []  # relative velocity array
    arr_th = []  # time headway array
    arr_y_0 = []  # original output
    arr_y_sc = []  # safety cage output
    arr_sc = []  # safety cage number
    arr_cof = []  # coefficient of friction

    arr_v_leader = []  # lead vehicle velocity
    arr_a_leader = []  # lead vehicle acceleration

    arr_rewards = []  # rewards list

    # lead vehicle states
    T_lead = []
    X_lead = []
    V_lead = []
    A_lead = []
    # read lead vehicle states from the corresponding traffic file (generated by traffic.py)
    with open('/vol/research/safeav/Sampo/condor-a2c/test/traffic_data/' + str(ep + 1) + '.csv') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            T_lead.append(float(row['t']))  # time
            X_lead.append(float(row['x']))  # long. position
            V_lead.append(float(row['v']))  # velocity
            A_lead.append(float(row['a']))  # acceleration

    # coefficient of friction
    scen = random.randint(1, 25)
    arr_scen.append(scen)

    cof = 0.375 + scen * 0.025  # calculate coefficient of friction

    # Run training using Ipg Proxy
    ep_r = 0  # set ep reward to 0
    ep_nr = 0  # set normalised ep reward to 0

    # set initial states
    t = 0
    v = 25.5  # 91.8 km/h
    a = 0
    x = 5

    # lead vehicle states
    t_iter = int(t // 0.02)  # current time step
    v_rel = V_lead[t_iter] - v  # relative velocity
    x_rel = X_lead[t_iter] - x  # relative distance
    if v != 0:  # check for division by 0
        t_h = x_rel / v
    else:
        t_h = x_rel

    observation = [v_rel, t_h, v, a]  # define input array
    crash = 0  # variable for checking if a crash has occurred (0=no crash, 1=crash)
    prev_output = 0

    # loop time-steps
    while t < 300 and crash == 0:
        action_for_state = sess.run(agent.actions,
                                     feed_dict={agent.state_ph: np.reshape(observation, (1, N_S)),
                                                agent.is_training_ph: False
                                                })

        # add temporally-correlated exploration noise to action (using an Ornstein-Uhlenbeck process)
        # calculate noise process
        noise_process = exploration_theta * (exploration_mu - noise_process) + exploration_sigma * np.random.randn(
            N_A)
        # add noise and clip action
        action_for_state += noise_scale * noise_process
        action_for_state = np.clip(action_for_state, A_BOUND[0], A_BOUND[1])  # ensure action is within specified action limits

        # safety cage stuff
        sc, output = sv.safety_cage2(t, v_rel, t_h, v, x_rel, a, action_for_state, 0)
        arr_y_0.append(float(action_for_state))
        arr_y_sc.append(float(output))
        arr_sc.append(sc)

        # read new states
        # host states
        t_ = t + 0.04  # time
        proxy_out = proxy.inference([v, a, cof, output, prev_output])
        v_ = float(proxy_out)
        delta_v = v_ - v
        if delta_v > 0.4:  # limit a to +/- 10m/s^2
            delta_v = 0.4
            v_ = delta_v + v  # clip new v to max. delta v
        elif delta_v < -0.4:
            delta_v = -0.4
            v_ = delta_v + v  # clip new v to max. delta v
        if v_ < 0:  # check for negative velocity
            v_ = 0
        a_ = delta_v / 0.04
        x_ = x + (v * 0.04)

        # lead vehicle states
        t_iter_ = int(t_ // 0.02)  # current time step nb: lead vehicle states in traffic_data iterate @ 20ms
        v_rel_ = V_lead[t_iter_] - v_  # relative velocity
        x_rel_ = X_lead[t_iter_] - x_  # relative distance

        # enter variables into arrays
        arr_a.append(a)
        arr_t.append(t)
        arr_x.append(x_rel)
        arr_v.append(v)
        arr_dv.append(v_rel)
        arr_th.append(t_h)
        arr_cof.append(cof)

        arr_v_leader.append(V_lead[t_iter])
        arr_a_leader.append(A_lead[t_iter])

        # calculate time headway
        if v_ != 0:
            t_h_ = x_rel_ / v_
        else:
            t_h_ = x_rel_

        next_observation = [v_rel_, t_h_, v_, a_]  # new observation

        # calculate reward
        if (t_ - t) != 0:
            delta_th = (t_h_ - t_h) / (t_ - t)
        else:
            delta_th = 0

        reward = calculate_reward2(t_h_, delta_th, x_rel_)

        ep_r += reward
        arr_rewards.append(reward)

        if t_ >= 300 or crash == 1:  # is ep done
            done = True
        else:
            done = False

        total_reward += reward

        # append buffers and replay memory
        # add to trauma memory buffer
        trauma_buffer.append((observation, output, reward, next_observation, 0.0 if done else 1.0))
        # check if crash occurs stop simulation if a crash occured
        if x_rel_ <= 0:
            crash = 1
            crash_count += 1
            print('crash occurred: simulation run stopped')
            if len(trauma_buffer) >= minibatch_size:
                add_to_memory(trauma_buffer)

        ER_buffer.append((observation, output, reward, next_observation, 0.0 if done else 1.0))

        # if buffer > mb_size add to experience replay and empty buffer
        if len(ER_buffer) >= minibatch_size:
            add_to_memory(ER_buffer)
            ER_buffer = []

        # update network weights to fit a minibatch of experience
        if total_steps % train_every == 0 and len(replay_memory) >= 1:
            # grab N (s,a,r,s') tuples from replay memory
            minibatch = sample_from_memory(1)[-1]

            # reset lstm cell state
            with graph.as_default():
                agent.reset_lstm()

            # update the critic and actor params using mean-square value error and deterministic policy gradient, respectively
            _, _ = sess.run([agent.critic_train_op, agent.actor_train_op],
                            feed_dict={
                                agent.state_ph: np.asarray([elem[0] for elem in minibatch]),
                                agent.action_ph: np.reshape(
                                    np.asarray([elem[1] for elem in minibatch]), (minibatch_size, N_A)),
                                agent.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                                agent.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                                agent.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
                                agent.is_training_ph: True
                                })

            # update slow actor and critic targets towards current actor and critic
            _ = sess.run(agent.update_slow_targets_op)

            # reset lstm again
            with graph.as_default():
                agent.reset_lstm()


        # update state variables
        observation = next_observation
        t = t_
        v = v_
        a = a_
        x = x_
        v_rel = v_rel_
        x_rel = x_rel_
        t_h = t_h_
        t_iter = t_iter_
        prev_output = output
        total_steps += 1
        steps_in_ep += 1

        if done:
            # Increment episode counter
            _ = sess.run(agent.episode_inc_op)
            break

    print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' % (
        ep, total_reward, steps_in_ep, noise_scale))

    summary = sess.run(merged, feed_dict={
        agent.state_ph: np.asarray([elem[0] for elem in minibatch]),
        agent.action_ph: np.reshape(
            np.asarray([elem[1] for elem in minibatch]), (minibatch_size, N_A)),
        agent.reward_ph: np.asarray([elem[2] for elem in minibatch]),
        agent.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
        agent.is_not_terminal_ph: np.asarray([elem[4] for elem in minibatch]),
        agent.is_training_ph: False})
    writer.add_summary(summary, ep)
    writer.flush()
    perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Reward', simple_value=float(total_reward))])
    writer.add_summary(perf_summary, ep)
    writer.flush()
    perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Mean_Reward', simple_value=float(np.mean(arr_rewards)))])
    writer.add_summary(perf_summary, ep)
    writer.flush()
    perf_summary = tf.Summary(value=[tf.Summary.Value(tag='Perf/Mean_Th', simple_value=float(np.mean(arr_th)))])
    writer.add_summary(perf_summary, ep)
    writer.flush()
    perf_summary = tf.Summary(value=[tf.Summary.Value(tag='noise_scale', simple_value=float(noise_scale))])
    writer.add_summary(perf_summary, ep)
    writer.flush()

    # store eps with crashes
    if crash == 1:
        try:
            if not os.path.exists(outdir + '/results'):
                os.makedirs(outdir + '/results')
            # calculate jerk array
            for k in range(0, 5):
                arr_j.append(float(0))

            for k in range(5, len(arr_t)):
                # calculate vehicle jerk
                if abs(arr_t[k] - arr_t[k - 5]) != 0:
                    arr_j.append(((arr_a[k]) - (arr_a[k - 5])) / (arr_t[k] - arr_t[k - 5]))  # jerk
                else:
                    arr_j.append(0)

            # write results to file
            headers = ['t', 'j', 'v', 'a', 'v_lead', 'a_lead', 'x_rel', 'v_rel', 'th', 'y_0', 'y_sc', 'sc', 'cof']
            with open(outdir + '/results/' + str(ep) + '.csv', 'w', newline='\n') as f:
                wr = csv.writer(f, delimiter=',')
                rows = zip(arr_t, arr_j, arr_v, arr_a, arr_v_leader, arr_a_leader, arr_x, arr_dv, arr_th,
                           arr_y_0,
                           arr_y_sc, arr_sc, arr_cof)
                wr.writerow(headers)
                wr.writerows(rows)

        except FileNotFoundError as e:
            # print error
            print('Error Occurred!' , e)


# Save results
print('Total no. of crashes = %d' % crash_count)
writefile('info.json', json.dumps(info))
#env.close()
#gym.upload(outdir)
if not os.path.exists(outdir):
    os.makedirs(outdir)
checkpoint_path = os.path.join(outdir, "model-ep-%d-finalr-%d.ckpt" % (ep, total_reward))
filename = saver.save(sess, checkpoint_path)
print("Model saved in file: %s" % filename)

