# runs ipg testing with recurrent DDPG model, can loop testing over multiple model files in DIRS and MODELS list

import numpy as np
import os
import tensorflow as tf
import ctypes
import csv
import random
import ipg_proxy
from collections import deque
import time
import datetime
import argparse
import pythonapi

timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')
fpath = 'S:/Research/safeav/Sampo/condor-a2c/test/log_ddpg/'  # use project directory path

# Models to test, testing loops over each model in DIRS
DIRS = []  # directory of mode
MODELS = []  # model filename

# PARAMETERS
gamma = 0.99  # reward discount factor
h1_actor = 50  # hidden layer 1 size for the actor
h2_actor = 50  # hidden layer 2 size for the actor
h3_actor = 50  # hidden layer 3 size for the actor
lstm_actor = 16
h1_critic = 50  # hidden layer 1 size for the critic
h2_critic = 50  # hidden layer 2 size for the critic
h3_critic = 50  # hidden layer 3 size for the critic
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
replay_memory_capacity = int(1e6)  # capacity of experience replay memory
minibatch_size = 64  # size of minibatch from experience replay memory for updates
initial_noise_scale = 1.0  # scale of the exploration noise process (1.0 is the range of each action dimension)
noise_decay = 0.997  # decay rate (per episode) of the scale of the exploration noise process
exploration_mu = 0.0  # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_theta = 0.15  # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_sigma = 0.2  # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt
restore_ep = 0
OUTPUT_GRAPH = True  # graph output
RENDER = True  # render one worker
RENDER_EVERY = 100   # render every N episodes
N_WORKERS = 1  # number of workers
MAX_EP_STEP = 200  # maximum number of steps per episode (unless another limit is used)
MAX_GLOBAL_EP = 120  # total number of episodes
MAX_PROXY_EP = 1000      # total number of episodes to train on proxy, before switching to ipg simulations
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 80  # sets how often the global net is updated
GAMMA = 0.99  # discount factor
ENTROPY_BETA = 0.001  # entropy factor
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
SAFETY_ON = 0   # safety cages, 0 = disabled 1 = enabled
REPLAY_MEMORY_CAPACITY = int(1e4)  # capacity of experience replay memory
TRAUMA_MEMORY_CAPACITY = int(1e2)  # capacity of trauma memory
MINIBATCH_SIZE = 64  # size of the minibatch for training with experience replay
TRAJECTORY_LENGTH = 80  # size of the trajectory used in weight updates
UPDATE_ENDSTEP = False  # update at the end of episode using previous MB_SIZE experiences
UPDATE_TRAUMA = 16       # update weights using the trauma memory every UPDATE_TRAUMA updates
OFF_POLICY = False       # update off-policy using ER/TM
ON_POLICY = False        # update on-policy using online experiences
CHECKPOINT_EVERY = 100  # sets how often to save weights
HN_A = 50
HN_C = 200
LSTM_UNITS = 16
MAX_GRAD_NORM = 0.5

# Action Space Shape
N_S = 4  # number of states
N_A = 1  # number of actions
A_BOUND = [-1, 1]  # action bounds


def get_arguments():
    parser = argparse.ArgumentParser(description='RL training')
    parser.add_argument(
        '--lr_a',
        type=float,
        default=LR_A,
        help='Actor learning rate'
    )
    parser.add_argument(
        '--lr_c',
        type=float,
        default=LR_C,
        help='Critic learning rate'
    )
    parser.add_argument(
        '--gamma',
        type=float,
        default=GAMMA,
        help='Discount rate gamma'
    )
    parser.add_argument(
        '--max_eps',
        type=int,
        default=MAX_GLOBAL_EP,
        help='Checkpoint file to restore model weights from.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=MINIBATCH_SIZE,
        help='Batch size. Must divide evenly into dataset sizes.'
    )
    parser.add_argument(
        '--trajectory',
        type=float,
        default=TRAJECTORY_LENGTH,
        help='Length of trajectories in minibatches'
    )
    parser.add_argument(
        '--checkpoint_every',
        type=int,
        default=CHECKPOINT_EVERY,
        help='Number of steps before checkpoint.'
    )
    parser.add_argument(
        '--ent_beta',
        type=float,
        default=ENTROPY_BETA,
        help='Entropy coefficient beta'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=LOG_DIR,
        help='Directory to put the log data.'
    )
    parser.add_argument(
        '--store_metadata',
        type=bool,
        default=False,
        help='Storing debug information for TensorBoard.'
    )
    parser.add_argument(
        '--hn_a',
        type=int,
        default=HN_A,
        help='Number of hidden neurons in actor network.'
    )
    parser.add_argument(
        '--hn_c',
        type=int,
        default=HN_C,
        help='Number of hidden neurons in critic network.'
    )
    parser.add_argument(
        '--lstm_units',
        type=int,
        default=LSTM_UNITS,
        help='Number of lstm cells in actor network.'
    )
    parser.add_argument(
        '--store_results',
        action='store_true',
        help='Storing episode results in csv files.'
    )
    parser.add_argument(
        '--trauma',
        action='store_true',
        help='If true use trauma memory in off-policy updates.'
    )
    parser.add_argument(
        '--max_norm',
        type=float,
        default=MAX_GRAD_NORM,
        help='Maximum L2 norm of the gradient for gradient clipping.'
    )

    return parser.parse_args()


def calculate_reward(th, delta_th, x_rel):

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


# replay memory
replay_memory = deque(maxlen=REPLAY_MEMORY_CAPACITY)  # used for O(1) popleft() operation


def add_to_memory(experience):
    replay_memory.append(experience)


def sample_from_memory(minibatch_size):
    return random.sample(replay_memory, minibatch_size)


# trauma memory
trauma_buffer = deque(maxlen=TRAJECTORY_LENGTH)
trauma_memory = deque(maxlen=TRAUMA_MEMORY_CAPACITY)


def add_to_trauma(experience):
    trauma_memory.append(experience)


def sample_from_trauma(minibatch_size):
    return random.sample(trauma_memory, minibatch_size)


tf.reset_default_graph()

# placeholders
state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='state')
action_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_A], name='action')
reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='next_state')
is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='is_not_terminal')  # indicators (go into target computation)
is_training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')  # for dropout

# episode counter
episodes = tf.Variable(float(restore_ep), trainable=False, name='episodes')
episode_inc_op = episodes.assign_add(1)


# will use this to initialize both the actor network its slowly-changing target network with same structure
def generate_actor_network(s, trainable, reuse):
    hidden = tf.layers.dense(s, h1_actor, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
    hidden_drop = tf.layers.dropout(hidden, rate=dropout_actor, training=trainable & is_training_ph)
    hidden_2 = tf.layers.dense(hidden_drop, h2_actor, activation=tf.nn.relu, trainable=trainable, name='dense_1',
                               reuse=reuse)
    hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout_actor, training=trainable & is_training_ph)
    hidden_3 = tf.layers.dense(hidden_drop_2, h3_actor, activation=tf.nn.relu, trainable=trainable, name='dense_2',
                               reuse=reuse)
    hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout_actor, training=trainable & is_training_ph)

    # Recurrent network for temporal dependencies
    global lstm_layer
    lstm_layer = tf.keras.layers.LSTM(lstm_actor, stateful=True, return_sequences=True)
    rnn_out = lstm_layer(tf.expand_dims(hidden_drop_3, [0]))
    rnn_out = tf.reshape(rnn_out, [-1, lstm_actor])

    actions_unscaled = tf.layers.dense(rnn_out, N_A, trainable=trainable, name='dense_3', reuse=reuse)
    actions = A_BOUND[0] + tf.nn.sigmoid(actions_unscaled) * (
            A_BOUND[1] - A_BOUND[0])  # bound the actions to the valid range
    return actions


# actor network
with tf.variable_scope('actor'):
    # Policy's outputted action for each state_ph (for generating actions and training the critic)
    actions = generate_actor_network(state_ph, trainable=True, reuse=False)

# slow target actor network
with tf.variable_scope('slow_target_actor', reuse=False):
    # Slow target policy's outputted action for each next_state_ph (for training the critic)
    # use stop_gradient to treat the output values as constant targets when doing backprop
    slow_target_next_actions = tf.stop_gradient(generate_actor_network(next_state_ph, trainable=False, reuse=False))


# will use this to initialize both the critic network its slowly-changing target network with same structure
def generate_critic_network(s, a, trainable, reuse):
    state_action = tf.concat([s, a], axis=1)
    hidden = tf.layers.dense(state_action, h1_critic, activation=tf.nn.relu, trainable=trainable, name='dense',
                             reuse=reuse)
    hidden_drop = tf.layers.dropout(hidden, rate=dropout_critic, training=trainable & is_training_ph)
    hidden_2 = tf.layers.dense(hidden_drop, h2_critic, activation=tf.nn.relu, trainable=trainable, name='dense_1',
                               reuse=reuse)
    hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout_critic, training=trainable & is_training_ph)
    #hidden_3 = tf.layers.dense(hidden_drop_2, h3_critic, activation=tf.nn.relu, trainable=trainable, name='dense_2',
    #                           reuse=reuse)
    #hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout_critic, training=trainable & is_training_ph)
    q_values = tf.layers.dense(hidden_drop_2, 1, trainable=trainable, name='dense_3', reuse=reuse)
    return q_values


with tf.variable_scope('critic') as scope:
    # Critic applied to state_ph and a given action (for training critic)
    q_values_of_given_actions = generate_critic_network(state_ph, action_ph, trainable=True, reuse=False)
    # Critic applied to state_ph and the current policy's outputted actions for state_ph (for training actor via deterministic policy gradient)
    q_values_of_suggested_actions = generate_critic_network(state_ph, actions, trainable=True, reuse=True)

# slow target critic network
with tf.variable_scope('slow_target_critic', reuse=False):
    # Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
    slow_q_values_next = tf.stop_gradient(
        generate_critic_network(next_state_ph, slow_target_next_actions, trainable=False, reuse=False))

# isolate vars for each network
actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='actor')
slow_target_actor_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_actor')
critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic')
slow_target_critic_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='slow_target_critic')

# update values for slowly-changing targets towards current actor and critic
update_slow_target_ops = []
for i, slow_target_actor_var in enumerate(slow_target_actor_vars):
    update_slow_target_actor_op = slow_target_actor_var.assign(tau * actor_vars[i] + (1 - tau) * slow_target_actor_var)
    update_slow_target_ops.append(update_slow_target_actor_op)

for i, slow_target_var in enumerate(slow_target_critic_vars):
    update_slow_target_critic_op = slow_target_var.assign(tau * critic_vars[i] + (1 - tau) * slow_target_var)
    update_slow_target_ops.append(update_slow_target_critic_op)

update_slow_targets_op = tf.group(*update_slow_target_ops, name='update_slow_targets')

# One step TD targets y_i for (s,a) from experience replay
# = r_i + gamma*Q_slow(s',mu_slow(s')) if s' is not terminal
# = r_i if s' terminal
targets = tf.expand_dims(reward_ph, 1) + tf.expand_dims(is_not_terminal_ph, 1) * gamma * slow_q_values_next

# 1-step temporal difference errors
td_errors = targets - q_values_of_given_actions

# critic loss function (mean-square value error with regularization)
critic_loss = tf.reduce_mean(tf.square(td_errors))
for var in critic_vars:
    if not 'bias' in var.name:
        critic_loss += l2_reg_critic * 0.5 * tf.nn.l2_loss(var)

# critic optimizer
critic_train_op = tf.train.AdamOptimizer(lr_critic * lr_decay ** episodes).minimize(critic_loss)

# actor loss function (mean Q-values under current policy with regularization)
actor_loss = -1 * tf.reduce_mean(q_values_of_suggested_actions)
for var in actor_vars:
    if not 'bias' in var.name:
        actor_loss += l2_reg_actor * 0.5 * tf.nn.l2_loss(var)

# actor optimizer
# the gradient of the mean Q-values wrt actor params is the deterministic policy gradient (keeping critic params fixed)
actor_train_op = tf.train.AdamOptimizer(lr_actor * lr_decay ** episodes).minimize(actor_loss, var_list=actor_vars)

# initialise noise process
noise_process = np.zeros(N_A)

# initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

args = get_arguments()  # get arguments

for i in range(len(DIRS)):
    print('Model no. %d' % i)

    global_rewards = []
    global_episodes = 0

    LOG_DIR = str(fpath + DIRS[i])
    checkpoint_path = fpath + DIRS[i] + '/' + MODELS[i]
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint_path)
    print('Restored model: %s' % str(checkpoint_path))
    # tf.global_variables_initializer().run()

    # merge tensorboard summaries
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(LOG_DIR, sess.graph)

    total_step = 1
    buffer_s, buffer_a, buffer_r = [], [], []

    # scenario array
    arr_scen = []


    # Initiate API connection to IPG CarMaker
    pythonapi.api_setup()
    pythonapi.subscribe_quants()
    pythonapi.ApoClnt_PollAndSleep()  # poll client

    trauma_counter = 0      # count how often to update from trauma memory

    # loop episodes
    while global_episodes < args.max_eps:

        # reset lstm layer
        lstm_layer.reset_states()

        # set states to zero
        b = 0
        v_rel = 0
        v = 0
        x_rel = 0
        a = 0
        t = 0
        t_h = 0

        ER_buffer = []  # experience replay buffer
        trauma_buffer.clear()  # clear trauma buffer

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

        arr_rewards = []    # rewards list

        # lead vehicle states
        # Here we have two methods for creating lead vehicle states
        # Option 1: Create new random trajectories from the traffic.py module
        # T_lead, X_lead, V_lead, A_lead = traffic.lead_vehicle()
        # Option 2: Use pre-generated trajectories (allows comparison to other tests performed using same trajectories)
        T_lead = []
        X_lead = []
        V_lead = []
        A_lead = []
        # read lead vehicle states from the corresponding traffic file (generated by traffic.py)
        with open('S:/Research/safeav/Sampo/condor-a2c/test/traffic_data_sl/' + str(global_episodes + 1) + '.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                T_lead.append(float(row['t']))  # time
                X_lead.append(float(row['x']))  # long. position
                V_lead.append(float(row['v']))  # velocity
                A_lead.append(float(row['a']))  # acceleration


        print('\ntest no. %d' % global_episodes)

        # load test run
        # Option 1: Use random coefficients of frictions
        #scen = random.randint(1, 25)
        #arr_scen.append(scen)
        # Option 2: Use a pre-determined list of coefficient of frictions
        with open('S:/Research/safeav/Sampo/condor-a2c/test/traffic_data_sl/scens.csv') as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                arr_scen.append(float(row['s']))  # test run id
        scen = int(arr_scen[global_episodes - 1])
        cof = 0.375 + scen * 0.025  # calculate coefficient of friction
        pythonapi.sim_loadrun2(scen)

        # Run training using Ipg Proxy
        ep_r = 0  # set ep reward to 0
        ep_nr = 0  # set normalised ep reward to 0

        # start simulation
        pythonapi.sim_start()
        pythonapi.sim_waitready()

        # read host states
        t = pythonapi.get_time()  # time
        v = pythonapi.get_hostvel()  # host velocity
        a = pythonapi.get_longacc()  # host longitudinal acceleration
        x = pythonapi.get_hostpos()  # host longitudinal position

        # lead vehicle states
        t_iter = int(t // 0.02)  # current time step
        v_rel = V_lead[t_iter] - v  # relative velocity
        x_rel = X_lead[t_iter] - x  # relative distance
        if v != 0:  # check for division by 0
            t_h = x_rel / v
        else:
            t_h = x_rel

        inputs = [v_rel, t_h, v, a]  # define input array
        crash = 0  # variable for checking if a crash has occurred (0=no crash, 1=crash)
        prev_output = 0
        # loop time-steps
        while pythonapi.sim_isrunning() != 0:  # check if simulation is running
            if t >= 0:  # to avoid errors check that time is not zero

                b += 1

                # evaluate neural network output
                # action
                action = sess.run(actions,
                                             feed_dict={state_ph: np.reshape(inputs, (1, N_S)),
                                                        is_training_ph: False})
                arr_y_0.append(float(action))

                output = action
                sc = 0

                arr_y_sc.append(float(output))
                arr_sc.append(sc)

                # convert normalised output to gas and brake signals
                if output < 0:  # output brake command
                    gas = 0
                    brake = abs(output)
                elif output > 0:  # output gas command
                    gas = output
                    brake = 0
                elif output == 0:  # both outputs are zero
                    gas = 0
                    brake = 0
                else:  # something has gone wrong
                    gas = 0
                    brake = 0
                    print('invalid control signal, setting pedal values to 0')

                #  send commands to carmaker
                pythonapi.set_gas(ctypes.c_double(gas))
                pythonapi.set_brake(ctypes.c_double(brake))

                # read new states
                # read host states
                pythonapi.ApoClnt_PollAndSleep()  # poll client
                t_ = pythonapi.get_time()  # time
                v_ = pythonapi.get_hostvel()  # host velocity
                a_ = pythonapi.get_longacc()  # host longitudinal acceleration
                x_ = pythonapi.get_hostpos()  # host longitudinal position

                # lead vehicle states
                t_iter_ = int(t_ // 0.02)  # current time step
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

                # define new input array
                inputs_ = [v_rel_, t_h_, v_, a_]

                # calculate reward
                if (t_ - t) != 0:
                    delta_th = (t_h_ - t_h) / (t_ - t)
                else:
                    delta_th = 0

                reward = calculate_reward(t_h_, delta_th, x_rel_)
                n_reward = calculate_reward2(t_h_, delta_th, x_rel_)  # normalised reward

                ep_r += reward
                ep_nr += n_reward
                arr_rewards.append(reward)

                # add to trauma memory buffer
                trauma_buffer.append((inputs, action, n_reward, inputs_))

                # stop simulation if a crash occurs
                if x_rel_ <= 0:
                    crash = 1
                    pythonapi.sim_stop()
                    print('crash occurred: simulation run stopped')
                    if len(trauma_buffer) >= TRAJECTORY_LENGTH:
                        add_to_trauma(trauma_buffer)

                # update buffers
                buffer_s.append(inputs)
                buffer_a.append(action)
                buffer_r.append(n_reward)

                ER_buffer.append((inputs, action, n_reward, inputs_))
                # if buffer > mb_size add to experience replay and empty buffer
                if len(ER_buffer) >= args.trajectory:
                    add_to_memory(ER_buffer)
                    ER_buffer = []

                # update weights
                if total_step % UPDATE_GLOBAL_ITER == 0:  # update global and assign to local net

                    #self.AC.update_global(feed_dict)  # actual training step, update global ACNet
                    buffer_s, buffer_a, buffer_r = [], [], []
                    #self.AC.pull_global()  # get global parameters to local ACNet

                # update state variables
                inputs = inputs_
                t = t_
                v = v_
                a = a_
                x = x_
                v_rel = v_rel_
                x_rel = x_rel_
                t_h = t_h_
                t_iter = t_iter_
                prev_output = output
                total_step += 1
                # pythonapi.ApoClnt_PollAndSleep()  # poll client every now and then

        # Run an update step at the end of episode
        if UPDATE_ENDSTEP:

            # v_s_ = 0  # terminal state
            # buffer_v_target = []
            # for r in buffer_r[::-1]:  # reverse buffer r
            #    v_s_ = r + GAMMA * v_s_
            #    buffer_v_target.append(v_s_)
            # buffer_v_target.reverse()

            # buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(
            #    buffer_v_target)
            minibatch = trauma_buffer
            batch_s = np.asarray([elem[0] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_S)
            batch_a = np.asarray([elem[1] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_A)
            batch_r = np.asarray([elem[2] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, 1)

        if OFF_POLICY:
            for off_pol_i in range(0, args.batch_size):
                if args.trauma and off_pol_i == 0 and len(trauma_memory) >= 1:  # run one update from trauma memory
                    minibatch = sample_from_trauma(1)[-1]
                else:
                    # grab N (s,a,r,s') tuples from replay memory
                    minibatch = sample_from_memory(1)[-1]  # sample and flatten minibatch

                batch_s = np.asarray([elem[0] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_S)
                batch_a = np.asarray([elem[1] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, N_A)
                batch_r = np.asarray([elem[2] for elem in minibatch]).reshape(TRAJECTORY_LENGTH, 1)

        buffer_s, buffer_a, buffer_r = [], [], []  # empty buffers

        # Update summaries and print episode performance before starting next episode

        # update tensorboard summaries

        # append episode reward to list
        global_rewards.append(ep_r)

        # print summary
        print(
            "Ep:", global_episodes,
            "| Ep_r: %i" % global_rewards[-1],
            "| Avg. Reward: %.5f" % np.mean(arr_rewards),
            "| Min. Reward: %.5f" % np.min(arr_rewards),
            "| Max. Reward: %.5f" % np.max(arr_rewards),
            "| Avg. Timeheadway: %.5f" % np.mean(arr_th),
        )
        print(b)
        global_episodes += 1

        # if args.store_results:
        # always store results when testing
        if not os.path.exists(LOG_DIR + '/ipg_results'):
            os.makedirs(LOG_DIR + '/ipg_results')
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
        with open(LOG_DIR + '/ipg_results/' + str(global_episodes) + '.csv', 'w', newline='\n') as f:
            wr = csv.writer(f, delimiter=',')
            rows = zip(arr_t, arr_j, arr_v, arr_a, arr_v_leader, arr_a_leader, arr_x, arr_dv, arr_th,
                       arr_y_0,
                       arr_y_sc, arr_sc, arr_cof)
            wr.writerow(headers)
            wr.writerows(rows)


