# ddpg_network.py loads up the trained ddpg model into its own tf session, and allows external modules
# to call its inference function to observe the network outputs, this approach was used as it makes
# using multiple neural networks in the same job simpler
# however the approach of handling rnn_state outside of ddpg_network currently is not ideal
import tensorflow as tf
import os
import numpy as np

# hyperparameters
gamma = 0.99  # reward discount factor
h1_actor = 400  # hidden layer 1 size for the actor
h2_actor = 300  # hidden layer 2 size for the actor
h3_actor = 200  # hidden layer 3 size for the actor
h1_critic = 400  # hidden layer 1 size for the critic
h2_critic = 300  # hidden layer 2 size for the critic
h3_critic = 200  # hidden layer 3 size for the critic
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
replay_memory_capacity = int(1e6)  # capacity of experience replay memory
minibatch_size = 64  # size of minibatch from experience replay memory for updates
initial_noise_scale = 1.0  # scale of the exploration noise process (1.0 is the range of each action dimension)
noise_decay = 0.997  # decay rate (per episode) of the scale of the exploration noise process
exploration_mu = 0.0  # mu parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_theta = 0.15  # theta parameter for the exploration noise process: dXt = theta*(mu-Xt)*dt + sigma*dWt
exploration_sigma = 0.2  # sigma parameter for the exploration noise process: dXt = theta*(mu-Xt	)*dt + sigma*dWt
V_MIN = 17
V_MAX = 30
restore_from = None
restore_ep = 0

# game parameters
#env = gym.make(env_to_use)
# Action Space Shape
N_S = 4  # number of states
N_A = 1  # number of actions
A_BOUND = [-1, 1]  # action bounds
MODEL_FILE = 'model-ep-4999-final-final.ckpt'
DATA_DIR = './data/'
LOG_DIR = '/vol/research/safeav/Sampo/condor-a2c/test/log_ddpg/ddpg_ssc_lstm'


class DdpgNetwork(object):
    """implements the Ddpg network model for estimating vehicle host actions"""

    def __init__(self):
        # set up tf session and model
        rl_graph = tf.Graph()
        rl_config = tf.ConfigProto()
        rl_config.gpu_options.allow_growth = True
        self.sess_rl = tf.Session(graph=rl_graph, config=rl_config)
        self.recurrent = True

        with self.sess_rl.as_default():
            with rl_graph.as_default():
                # BUILD MODEL
                # placeholders
                self.state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='state')
                self.action_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_A], name='action')
                self.reward_ph = tf.placeholder(dtype=tf.float32, shape=[None], name='reward')
                self.next_state_ph = tf.placeholder(dtype=tf.float32, shape=[None, N_S], name='next_state')
                self.is_not_terminal_ph = tf.placeholder(dtype=tf.float32, shape=[None],
                                                         name='is_not_terminal')  # indicators (go into target computation)
                self.is_training_ph = tf.placeholder(dtype=tf.bool, shape=(), name='is_training')  # for dropout

                # create networks
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
                    q_values_of_given_actions = self.generate_critic_network(self.state_ph, self.action_ph,
                                                                             trainable=True, reuse=False)
                    # Critic applied to state_ph and the current policy's outputted actions for state_ph (for training actor via deterministic policy gradient)
                    q_values_of_suggested_actions = self.generate_critic_network(self.state_ph, self.actions,
                                                                                 trainable=True, reuse=True)

                # slow target critic network
                with tf.variable_scope('slow_target_critic', reuse=False):
                    # Slow target critic applied to slow target actor's outputted actions for next_state_ph (for training critic)
                    slow_q_values_next = tf.stop_gradient(
                        self.generate_critic_network(self.next_state_ph, slow_target_next_actions, trainable=False,
                                                     reuse=False))

                # GET SAVED WEIGHTS
                saver = tf.train.Saver()
                checkpoint_path = os.path.join(LOG_DIR, MODEL_FILE)
                saver.restore(self.sess_rl, checkpoint_path)
        print('rl_model: Restored model: %s' % MODEL_FILE)

    # Build the network
    def generate_actor_network(self, s, trainable, reuse):
        hidden = tf.layers.dense(s, h1_actor, activation=tf.nn.relu, trainable=trainable, name='dense', reuse=reuse)
        hidden_drop = tf.layers.dropout(hidden, rate=dropout_actor, training=trainable & self.is_training_ph)
        hidden_2 = tf.layers.dense(hidden_drop, h2_actor, activation=tf.nn.relu, trainable=trainable, name='dense_1',
                                   reuse=reuse)
        hidden_drop_2 = tf.layers.dropout(hidden_2, rate=dropout_actor, training=trainable & self.is_training_ph)
        hidden_3 = tf.layers.dense(hidden_drop_2, h3_actor, activation=tf.nn.relu, trainable=trainable, name='dense_2',
                                   reuse=reuse)
        hidden_drop_3 = tf.layers.dropout(hidden_3, rate=dropout_actor, training=trainable & self.is_training_ph)

        if self.recurrent:
            # Recurrent network for temporal dependencies
            self.lstm_layer = tf.keras.layers.LSTM(lstm_actor, stateful=True, return_sequences=True)
            rnn_out = self.lstm_layer(tf.expand_dims(hidden_drop_3, [0]))
            rnn_out = tf.reshape(rnn_out, [-1, lstm_actor])
            actions_unscaled = tf.layers.dense(rnn_out, N_A, trainable=trainable, name='dense_3', reuse=reuse)
        else:
            actions_unscaled = tf.layers.dense(hidden_drop_3, N_A, trainable=trainable, name='dense_3', reuse=reuse)
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

    def inference(self, s):
        s = np.reshape(s, (1, N_S))  # reshape state vector

        return self.sess_rl.run(self.actions, {self.state_ph: s,
                                               self.is_training_ph: False})[0]
