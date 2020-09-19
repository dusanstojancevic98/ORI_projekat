# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import os
from collections import deque
from datetime import date, datetime

from captureAgents import CaptureAgent
from featureExtractors import *
from capture import AgentRules
import random, time, util
from game import Directions
import game

import tensorflow.keras

# plaidml.keras.install_backend()
#
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import keras.backend as K

import numpy as np

from ORI_projekat_1.game import Actions

num_actions = 5


#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first='ApproximateQAgent', second='ApproximateQAgent', **args):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    # The following line is an example only; feel free to change it.
    return [eval(first)(firstIndex), eval(second)(secondIndex)]


##########
# Agents #
##########

class DummyAgent(CaptureAgent):
    """
    A Dummy agent to serve as an example of the necessary agent structure.
    You should look at baselineTeam.py for more details about how to
    create an agent as this is the bare minimum.
    """

    def registerInitialState(self, gameState):
        """
        This method handles the initial setup of the
        agent to populate useful fields (such as what team
        we're on).

        A distanceCalculator instance caches the maze distances
        between each pair of positions, so your agents can use:
        self.distancer.getDistance(p1, p2)

        IMPORTANT: This method may run for at most 15 seconds.
        """

        '''
        Make sure you do not delete the following line. If you would like to
        use Manhattan distances instead of maze distances in order to save
        on initialization time, please take a look at
        CaptureAgent.registerInitialState in captureAgents.py.
        '''
        CaptureAgent.registerInitialState(self, gameState)

        '''
        Your initialization code goes here, if you need any.
        '''

    def chooseAction(self, gameState):
        """
        Picks among actions randomly.
        """
        actions = gameState.getLegalActions(self.index)

        '''
        You should change this in your own agent.
        '''

        return random.choice(actions)


class QLearningAgent(CaptureAgent):

    def __init__(self, index, timeForComputing=.1, **args):
        "You can initialize Q-values here..."
        CaptureAgent.__init__(self, index, timeForComputing=.1)
        self.qValues = util.Counter()
        self.epsilon = args['epsilon']
        self.alpha = args['alpha']
        self.gamma = args['gamma']
        self.numTraning = args['numTraining']

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        possibleStateQValues = util.Counter()
        for action in state.getLegalActions(self.index):
            possibleStateQValues[action] = self.getQValue(state, action)

        if len(possibleStateQValues) > 0:
            return possibleStateQValues[possibleStateQValues.argMax()]
        return 0.0

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        possibleStateQValues = util.Counter()
        possibleActions = state.getLegalActions(self.index)
        if len(possibleActions) == 0:
            return None

        for action in possibleActions:
            possibleStateQValues[action] = self.getQValue(state, action)

        best_actions = []
        best_value = possibleStateQValues[possibleStateQValues.argMax()]
        # print("BV:".format(best_value))
        for action, value in possibleStateQValues.items():
            # print("Act, val: {}, {}".format(action, value))
            if value == best_value:
                best_actions.append(action)
        # print("best actions: {} \n best_value:{}:\n possiblesval: {}".format(best_actions, best_value,
        #                                                                          possibleStateQValues))
        chosen_action = random.choice(best_actions)

        return chosen_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = state.getLegalActions(self.index)
        action = None

        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """




        self.qValues[(state, action)] = self.getQValue(state, action) + self.alpha * (
                reward + self.gamma * self.getValue(nextState) - self.getQValue(state, action))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, index, timeForComputing=.1, epsilon=0.2, gamma=0.9, alpha=0.05, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        QLearningAgent.__init__(self, index, timeForComputing, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)

        # self.doAction(state,action)
        return action


class DQN(keras.Model):
    """Dense neural network class."""




    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(32, activation="relu")
        self.dense3 = keras.layers.Dense(num_actions, dtype=tensorflow.float32)  # No activation


    def call(self, x, **kwargs):
        """Forward pass."""
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class ReplayBuffer(object):
    """Experience replay buffer that samples uniformly."""

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, num_samples):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        idx = np.random.choice(len(self.buffer), num_samples)
        for i in idx:
            elem = self.buffer[i]
            state, action, reward, next_state, done = elem
            states.append(np.array(state, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        return states, actions, rewards, next_states, dones



class DQNQAgent(PacmanQAgent):
    """
       DQNQLearningAgent
    """
    tip = 0

    def __init__(self, index, timeForComputing=.1, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()

        PacmanQAgent.__init__(self, index, timeForComputing, **args)
        self.weights = util.Counter()
        if DQNQAgent.tip % 2 == 0:
            self.type = "Defence"
            DQNQAgent.tip += 1


        else:
            self.type = "Offense"
        # DQN part
        self.optimizer = tensorflow.keras.optimizers.Adam(1e-4)
        self.main_nn = DQN()
        self.target_nn = DQN()
        self.mse = tensorflow.losses.MeanSquaredError()

        self.batch_size = 32

        self.buffer = ReplayBuffer(200)

        self.actions = Actions._directionsAsList

        self.i = 0

    @tensorflow.function
    def train_step(self, states, actions, rewards, next_states, dones):
        """Perform a training iteration on a batch of data sampled from the experience
        replay buffer."""
        # Calculate targets.
        next_qs = self.target_nn(next_states)
        max_next_qs = tensorflow.reduce_max(next_qs, axis=-1)
        target = rewards + (1. - dones) * self.gamma * max_next_qs
        with tensorflow.GradientTape() as tape:
            qs = self.main_nn(states)
            action_masks = tensorflow.one_hot(actions, num_actions)
            masked_qs = tensorflow.reduce_sum(action_masks * qs, axis=-1)
            loss = self.mse(target, masked_qs)
        grads = tape.gradient(loss, self.main_nn.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.main_nn.trainable_variables))
        return loss

    def getAction(self, state):
        legalActions = state.getLegalActions(self.index)
        action = None
        if len(legalActions) > 0:
            if util.flipCoin(self.epsilon):
                action = random.choice(legalActions)
            else:
                return self.actions[tensorflow.argmax(self.main_nn(state)[0]).numpy()]  # Greedy action for state.
        return action



    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        qValue = 0.0
        features = self.featExtractor.getFeatures(state, action, self)
        # if self.type == "Offense":
        #     print(self.weights)
        for key in features.keys():
            qValue += (self.weights[key] * features[key])

        # if self.type == "Offense":
        #     print("Q: {}".format(qValue))
        return qValue

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """

        self.buffer.add()

        if self.i % 2000 == 0:
            self.target_nn.set_weights(self.main_nn.get_weights())

        if len(self.buffer) >= self.batch_size:
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
            loss = self.train_step(states, actions, rewards, next_states, dones)

        features = self.featExtractor.getFeatures(state, action, self)

        diff = self.alpha * ((reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action))

        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]
        # if self.type == "Offense":
        #     print("Diff: {}".format(diff))
        #     print(self.weights)

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)
        self.numTraning += 1
        print(self.type)
        print(self.weights)

        if not self.type == "Defence":
            print("Reward: " + str(state.reward))
            if state.reward > 0:
                print("Won!!!!!!!!!!")
            print("Number of Training: " + str(self.numTraning) + "\n")
        # did we finish training?
        # if self.episodesSoFar == self.numTraining:
        #   # you might want to print your weights here for debugging
        #   "*** YOUR CODE HERE ***"
        #   pass
