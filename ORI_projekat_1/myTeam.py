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
from datetime import date, datetime

from captureAgents import CaptureAgent
from featureExtractors import *
from capture import AgentRules
import random, time, util
from game import Directions
import game


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


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    tip = 0

    def __init__(self, index, timeForComputing=.1, extractor='SimpleExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()

        PacmanQAgent.__init__(self, index, timeForComputing, **args)
        self.weights = util.Counter()
        if ApproximateQAgent.tip % 2 == 0:
            self.type = "Defence"
            ApproximateQAgent.tip += 1
            # self.weights = {'score': 20.47356009035358, 'agent-food': 38.52028097266377, 'invaderDistance': 25.90204344890279, 'numInvaders': -87.81723216036399, 'distance-ghost': -42.862848199584064, 'n-returned': 4.510724178773814, 'returned': -8.739123250708507, 'rip': -32.841289450028924, 'agent-eaten': -25.22588487404809, 'distance-scared-ghost': 0.14864383729580835, 'eats-food': 46.14520982045911}



            # PRVA USPESNA
            # self.weights = {'agent-food': 68.20348111559287, 'invaderDistance': 11.934755962856336, 'numInvaders': -127.23059393037634, 'agent-eaten': -9.907665210358681, 'eats-food': 40.36366979564748}


        else:
            self.type = "Offense"
            # self.weights = {'score': -5.064286088768164, 'agent-food': -11.31883425723904, 'num-food': 8.67516304615496, 'distance-capsule': -5.158073428871487, 'invaderDistance': 29.849231305566832, 'numInvaders': 16.716719210188582, 'distance-ghost': 373.05534720176627, 'n-returned': 66.21598683269255, 'returned': -2.1995965349422844, 'eats-food': 45.87131882469542, 'rip': -20.988655403674606, 'agent-eaten': -4.911390803884702, 'distance-scared-ghost': -4.277345362236672}


            # PRVA USPESNA
            # self.weights = {'agent-food': 5.60051272121332, 'num-food': 212.6511052356747, 'distance-food': -72.83238615807366, 'distance-capsule': -112.05940450838142, 'dist-start': -432.82672797854974, 'distance-ghost': 6.744978312108721, 'invaderDistance': 19.302142156423415, 'numInvaders': -37.0512715156541, 'eats-food': 45.631237461610404, 'agent-eaten': -38.337500383379854, 'distance-scared-ghost': 18.50925429875891, 'score': 328.668044926868}

            # PROVERA MALA SMRTNOST
            # self.weights = {'agent-food': 9.199616412657567, 'num-food': 248.12993914115313, 'distance-food': 4.558181181375859,'distance-capsule': -7.732628777983125, 'dist-start': -170.84407395839963,             'distance-ghost': -23.908719547081308, 'invaderDistance': -7.288905435011315,             'numInvaders': 2.0729728145112665, 'agent-eaten': -2.0407008012272856, 'eats-food': 48.65018915591999,             'distance-scared-ghost': -0.16978866581818733, 'score': 41.88136160804298}

    def getWeights(self):
        return self.weights

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
        features = self.featExtractor.getFeatures(state, action, self)

        diff = self.alpha * ((reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action))

        # if self.type == "Offense":
        #     print("Reward: {}".format(reward))
        #     print(" Off Diff: {}".format(diff))
        #     print(self.weights)
        #     print(features)
        # else:
        #     print(" Def Diff: {}".format(diff))


        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + diff * features[feature]


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
