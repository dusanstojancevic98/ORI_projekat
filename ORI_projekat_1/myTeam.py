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


from captureAgents import CaptureAgent
from featureExtractors import *
from capture import AgentRules
import random, time, util
from game import Actions
import game

#################
# Team creation #
#################
def createTeam(firstIndex, secondIndex, isRed,
               first = 'ApproximateQAgent', second = 'ApproximateQAgent',**args):
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

  def __init__(self, index, timeForComputing = .1, **args):
    "You can initialize Q-values here..."
    CaptureAgent.__init__(self, index, timeForComputing = .1)
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

    for action, value in possibleStateQValues.items():
      if value == best_value:
        best_actions.append(action)

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
    legalActions = state.getLegalActions( self.index )
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

  def __init__(self, index, timeForComputing = .1, epsilon=0.15,gamma=0.9,alpha=0.05, numTraining=0, **args):
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
    QLearningAgent.__init__(self, index, timeForComputing,**args)



class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  tip = 0
  def __init__(self, index, timeForComputing = .1,extractor='SimpleExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()

    PacmanQAgent.__init__(self, index, timeForComputing,**args)
    self.weights = util.Counter()
    if ApproximateQAgent.tip == 0:
        self.type = "Defence"
        ApproximateQAgent.tip+=1
        # self.weights = {'invaderDistance' : -1,
        #                 'numInvaders': -1,
        #                 'agent-food': 20,
        #                 'scared' : -1}

    else:
        self.type = "Offense"

        # self.weights = {"carrying-food": 5,
        #                 'num-food': 10,
        #                 "eats-food": 10,
        #                 'distanceToFood': -1,
        #                 'distanceToGhost': 0.3,
        #                 "enable-eating": 10,
        #                 "#-of-ghosts-1-step-away": -2,
        #                 "eaten": -10}

    # za sad najbolji rezovi
    # Defence
    # {'bias': -1.315733339507866, 'go-atk': 5.589852178308997, 'carrying-food': 4.470346842023858, 'distanceToFood': -0.4380758428443762, 'score': 9.802956193209763, 'invaderDistance': -1.1818754486438479, 'numInvaders': -6.905585517816883, 'scared': -0.5867167092036796, 'agent-food': 6.87059374272093, 'eats-food': 0.38917595473673783, 'run': 0.590419694785678, 'distanceToGhost': 0.03954786317624587, 'number-of-moves': 0.41619784913633, 'return-home': -0.539000624405877}
    # Offense
    # {'bias': -13.463570816602461, 'num-food': -9.04933348210986, 'carrying-food': 37.537768140286644, 'distanceToFood': -20.457554363759694, 'score': 50.253712339050054, 'eats-food': 0.5117247182743323, 'return-home': -12.729865224697745, 'run': -5.292313710288039, 'distanceToGhost': 0.5878365963624352, 'number-of-moves': 0.5509506295469199}


  def getWeights(self):
    return self.weights

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)

    #self.doAction(state,action)
    nextState = state.generateSuccessor( self.index, action )
    reward = self.calculate_reward(state,action,nextState)
    self.update(state,action,nextState,reward)
    return action

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    qValue = 0.0
    features = self.featExtractor.getFeatures(state, action, self)

    for key in features.keys():
          qValue += (self.weights[key] * features[key])

    return qValue


  def calculate_reward(self,state,action,nextState):
    foodList1 = self.getFood(nextState).asList()
    foodList2 = self.getFood(state).asList()
    foodDefend1 = self.getFoodYouAreDefending(nextState).asList()
    foodDefend2 = self.getFoodYouAreDefending(state).asList()
    myNState = nextState.getAgentState(self.index)
    myState = state.getAgentState(self.index)
    myPos = myNState.getPosition()
    myPPos = myState.getPosition()
    foodDif = -(len(foodList1) - len(foodList2))
    foodTeamDif = (len(foodDefend1) - len(foodDefend2))
    if foodDif < 0 :
      foodDif = -1
    if not myNState.isPacman and not myState.isPacman:
      penalty = -0.5
    else:
      penalty = 0
    if len(foodDefend1) < 10:
      defendPen = -2
    else:
      defendPen = 0
    agent_could_be_eaten = 0


    walls = state.getWalls()

    x, y = myPos
    dx, dy = Actions.directionToVector(action)
    next_x, next_y = int(x + dx), int(y + dy)

    enemies = [state.getAgentState(i) for i in self.getOpponents(state)]
    ghostDefenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    ghostAround = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghostDefenders)

    if self.type == "Offense":
      if ghostAround:
        agent_could_be_eaten -= 20
        if len(Actions.getLegalNeighbors(myPos, walls)) - 1 == 1:
          agent_could_be_eaten -= 10
      reward = 50 * (self.getScore(nextState) - self.getScore(state)) + penalty +  myState.numCarrying + agent_could_be_eaten
    else:
      reward = 10 * (self.getScore(nextState) - self.getScore(state)) + 2* foodTeamDif + defendPen


    return reward


  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    features = self.featExtractor.getFeatures(state, action,self)
    diff = self.alpha * ((reward + self.gamma * self.getValue(nextState)) - self.getQValue(state, action))

    for feature in features.keys():
      self.weights[feature] = self.weights[feature] + diff * features[feature]
    # if self.type == "Offense":
    #   print(self.type)
    #   print(action)
    #   print(reward)
    #   print(self.weights)


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
