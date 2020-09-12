# featureExtractors.py
# --------------------
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


"Feature extractors for Pacman game states"

from game import Directions, Actions
import util
import math

class FeatureExtractor:
    def getFeatures(self, state, action, agent):
        """
          Returns a dict from features to counts
          Usually, the count will just be 1.0 for
          indicator functions.
        """
        util.raiseNotDefined()

class IdentityExtractor(FeatureExtractor):
    def getFeatures(self, state, action, agent):
        feats = util.Counter()
        feats[(state,action)] = 1.0
        return feats

class CoordinateExtractor(FeatureExtractor):
    def getFeatures(self, state, action, agent):
        feats = util.Counter()
        feats[state] = 1.0
        feats['x=%d' % state[0]] = 1.0
        feats['y=%d' % state[0]] = 1.0
        feats['action=%s' % action] = 1.0
        return feats

def closestFood(pos, food, walls):
    """
    closestFood -- this is similar to the function that we have
    worked on in the search project; here its all in one place
    """
    fringe = [(pos[0], pos[1], 0)]
    expanded = set()
    while fringe:
        pos_x, pos_y, dist = fringe.pop(0)
        if (pos_x, pos_y) in expanded:
            continue
        expanded.add((pos_x, pos_y))
        # if we find a food at this location then exit
        if food[pos_x][pos_y]:
            return dist
        # otherwise spread out from the location to its neighbours
        nbrs = Actions.getLegalNeighbors((pos_x, pos_y), walls)
        for nbr_x, nbr_y in nbrs:
            fringe.append((nbr_x, nbr_y, dist+1))
    # no food found
    return None

class SimpleExtractor(FeatureExtractor):
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getFeatures(self, state, action, agent):
        previuse_state = state
        myPState = previuse_state.getAgentState(agent.index)
        state = state.generateSuccessor(agent.index, action)
        food = agent.getFood(state)
        myState = state.getAgentState(agent.index)
        myPos = myState.getPosition()
        walls = state.getWalls()

        features = util.Counter()

        enemies = [state.getAgentState(i) for i in agent.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        ghostDefenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]
        foodDefend = agent.getFoodYouAreDefending(state).asList()
        features['bias'] = 1

        if agent.type == "Defence":
            if len(invaders) > 0:
                dists = [agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                features['invaderDistance'] = min(dists) / 10
                features['numInvaders'] = len(invaders)
                features['scared'] = myState.scaredTimer / 5

                features['agent-food'] = len(foodDefend) / 10
            else:
                features['go-atk'] = 1

        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(agent.index)

        myPos = (x,y)

        foodList = agent.getFood(state).asList()
        foodNumb = len(foodList)



        x, y = myPos
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        ghostAround = sum((next_x, next_y) in Actions.getLegalNeighbors(g.getPosition(), walls) for g in ghostDefenders)

        if ghostAround and myState.isPacman:
            features.clear()
            features["run"] = 3
            return features
        elif food[next_x][next_y]:
            features["eats-food"] = 1

        if agent.type == "Offense":
            features['num-food'] = -foodNumb / 4


        features["carrying-food"] = myState.numCarrying

        foodListPrevouse = agent.getFood(previuse_state).asList()
        eaten = len(foodListPrevouse) - foodNumb
        if eaten == 1:
            features["eats-food"] = 1
        if len(foodList) > 0:
            minDistance = min([agent.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance / 1000
        else:
            features['run'] = 1
        if len(ghostDefenders) > 0:
            minDistance = min([agent.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghostDefenders])
            if minDistance <= 3:
                # features['distanceToFood'] = 0
                # features['num-food'] = 0
                # features['carrying-food'] = 0
                # features['eats-food'] = 0
                features.clear()
                features['run'] = 1
                features['distanceToGhost'] = 2 * minDistance
                features['number-of-moves'] = len(Actions.getLegalNeighbors(myPos, walls)) - 1
                return features


        features['number-of-moves'] = len(Actions.getLegalNeighbors(myPos, walls)) - 1
        if myState.numCarrying > 0:
            distanceHome = agent.getMazeDistance(myPos, (1,myPos[1])) / 10 # ovde je bilo 4
            features['return-home'] = distanceHome * myState.numCarrying / 4

        features['score'] = agent.getScore(state) - agent.getScore(previuse_state)


        return features

def min_distance(point1, point2, walls):
    states = [(point1[0], point1[1], 0)]
    passed = set()
    while states:
        x, y, distance = states.pop(0)


        if (x, y) in passed:
            continue
        passed.add((x, y))

        if (x, y) == point2:
            return distance

        nextStates = Actions.getLegalNeighbors((x, y), walls)
        for next_x, next_y in nextStates:
            states.append((next_x, next_y, distance+1))
    return None
