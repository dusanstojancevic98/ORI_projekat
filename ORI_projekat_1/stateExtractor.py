from ORI_projekat_1 import util


class SimpleExtractor():
    """
    Returns simple features for a basic reflex Pacman:
    - whether food will be eaten
    - how far away the next food is
    - whether a ghost collision is imminent
    - whether a ghost is one step away
    """

    def getStateTensor(self, state, action, agent):
        # extract the grid of food and wall locations and get the ghost locations
        previuse_state = state
        myPState = state.getAgentState(agent.index)
        state = state.generateSuccessor(agent.index, action)
        food = agent.getFood(state)
        pfood = agent.getFood(previuse_state)
        foodDefend = agent.getFoodYouAreDefending(state).asList()

        myState = state.getAgentState(agent.index)
        myPos = myState.getPosition()
        myPPos = myPState.getPosition()
        walls = state.getWalls()

        features = util.Counter()

        enemies = [state.getAgentState(i) for i in agent.getOpponents(state)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        ghostDefenders = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer == 0]
        ghostsScared = [a for a in enemies if not a.isPacman and a.getPosition() != None and a.scaredTimer > 0]
        scaredTimer = myState.scaredTimer

        carry = myState.numCarrying

        # old
        # features['n-returned'] = myState.numReturned / 10
        # features['n-carrying'] = myState.numCarrying / 10
        # new
        if agent.type == "Offense":
            score = state.getScore()
            if score > 0:
                features["score"] = score / 20

        # features['legal-actions'] = len(previuse_state.getLegalActions(agent.index))

        distsInv = [agent.getMazeDistance(myPos, a.getPosition()) for a in invaders]

        if not myPState.isPacman:
            if len(invaders) > 0:
                if scaredTimer > 0:

                    features['invaderDistance'] = (min(distsInv) + 1) / 100
                else:
                    features['invaderDistance'] = - (min(distsInv) + 1) / 100
                features['numInvaders'] = len(invaders)

            features['agent-food'] = len(foodDefend) / 10

        if (myPState.isPacman and not myState.isPacman) and myPos == myState.start.pos:
            features['agent-eaten'] = 1.0
        # compute the location of pacman after he takes the action
        x, y = state.getAgentPosition(agent.index)

        myPos = (x, y)

        foodList = food.asList()
        capsules = agent.getCapsules(state)
        foodNumb = len(foodList)
        pfoodList = pfood.asList()
        pfoodNumb = len(pfoodList)

        if agent.type == "Offense":
            features['num-food'] = -  foodNumb / 10

        x, y = myPos
        dx, dy = Actions.directionToVector(action)
        next_x, next_y = int(x + dx), int(y + dy)

        # count the number of ghosts 1-step away
        legal = len(Actions.getLegalNeighbors(myPos, walls))

        ghostsAround = [g for g in ghostDefenders if agent.getMazeDistance(g.getPosition(), myPos) <= 1]
        if len(ghostsAround) > 0:
            ghostAround = True
            legal -= len(ghostsAround)
            features["ghost-around"] = 2
        else:
            ghostAround = False

        invadersAround = [i for i in invaders if agent.getMazeDistance(i.getPosition(), myPos) <= 1]
        if len(invadersAround) > 0:
            invaderAround = True
            if myState.scaredTimer > 0 and not myState.isPacman:
                legal -= len(invadersAround)
        else:
            invaderAround = False

        # if myState.isPacman:
        #     features["legal"] = legal / 2

        if not ghostAround and food[next_x][next_y]:
            # print("JEO SAM MAMA BEZ DUHOVA")
            features["eats-food"] = 1.0

        # if myPState.isPacman and not myState.isPacman and agent.getMazeDistance(myPos,
        #                                                                         myState.start.pos) > 3 and agent.getMazeDistance(
        #     myPPos, myState.start.pos) > 3:
        #     features["returned"] = 2
        # print("PRESAO")

        eaten = pfoodNumb - foodNumb
        if eaten == 1:
            # print("JEO SAM MAMA")
            features["eats-food"] = 2

        food_dist = -1
        ghost_dist = -1

        if len(foodList) > 0 and agent.type == "Offense":
            minDistance = min([agent.getMazeDistance(myPos, food) for food in foodList])

            # features['distanceToFood'] = (50 - minDistance) / 10
            # features['distanceToFood'] = minDistance / 10
            # features['distanceToFood'] = 100/(minDistance + 1)**2

            # features['n-carrying'] = carry / 10

            # features['distanceToFood'] = (minDistance / 100) + (carry+1)**2
            # if carry > 5:
            #     features['distance-food'] = - (minDistance + 1) / 100
            # else:
            #     features['distance-food'] = (minDistance + 1) / 100

            features['distance-food'] = minDistance / 100
            # food_dist = minDistance / 100

            # print("DTF: {}".format(minDistance))
            if len(capsules) > 0:
                minDistanceCapsules = min([agent.getMazeDistance(myPos, c) for c in capsules])
                features['distance-capsule'] = (minDistanceCapsules + 1) / 100

        ghost_number = len(ghostDefenders)
        start_distance = 0

        if agent.type == "Offense":
            # features["distance-start"] = - (agent.getMazeDistance(myPos, myPState.start.pos)) * carry/ 1000
            if carry > 0:
                features["dist-start"] = carry * agent.getMazeDistance(myPos, myPState.start.pos) / 200

            if ghost_number > 0:
                minDistance = min([agent.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghostDefenders])

                # old
                # features['distanceToGhost'] = (50 - minDistance) / 10
                # features['distanceToGhost'] = 4 / (minDistance + 1)

                # features['distance-ghost'] = (minDistance + 1) / 100
                features['distance-ghost'] = (100 - minDistance) / 100
                # ghost_dist = (100 - minDistance) / 100
                # features['ghost-number'] = ghost_number
                # features['distanceToGhost'] = 1000 / (minDistance + 1)
                # print("DTG: {}".format(features["distanceToGhost"]))

                # print("DTI: {}".format(features["distanceToInvader"]))
                if len(ghostsScared) > 0:
                    minDistanceScared = min(
                        [agent.getMazeDistance(myPos, ghost.getPosition()) for ghost in ghostsScared])
                    features['distance-scared-ghost'] = (minDistanceScared + 1) / 100

        # features["maximus"] = max(food_dist, ghost_dist) + carry * start_distance

        # if agent.type == "Offense":
        #     print("F ", features)
        # print(food_dist, 1 / ghost_dist)

        return features
