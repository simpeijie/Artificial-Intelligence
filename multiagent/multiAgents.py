# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
from random import randint

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        # print legalMoves[chosenIndex]
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        ghostPositions = successorGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        distanceToFood = []
        distanceToGhost = []

        if not newFood:
          distanceToFood.append(1)
        for nf in newFood:
          distanceToFood.append(manhattanDistance(newPos, nf) + 1)

        for ghost in ghostPositions:
          distanceToGhost.append(manhattanDistance(newPos, ghost) + 1)

        return successorGameState.getScore() - len(newFood) + 0.6/min(distanceToFood) - 0.7/max(distanceToGhost) 

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.isWin():
            Returns whether or not the game state is a winning state

          gameState.isLose():
            Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***" 
        return self.value(gameState, 0)[1]

    def value(self, gameState, index):
      numAgents = gameState.getNumAgents()
      if gameState.isWin() or gameState.isLose() or index / numAgents == self.depth:
        return (self.evaluationFunction(gameState), None) 
      # pacman
      if index % numAgents == 0:
        return self.maxValue(gameState, index)
      else:
        return self.minValue(gameState, index)

    def maxValue(self, gameState, index):
      v = []
      legalActions = gameState.getLegalActions()
      for action in legalActions:
        v.append( (self.value(gameState.generateSuccessor(0, action), index+1)[0], action) )
      return max(v, key=lambda x:x[0])

    def minValue(self, gameState, index):
      v= []
      numAgents = gameState.getNumAgents()
      legalActions = gameState.getLegalActions(index % numAgents)
      for action in legalActions:
        v.append( (self.value(gameState.generateSuccessor(index % numAgents, action), index+1)[0], action) )
      return min(v, key=lambda x:x[0])

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, float('-inf'), float('inf'), 0)[1]
      
    def value(self, gameState, alpha, beta, index): 
      numAgents = gameState.getNumAgents()
      if gameState.isWin() or gameState.isLose() or index / numAgents == self.depth:
        return (self.evaluationFunction(gameState), None)
      if index % numAgents == 0:
        return self.max_value(gameState, alpha, beta, index)
      else:
        return self.min_value(gameState, alpha, beta, index)

    def max_value(self, gameState, alpha, beta, index):
      v = (float('-inf'), None)
      legalActions = gameState.getLegalActions()
      for action in legalActions: 
        v = max(v, (self.value(gameState.generateSuccessor(0, action), alpha, beta, index+1)[0], action), key=lambda x:x[0])
        if v[0] > beta:
          return v
        alpha = max(alpha, v[0])
      return v

    def min_value(self, gameState, alpha, beta, index):
      v = (float('inf'), None)
      numAgents = gameState.getNumAgents()
      legalActions = gameState.getLegalActions(index % numAgents)
      for action in legalActions:
        v = min(v, (self.value(gameState.generateSuccessor(index % numAgents, action), alpha, beta, index+1)[0], action), key=lambda x:x[0])
        if v[0] < alpha:
          return v
        beta = min(beta, v[0])
      return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.value(gameState, 0)[1]

    def value(self, gameState, index): 
      numAgents = gameState.getNumAgents()
      if gameState.isWin() or gameState.isLose() or index / numAgents == self.depth:
        return (self.evaluationFunction(gameState), None)
      if index % numAgents == 0:
        return self.max_value(gameState, index)
      else:
        return self.expected_value(gameState, index)

    def max_value(self, gameState, index):
      v = []
      legalActions = gameState.getLegalActions()
      for action in legalActions:
        v.append( (self.value(gameState.generateSuccessor(0, action), index+1)[0], action) )
      return max(v, key=lambda x:x[0])

    def expected_value(self, gameState, index):
      v = []
      numAgents = gameState.getNumAgents()
      legalActions = gameState.getLegalActions(index % numAgents)
      for action in legalActions:
        v.append( (self.value(gameState.generateSuccessor(index % numAgents, action), index+1)[0], action) )
      return (float(sum([pair[0] for pair in v])) / len(v), legalActions[randint(0, len(legalActions)-1)])

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: average distance between pacman and all the ghosts
    """
    "*** YOUR CODE HERE ***"
    pacmanPos = currentGameState.getPacmanPosition()
    ghostsPos = currentGameState.getGhostPositions()
    food = currentGameState.getFood().asList()
    scaredTimes = [g.scaredTimer for g in currentGameState.getGhostStates()][0]

    distanceToFood = []
    distanceToGhost = []
    distanceToCapsule = []

    if not food:
      distanceToFood.append(1)
    for nf in food:
      distanceToFood.append(manhattanDistance(pacmanPos, nf) + 1)
    for ghost in ghostsPos:
      distanceToGhost.append(manhattanDistance(pacmanPos, ghost) + 1)
    
    if scaredTimes != 0:
      return currentGameState.getScore() - len(food) + 0.6/min(distanceToFood) - 0.4/max(distanceToGhost)

    return currentGameState.getScore() - len(food) + 0.6/min(distanceToFood) - 0.7/max(distanceToGhost)


# Abbreviation
better = betterEvaluationFunction

