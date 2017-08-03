# Multi-Agent

This project includes designed agents (Pacman and ghosts) for the classic version of Pacman as well as implement both minimax with alpha-beta pruning and expectimax search. The point of this project is to get Pacman to reach his goals in scenarios where there are more than one adversary.

## Implementation

### Minimax Agent

Pacman is the maximizing agent, whereas the ghosts are the minimizing agents. At each step, Pacman aims to maximize its value by calling `MinimaxAgent.maxValue()` for all the actions it can legally take from a particular game state. The same applies to the ghosts which calls `MinimaxAgent.minValue()`. 

When Pacman believes that his death is unavoidable, he will try to end the game as soon as possible because of the constant penalty for living. Sometimes, this is the wrong thing to do with random ghosts, but minimax agents always assume the worst. 

### Alpha-Beta Agent

An alpha beta agent prunes unnecessary actions to more efficiently explore the minimax tree. 

### Expectimax Agent

Expectimax attempts to rectify the pessimistic behavior of a minimax agent in which the agent always assumes that it's playing against an adversary who makes optimal decisions. Essentially, an expectimax agent is useful for modeling probabilistic behavior of agents who may make suboptimal choices.

An expectimax agent is able to win the game about 50% of the time compared to a minimax agent since Pacman assumes that the ghosts take a random action at each step and that they choose amongst their `getLegalActions` uniformly at random.
