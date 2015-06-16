# search.py
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())

    The non-recursive implementation is similar to breadth-first search but differs from it in two ways: it uses a stack instead of a queue, and it delays checking whether a vertex has been discovered until the vertex is popped from the stack rather than making this check before pushing the vertex.
    """

    start_state = problem.getStartState()
    stack = util.Stack()
    stack.push(start_state)
    actions = {start_state: []}
    visited_states = set()
    while not stack.isEmpty():
        state = stack.pop()
        if state not in visited_states:
            visited_states.add(state)
            if problem.isGoalState(state):
                return actions[state]
            else:
                for successor in problem.getSuccessors(state):
                    next_state, action = successor[:2]
                    stack.push(next_state)
                    actions[next_state] = actions[state] + [action]
                del actions[state]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()
    queue = util.Queue()
    queue.push(start_state)
    actions = {start_state: []}
    visited_states = {start_state}
    while not queue.isEmpty():
        state = queue.pop()
        if problem.isGoalState(state):
            return actions[state]
        else:
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    queue.push(next_state)
                    actions[next_state] = actions[state] + [action]
            del actions[state]

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return aStarSearch(problem)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    def cheaper(actions_0, actions_1):
        if problem.getCostOfActions(actions_1) < problem.getCostOfActions(actions_0):
            return actions_1
        return actions_0

    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    priority_queue.push(start_state, heuristic(start_state, problem))
    actions = {start_state: []}
    visited_states = set()
    while not priority_queue.isEmpty():
        state = priority_queue.pop()
        if problem.isGoalState(state):
            return actions[state]
        elif state not in visited_states:
            visited_states.add(state)
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                greedy_actions = actions[state] + [action]
                if next_state in actions:
                    actions[next_state] = cheaper(actions[next_state], greedy_actions)
                else:
                    actions[next_state] = greedy_actions
                a_star_cost = problem.getCostOfActions(actions[next_state]) + heuristic(next_state, problem)
                priority_queue.push(next_state, a_star_cost)

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
