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

from MBALearnsToCode.WORK.WALTER_2015___RoboAI.search import util


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
    from MBALearnsToCode.WORK.WALTER_2015___RoboAI.search.game import Directions
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
    """
    start_state = problem.getStartState()
    stack = util.Stack()
    stack.push((start_state,))
    actions___dict = {(start_state,): []}
    visited_states = {start_state}
    while not stack.isEmpty():
        path = stack.pop()
        state = path[-1]
        if problem.isGoalState(state):
            return actions___dict[path]
        else:
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    extended_path = path + (next_state,)
                    stack.push(extended_path)
                    actions___dict[extended_path] = actions___dict[path] + [action]
            del actions___dict[path]

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    start_state = problem.getStartState()
    queue = util.Queue()
    queue.push((start_state,))
    actions___dict = {(start_state,): []}
    visited_states = {start_state}
    while not queue.isEmpty():
        path = queue.pop()
        state = path[-1]
        if problem.isGoalState(state):
            return actions___dict[path]
        else:
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    extended_path = path + (next_state,)
                    queue.push(extended_path)
                    actions___dict[extended_path] = actions___dict[path] + [action]
            del actions___dict[path]


def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    priority_queue.push((start_state,), 0)
    actions___dict = {(start_state,): []}
    visited_states = {start_state}
    while not priority_queue.isEmpty():
        path = priority_queue.pop()
        state = path[-1]
        if problem.isGoalState(state):
            return actions___dict[tuple(path)]
        else:
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    extended_path = path + (next_state,)
                    actions___dict[extended_path] = actions___dict[path] + [action]
                    extended_cost = problem.getCostOfActions(actions___dict[extended_path])
                    priority_queue.push(extended_path, extended_cost)
            del actions___dict[path]


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    start_state = problem.getStartState()
    priority_queue = util.PriorityQueue()
    priority_queue.push((start_state,), 0)
    actions___dict = {(start_state,): []}
    visited_states = {start_state}
    while not priority_queue.isEmpty():
        path = priority_queue.pop()
        state = path[-1]
        if problem.isGoalState(state):
            return actions___dict[path]
        else:
            for successor in problem.getSuccessors(state):
                next_state, action = successor[:2]
                if next_state not in visited_states:
                    visited_states.add(next_state)
                    extended_path = path + (next_state,)
                    actions___dict[extended_path] = actions___dict[path] + [action]
                    a_star_cost = (problem.getCostOfActions(actions___dict[extended_path]) +
                                   heuristic(next_state, problem))
                    priority_queue.push(extended_path, a_star_cost)
            del actions___dict[path]


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
