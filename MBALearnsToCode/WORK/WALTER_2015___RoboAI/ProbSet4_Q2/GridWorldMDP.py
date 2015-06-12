from __future__ import division
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib.pyplot as pyplot
import numpy as np
import Tkinter


class GridWorldMDP(object):
    # Construct an GridWorld representation of the form
    #
    #  20  21  22  23  24
    #  15   x  17  18  19 
    #  10   x  12   x  14
    #   5   6   7   8   9
    #   0   1   2   3   4
    #
    # The position marked with x is an obstacle. States 0, 1, 2, 3,
    # and 4 are absorbing states with negative reward (e.g., cliffs)
    # and states 12 and 14 are absorbing states with positive reward.
    #
    # with the following variables
    #
    #    T:       A 24x24x4 array where T[i,j,k] is the likelihood
    #             of transitioning from state i to state j when taking
    #             action A[k]
    #
    #    R:       A 24x24x4 array where R[i,j,k] expresses the
    #             reward received when going from state i to state j
    #             via action A[k]
    #
    #    A:       A list of actions A = [N=0, E=1, S=2, W=3]
    #
    #    noise:   The likelihood that the action is incorrect
    #
    #    gamma:   The discount factor
    
    def __init__(self, noise=0.2, gamma=0.9):

        self.gamma = gamma
        
        # The actions
        self.A = ['N','E','S','W']

        self.width = 5
        self.height = 5
        self.numstates = self.width*self.height
        self.absorbing_states = [0,1,2,3,4,12,14]
        obstacles = [11,13,16]
        
        
        
        # The transition matrix
        self.T = np.zeros([25,25,4])


        
        for i in self.absorbing_states:
            self.T[i,i,0] = 1;
            self.T[i,i,1] = 1;
            self.T[i,i,2] = 1;
            self.T[i,i,3] = 1;

            
        for i in obstacles:
            self.T[i,i,0] = 1;
            self.T[i,i,1] = 1;
            self.T[i,i,2] = 1;
            self.T[i,i,3] = 1;

            
        # Nominally set the transition likelihoods
        for i in range(0,self.width*self.height):

            # We've already taken care of the absorbing and obstacle states
            if i in self.absorbing_states:
                continue

            if i in obstacles:
                continue


            # Are we bounded above, below, left, or right by a
            # boundary or an obstacle
            btop = False
            bbottom = False
            bleft = False
            bright = False

            if (i >= (self.width*(self.height-1))) or (i+self.width in obstacles):
                btop = True

            if (i<self.width) or (i-self.width in obstacles):
                bbottom = True

            if ((i+1) % 5 == 0) or (i+1 in obstacles):
                bright = True

            if (i % 5 == 0) or (i-1 in obstacles):
                bleft = True
                    
            # North
            a = 0

            if btop:
                self.T[i,i,a] = 1 - noise;
            else:
                self.T[i,i+self.width,a] = 1 - noise;

            if bleft:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i-1,a] = noise/2

            if bright:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i+1,a] = noise/2




            # East
            a = 1

            if bright:
                self.T[i,i,a] = 1 - noise;
            else:
                self.T[i,i+1,a] = 1 - noise;

            if btop:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i+self.width,a] = noise/2

            if bbottom:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i-self.width,a] = noise/2                


            # South
            a = 2

            if bbottom:
                self.T[i,i,a] = 1 - noise;
            else:
                self.T[i,i-self.width,a] = 1 - noise;

            if bleft:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i-1,a] = noise/2

            if bright:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i+1,a] = noise/2


            # West
            a = 3

            if bleft:
                self.T[i,i,a] = 1 - noise;
            else:
                self.T[i,i-1,a] = 1 - noise;

            if btop:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i+self.width,a] = noise/2

            if bbottom:
                self.T[i,i,a] += noise/2
            else:
                self.T[i,i-self.width,a] = noise/2  
                
                
                



        # The rewards
        self.R = np.zeros([self.numstates,self.numstates,4])

        # Any move that transitions to the bottom row receives -10
        # reward
        for i in range(5,10):
            for a in range(0,4):
                self.R[i,i-self.width,a] = -10


        for i in [7,17]:
            for a in range(0,4):
                self.R[i,12,a] = 1.0

        for i in [9,19]:
            for a in range(0,4):
                self.R[i,14,a] = 10.0
                

    def drawWorld(self, V, Pi):

        fig = pyplot.figure()
        ax = fig.add_subplot(111)

        ax.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            labelbottom='off',
            labelleft='off')
        
        size = 10
        for i in range(self.height):
            ax.plot([0,self.width*size],[i*size,i*size],'k-')

        for i in range(self.width):
            ax.plot([i*size,i*size],[0,self.height*size],'k-')


        # Draw the obstacles
        verts = [
            (1.*size,2.*size),
            (1.*size,4.*size),
            (2.*size,4.*size),
            (2.*size,2.*size),
            (1.*size,2.*size),
        ]
            
        codes = [Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY,
         ]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='black', lw=1)
        ax.add_patch(patch)

        
        verts = [
            (1.*size,3.*size),
            (1.*size,4.*size),
            (2.*size,4.*size),
            (2.*size,3.*size),
            (1.*size,3.*size),
        ]

        path = Path(verts, codes)

        patch = patches.PathPatch(path, facecolor='black', lw=1)
        ax.add_patch(patch)


        verts = [
            (3.*size,1.*size),
            (3.*size,2.*size),
            (4.*size,2.*size),
            (4.*size,1.*size),
            (3.*size,1.*size),
        ]

        path = Path(verts, codes)


        for i in range(0,5):
            verts = [
                (i*size,0.*size),
                (i*size,1.*size),
                ((i+1)*size,1.*size),
                ((i+1)*size,0.*size),
                (i*size,0.*size),
            ]

            path = Path(verts, codes)
    
            patch = patches.PathPatch(path, facecolor='red', lw=1)
            ax.add_patch(patch)


        verts = [
            (3.*size,2.*size),
            (3.*size,3.*size),
            (4.*size,3.*size),
            (4.*size,2.*size),
            (3.*size,2.*size),
        ]

        path = Path(verts, codes)
        

        patch = patches.PathPatch(path, facecolor='black', lw=2)
        ax.add_patch(patch)


        # Draw the goal regions
        verts = [
            (2.*size,2.*size),
            (2.*size,3.*size),
            (3.*size,3.*size),
            (3.*size,2.*size),
            (2.*size,2.*size),
        ]

        path = Path(verts, codes)
        

        patch = patches.PathPatch(path, facecolor='green', lw=2)
        ax.add_patch(patch)


        verts = [
            (4.*size,2.*size),
            (4.*size,3.*size),
            (5.*size,3.*size),
            (5.*size,2.*size),
            (4.*size,2.*size),
        ]

        path = Path(verts, codes)
        

        patch = patches.PathPatch(path, facecolor='green', lw=2)
        ax.add_patch(patch)                        
        

        # Draw the value function
        for k in range(0,len(V)):
            j = np.floor(k/self.width)
            i = k - j*self.width

            v = '%.2f' % V[k]
                        
            ax.text((i+0.4)*size, (j+0.45)*size, v)

            if Pi[k] == 0:
                ax.arrow ((i+0.5)*size, (j+0.6)*size, 0.0, 0.3*size, head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 1:
                ax.arrow ((i+0.7)*size, (j+0.5)*size, 0.15*size, 0.0, head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 2:
                ax.arrow ((i+0.5)*size, (j+0.4)*size, 0.0, -0.3*size, head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

            if Pi[k] == 3:
                ax.arrow ((i+0.3)*size, (j+0.5)*size, -0.15*size, 0.0, head_width=0.1*size, head_length=0.1*size, fc='k', ec='k')

                   
        pyplot.show()
 
    def valueIteration(self, epsilon):
    # Perform value iteration with the following variables
    #
    # INPUT:
    #    epsilon:  The threshold for the stopping criterion
    #
    #         |Vnew - Vprev|_inf <= epsilon
    #
    #    where |x|_inf is the infinity norm (i.e., max(abs(V[i])) over all i)
    #
    #      gamma:  The discount factor
    #
    # OUTPUT:
    #          V: The value of each state encoded as a 12x1 array
    #         Pi: The action associated with each state (the policy) encoded as a 12x1 array

        # Your function should populate the following arrays
        V = np.zeros([self.numstates])  # Value function
        V[0:5] = -10
        V[12] = 1
        V[14] = 10
        non_terminal_state = np.ones([self.numstates])
        non_terminal_state[0:5] = 0.
        non_terminal_state[12] = 0.
        non_terminal_state[14] = 0.

        Pi = np.zeros([self.numstates]) # Policy where Pi[i] is 0 (N), 1 (E), 2 (S), 3(W)

        n = 0 # Keep track of the number of iterations

        change = np.array(self.numstates * [np.inf])
        while max(abs(change)) > epsilon:
            n += 1
            V_new = V.copy()
            for s in (5, 6, 7, 8, 9, 10, 15, 17, 18, 19, 20, 21, 22, 23, 24):
                V_array = self.gamma * np.sum(self.T[s, :, :] *
                                              (self.R[s, :, :] + np.atleast_2d(non_terminal_state * V).T), axis=0)
                m = -np.inf
                for a in range(4):
                    if V_array[a] > m:
                        m = V_array[a]
                        Pi[s] = a
                        V_new[s] = m
            change = V_new - V
            V = V_new

        return (V,Pi,n)