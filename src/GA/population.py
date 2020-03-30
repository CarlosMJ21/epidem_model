# MIT License
#
# Copyright (c) 2020 Carlos Moreno
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Adaptation of the repository:
# https://github.com/CarlosMJ21/GA
#
#

"""
Population class

"""

#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import numpy as np


# Own Libs
from GA.individual import Individual


#######################################################################


class Population():
    """
    Class to represent a population.

    Attributes
    ----------
    config : dict
        Configuration of the population

    fitnessFunc : function
        Fitness function associated to individual

    individuals : list [~src.ga.individual]
        Individual of a population

    numInd : int
        Number of individuals in the population


    Methods
    ----------
    initialise_population()
        Initialise the population given certain values at the config

    mutation()
        Calculates the mutation of the individuals

    new_generation()
        Computes the new generation of individuals

    optimise()
        Optimise the population to the fitness problem


    """
    def __init__(self, config: dict, fitnessFunc, individuals=None):
        """
        Constructor of a generic population.

        Parameters
        ----------
        config : dict
            Configuration of the population

        fitnessFunc : function
            Fitness function associated to individual

        individuals : list [~src.ga.individual]
            Individual of a population

        Returns
        ----------

        """
        self.config = config
        self.fitnessFunc = fitnessFunc
        self.individuals = individuals
        self.numInd = None

    def initialise_population(self):
        """
        Initialise the individuals of a population.

        Parameters
        ----------

        Returns
        ----------

        """
        config = self.config
        self.individuals = []
        indAp = self.individuals.append

        for _ in range(config['size_population']):
            chromosome = np.random.rand(config['num_genes']) \
                * (config['max_values'] - config['min_values']) \
                + config['min_values']

            indAp(Individual(self.fitnessFunc,
                             config['crossover'],
                             config['mutation'],
                             chromosome))

        self.numInd = len(self.individuals)

    def mutation(self, probMutation):
        """
        Computes the mutation over the entire population.

        Parameters
        ----------
        probMutation : float
            Probability of mutation of one individual

        Returns
        ----------

        """

        for i in range(self.numInd):
            if np.random.rand() < probMutation:
                self.individuals[i].mutate(self.config['pressure'])

    def new_generation(self):
        """
        Computes the new generation of the population.

        Parameters
        ----------

        Returns
        ----------

        """
        optimiseDict = {'maximise': 1,
                        'minimise': -1
                        }
        m = optimiseDict[self.config['optimisation']]

        scores = np.array(self._scores())**m
        newGeneration = []
        newGenAp = newGeneration.append

        for _ in range(self.numInd):
            indices = self._select_individuals(scores)

            child1 = \
                self.individuals[indices[0]].offspring(
                    self.individuals[indices[1]])

            newGenAp(child1)

        newGeneration = newGeneration[:self.numInd]
        self.individuals = newGeneration

    def optimise(self):
        """
        Optimise the problem.

        Parameters
        ----------

        Returns
        ----------

        """
        for _ in range(self.config['num_generations']):
            self.new_generation()

            self.mutation(self.config['prob_mutation'])

    def _select_individuals(self, scores: np.ndarray) -> np.ndarray:
        """
        Select the individuals to breed

        Parameters
        ----------
        scores : np.ndarray (self.numInd) [float]
            Fitting scores of the individuals

        Returns
        ----------
        selectedIndices : np.ndarray [int]
            Indices selected to be parents


        """
        selectedIndices = np.repeat(None, 2)
        for i in np.arange(2):

            # Two pairs of parents are chosen
            firstCandidate = np.random.choice(a=np.arange(self.numInd),
                                              size=2,
                                              replace=False)
            secondCandidate = np.random.choice(a=np.arange(self.numInd),
                                               size=2,
                                               replace=False)

            # The one with better score is taken from each pair
            if scores[firstCandidate[0]] > scores[firstCandidate[1]]:
                firstChosen = firstCandidate[0]
            else:
                firstChosen = firstCandidate[1]

            if scores[secondCandidate[0]] > scores[secondCandidate[1]]:
                secondChosen = secondCandidate[0]
            else:
                secondChosen = secondCandidate[1]

            # The two winners are compared
            if scores[firstChosen] > scores[secondChosen]:
                finalIndex = firstChosen
            else:
                finalIndex = secondChosen

            selectedIndices[i] = finalIndex

        return selectedIndices

    def _scores(self):
        """
        Computes the score for each individual chromosome against the
        fitness function

        Parameters
        ----------

        Returns
        ----------
        scores : list [float]


        """

        scores = [individual.fitness_function(self.config['fitness_function'])
                  for individual in self.individuals]

        return scores
