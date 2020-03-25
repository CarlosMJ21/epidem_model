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
from individual import Individual


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
        optimiseDict = {'maximise': -1,
                        'minimise': 1
                        }
        m = optimiseDict[self.config['optimisation']]

        scores = self._scores()
        ranking = np.argsort(scores)[::m]
        newGeneration = []
        newGenAp = newGeneration.append

        for i, _ in enumerate(ranking):
            if i <= self.numInd / 2:
                child1, child2 = \
                    self.individuals[ranking[i]].offspring(
                        self.individuals[ranking[i+1]])

                newGenAp(child1)
                newGenAp(child2)

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

        scores = [individual.fitness_function() for individual in
                  self.individuals]

        return scores
