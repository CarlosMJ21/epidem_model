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
Individual class

"""

#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import numpy as np


# Own Libs


#######################################################################


class Individual():
    """
    Class to represent an individual.

    Attributes
    ----------
        chromosome : np.ndarray (N) [float]
            Genes array of the individual

        crossover : str
            Type of crossover between individuals

        fitnessFunc :
            Fitness function associated to individual

        mutation : str
            Type of mutation of the individual

        numGenes : int
            Number of genes in the chromosome


    Methods
    ----------
    fitness_function()
        Return the evaluation of the individual genes on the fitness
        function

    offspring(secondParent)
        Return the offspring of two individuals

    mutate()
        Create a mutation in a random number of genes

    """

    def __init__(self, fitnessFunc, crossover: str, mutation: str,
                 chromosome: np.ndarray):

        """
        Constructor of a generic individual.

        Parameters
        ----------
        chromosome : np.ndarray (N) [float]
            Genes array of the individual

        crossover : str
            Type of crossover between individuals

        fitnessFunc : function
            Fitness function associated to individual

        mutation : str
            Type of mutation of the individual



        Returns
        ----------

        """
        self.chromosome = chromosome
        self.crossover = crossover
        self.fitnessFunc = fitnessFunc
        self.mutation = mutation
        self.numGenes = len(chromosome)

    def fitness_function(self, config: dict):
        """
        Performs the evaluation of the fitness function.


        Parameters
        ----------
        config : dict
            Configuration with the parameters of the fitness_function

        Returns
        -------

        """
        config['params'] = self.chromosome

        return self.fitnessFunc(**config)

    def offspring(self, secondParent):
        """
        Computates the offspring of two individuals.


        Parameters
        ----------
        secondParent : ~individual.Individual
            Second individual implied in the crossover

        Returns
        -------

        """
        # Select the crossover method for the offspring
        offspringDict = {'one_point': self._crossover_one_point,
                         'multiple_points': None}

        return offspringDict[self.crossover](secondParent)

    def mutate(self, pressure) -> None:
        """
        Computates the mutation of the individual's chromosome.


        Parameters
        ----------
        pressure : float
            Percentage of genes to mutate

        Returns
        -------

        """

        mutationDict = {'normal': self._mutation_normal,
                        'uniform': None}

        mutationDict[self.mutation](pressure)

    def _crossover_one_point(self, secondParent):
        """
        Computates the offspring of two individuals.


        Parameters
        ----------
        secondParent : ~individual.Individual
            Second individual implied in the crossover

        Returns
        -------
        child1 : ~individual.Individual
            First individual of the offspring

        child2 : ~individual.Individual
            Second individual of the offspring

        """
        N = self.numGenes
        # Termination point
        tP = int(np.random.rand() * N)

        chromosome1 = np.zeros(N)
        chromosome2 = np.zeros(N)

        # Creates the chromosomes intersecting both parent genes
        chromosome1[:tP] = self.chromosome[:tP]
        chromosome1[tP:] = secondParent.chromosome[tP:]

        chromosome2[tP:] = self.chromosome[tP:]
        chromosome2[:tP] = secondParent.chromosome[:tP]

        # Creates the child
        child1 = self.__class__(self.fitnessFunc, self.crossover,
                                self.mutation, chromosome1)

        child2 = self.__class__(secondParent.fitnessFunc,
                                secondParent.crossover,
                                secondParent.mutation, chromosome2)

        return child1, child2

    def _mutation_normal(self, pressure) -> None:
        """
        Create mutations on the individual.


        Parameters
        ----------
        pressure : float
            Percentage of genes to mutate

        Returns
        -------

        """
        nMutatedGenes = int(pressure * self.numGenes)

        # Select the mutated genes
        mutatedGenes = np.random.randint(0, self.numGenes, nMutatedGenes)
        mutatedGenes = list(set(mutatedGenes))

        # Add a normal error over the mutated genes
        for gene in mutatedGenes:
            self.chromosome[gene] += np.random.normal(self.chromosome[gene],
                                                      np.std(self.chromosome))
