#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#    Epidemic Models - Calculates parameters of epidemic models
#    Copyright (C) 2020 Carlos Moreno
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
#    See LICENSE


"""
Main library


"""

#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

# Own Libs
from models import EPIDEMIC_MODELS
from integrators import runge_kutta_4
from GA.population import Population

#######################################################################


def main():
    """
    Main program to execute an optimisation in a population

    Parameters
    ----------

    Returns
    ----------

    """

    config = toml.load('../config/configuration.toml', _dict=dict)

    realDataDf = pd.read_csv('../data/IRD_Madrid.csv')
    realData = np.zeros((realDataDf.shape[0], 3))
    realData[:, 0] = realDataDf[realDataDf.columns[1]]
    realData[:, 1] = realDataDf[realDataDf.columns[2]]
    realData[:, 2] = realDataDf[realDataDf.columns[3]]
    config['population']['fitness_function']['realData'] = realData

    population = Population(config['population'], fitness_function)

    population.initialise_population()

    population.optimise()

    params = population.individuals[0].chromosome

    return params


def test(params):
    """
    Test program to execute an integration arc

    Parameters
    ----------
    params : np.ndarray (5) [β ε σ ρ μ]
        Parameters of the model. See README

    Returns
    ----------

    """

    config = toml.load('../config/configuration.toml', _dict=dict)[
        'population']['fitness_function']

    # Initial states [S E I R D]
    initialStates = config['initialStates']
    N = np.sum(initialStates)

    # Step in hours
    step = config['step']

    # Period in days
    T = config['period']

    statesAllPeriod, time = get_curves('SEIR', initialStates, params, T, step)

    # Cost function
    realDataDf = pd.read_csv('../data/IRD_Madrid.csv')
    realData = np.zeros((realDataDf.shape[0], 3))
    realData[:, 0] = realDataDf[realDataDf.columns[1]]
    realData[:, 1] = realDataDf[realDataDf.columns[2]]
    realData[:, 2] = realDataDf[realDataDf.columns[3]]

    config['params'] = params
    config['realData'] = realData
    cost = fitness_function(**config)

    print(cost)

    plt.figure()
    ax = plt.axes()
    # Set the limits to the axes
    ax.set_xlim(0, T)
    ax.set_ylim(0, N*1.01)

    # Set the title of the animation
    ax.set_title('Epidemic curves')

    # Set the label of the axes
    ax.set_xlabel('Time [day]')
    ax.set_ylabel('Number of people')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(24))

    # Plot the different
    ax.plot(time, statesAllPeriod[:, 0], color='blue', lw='2',
            label='Susceptible cases')
    ax.plot(time, statesAllPeriod[:, 1], color='yellow', lw='2',
            label='Exposed cases')
    ax.plot(time, statesAllPeriod[:, 2], color='red', lw='2',
            label='Infected cases')
    ax.plot(time, statesAllPeriod[:, 3], color='green', lw='2',
            label='Recovered cases')
    ax.plot(time, statesAllPeriod[:, 4], color='black', lw='2',
            label='Dead cases')

    ax.legend()

    return statesAllPeriod[::int(24/step)]


def get_curves(epidemicModel: str, initialStates: list, params: list,
               period: float, step: float) -> np.ndarray:
    """
    Function that obtain the integrated curves of states

    Parameters
    ----------
    epidemicModel : function
        Epidemic model

    initialStates : np.ndarray (5) [S E I R D]
        Initial states of the population

    params : np.ndarray (5) [β ε σ ρ μ]
        Parameters of the model. See README

    period : float [day]
        Duration of the integration

    step : float [h]
        Time steps of the integration

    Returns
    ----------
    statesAllPeriod : np.ndarray (Nx5) [S E I R D]
        Integration over the whole period

    time : np.ndarray (N) [d]
        Time of integrated states

    """
    # Epidemic model chosen
    model = EPIDEMIC_MODELS[epidemicModel]

    N = np.sum(initialStates)
    n = int(period * 24 / step)
    time = np.linspace(0, period*24, n+1)[:-1]/24

    statesAllPeriod = np.zeros((n, 5))

    statesAllPeriod[0, :] = initialStates

    for i in range(n-1):
        statesAllPeriod[i+1, :] = runge_kutta_4(model,
                                                N,
                                                statesAllPeriod[i, :],
                                                params,
                                                step)

    return statesAllPeriod, time


def fitness_function(epidemicModel: str, initialStates: list, params: list,
                     period: float, step: float, realData: np.ndarray
                     ) -> float:
    """
    Function that obtain the integrated curves of states

    Parameters
    ----------
    epidemicModel : function
        Epidemic model

    initialStates : np.ndarray (5) [S E I R D]
        Initial states of the population

    params : np.ndarray (5) [β ε σ ρ μ]
        Parameters of the model. See README

    period : float [day]
        Duration of the integration

    step : float [h]
        Time steps of the integration

    Returns
    ----------
    cost : float
        Evaluation of the cost function

    """
    simData, _ = get_curves(epidemicModel, initialStates, params, period, step)
    simDataIRD = simData[::int(24/step), 2:]

    cost = np.sqrt(np.mean((simDataIRD - realData)**2))

    return cost


if __name__ == "__main__":
    PARAMS = main()
    statesAllPeriod = test(PARAMS)
