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

# Own Libs
from models import seir_model
from integrators import runge_kutta_4

#######################################################################


def main():
    """
    Main program to execute an integration arc

    Parameters
    ----------

    Returns
    ----------

    """

    # Initial states [S E I R D]
    initialStates = [19995., 0., 5., 0., 0.]
    N = np.sum(initialStates)

    # Parameters [β ε σ ρ μ]
    params = [1/96, 1/48, 1/(7*24), 1/(14*24), 0.]

    # Step in hours
    step = 1.

    # Period in days
    T = 30.

    n = int(T * 24 / step)
    time = np.linspace(0, T*24, n+1)[:-1]/24

    statesAllPeriod = np.zeros((n, 5))

    statesAllPeriod[0, :] = initialStates

    for i in range(n-1):
        statesAllPeriod[i+1, :] = runge_kutta_4(seir_model,
                                                N,
                                                statesAllPeriod[i, :],
                                                params,
                                                step)

    plt.figure()
    ax = plt.axes()
    # Set the limits to the axes
    ax.set_xlim(0, T)
    ax.set_ylim(0, N*1.01)

    # Set the title of the animation
    ax.set_title('Epidemic curves')

    # Set the label of the axes
    ax.set_xlabel('Time [h]')
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


if __name__ == "__main__":
    main()
