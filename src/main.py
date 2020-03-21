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
    params = [1/12, 1/24, 1/(7*24), 1/(14*24), 0.]

    # Step in hours
    step = 1.

    # Period in days
    T = 60.

    n = int(T * 24 / step)

    statesAllPeriod = np.zeros((n, 5))

    statesAllPeriod[0, :] = initialStates

    for i in range(n-1):
        statesAllPeriod[i+1, :] = runge_kutta_4(seir_model,
                                                N,
                                                statesAllPeriod[i, :],
                                                params,
                                                step)





if __name__ == "__main__":
    main()

