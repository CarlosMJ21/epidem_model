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
Integrators library

"""

#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import numpy as np


# Own Libs


#######################################################################


def runge_kutta_4(epidemicModel, N: int, states: np.ndarray,
                  params: np.ndarray, step: float) -> np.ndarray:
    """
    Method of Runge-Kutta 4

    Parameters
    ----------
    epidemicModel : function
        Epidemic model

    states : np.ndarray (5) [S E I R D]
        Different states of the population. See README

    params : np.ndarray (5) [β ε σ ρ μ]
        Parameters of the model. See README

    step : float [sec]
        Time step implemented

    Returns
    ----------
    statesNext : np.ndarray (5) [S E I R D]
        Next temporal states, as a result of the numerical integration

    """

    # Evaluate function with current position and velocity
    states1 = epidemicModel(N, states, params)
    states2 = epidemicModel(N, states + 1/2 * step * states1, params)
    states3 = epidemicModel(N, states + 1/2 * step * states2, params)
    states4 = epidemicModel(N, states + 1/2 * step * states3, params)

    # Calculate next position and velocity
    statesNext = states + 1/6 * step * (states1 + 2*states2 + 2*states3
                                        + states4)

    # Return next state
    return statesNext
