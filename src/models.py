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
Epidemic models

"""

#######################################################################
# Imports area
#######################################################################

# Generic / Built-in


# Other Libs
import numpy as np


# Own Libs


#######################################################################

def seir_model(N: int, states: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    SEIR epidemic scheme

    Parameters
    ----------
    N : int
        Total population

    states : np.ndarray (5) [S E I R D]
        Different states of the population. See README

    params : np.ndarray (5) [β ε σ ρ μ]
        Parameters of the model. See README

    Returns
    ----------
    changeStates : np.ndarray (5) [S E I R D]
        New states

    """

    # To clarify the equations (there are other ways to do it faster)
    S, E, I, _, _ = states
    β, ε, σ, ρ, μ = params

    changeStates = np.array([- (β*I + ε*E) * S/N,
                             (β*I + ε*E) * S/N - σ*E,
                             σ*E - ρ*I - μ*I,
                             ρ*I,
                             μ*I
                             ])

    return changeStates


EPIDEMIC_MODELS = {'SEIR': seir_model}