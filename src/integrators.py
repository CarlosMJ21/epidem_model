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
    statesNext = states + 1/6 * step * (states1 + 2*states2 + 2*states3 \
        + states4)

    # Return next state
    return statesNext