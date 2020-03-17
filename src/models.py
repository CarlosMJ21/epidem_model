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
        New

    """

    # To clarify the equations (there are other ways to do it faster)
    S, E, I, R, D = states
    β, ε, σ, ρ, μ = params

    changeStates = np.ndarray([- (β*I + ε*E) * S/N,
                               (β*I + ε*E) * S/N - σ*E,
                               σ*E - ρ*I - μ*I,
                               ρ*I,
                               μ*I
                               ])

    return changeStates
