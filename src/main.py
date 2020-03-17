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
    params = [1/12, 1/24, 1/(5*24), 1/(7*24), 0.]

    # Step in hours
    step = 1.

    # Period in days
    T = 30.

    n = int(T * 24 / step)

    statesAllPeriod = np.zeros((n, 5))

    statesAllPeriod[0, :] = initialStates

    for i in range(n-1):
        statesAllPeriod[i+1, :] = runge_kutta_4(seir_model,
                                                N,
                                                statesAllPeriod[i, :],
                                                params,
                                                step)

    print('hola')


if __name__ == "__main__":
    main()

