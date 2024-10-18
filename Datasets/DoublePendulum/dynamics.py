"""
    Copyright (C) 2023, Christopher Paul Ley
    Asynchronous Graph Generator

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from functools import partial

import numpy as np
from scipy.integrate import solve_ivp


def double_pendulum_ODE(
    t: float,
    y: np.ndarray,
    dynamics_params: dict = None,
):
    if dynamics_params is None:
        dynamics_params = {
            "l_1": 1,
            "l_2": 1,
            "m_1": 1,
            "m_2": 1,
            "k_1": 0.05,
            "g": 9.81,
        }
    l_1 = dynamics_params["l_1"]
    l_2 = dynamics_params["l_2"]
    m_1 = dynamics_params["m_1"]
    m_2 = dynamics_params["m_2"]
    k_1 = dynamics_params["k_1"]
    g = dynamics_params["g"]

    f = np.array([0.0, 0.0, 0.0, 0.0])
    O_1 = y[0]
    O_1d = y[1]
    O_2 = y[2]
    O_2d = y[3]
    f[0] = O_1d
    f[1] = (
        -(O_1d**2) * l_1**2 * m_2 * np.sin(2 * O_1 - 2 * O_2) / 2
        - O_1d * k_1
        - O_2d**2 * l_1 * l_2 * m_2 * np.sin(O_1 - O_2)
        - g * l_1 * m_1 * np.sin(O_1)
        - g * l_1 * m_2 * np.sin(O_1) / 2
        - g * l_1 * m_2 * np.sin(O_1 - 2 * O_2) / 2
    ) / (l_1**2 * (m_1 - m_2 * np.cos(O_1 - O_2) ** 2 + m_2))
    f[2] = O_2d
    f[3] = (
        O_1d**2 * l_1**2 * m_1 * np.sin(O_1 - O_2)
        + O_1d**2 * l_1**2 * m_2 * np.sin(O_1 - O_2)
        + O_1d * k_1 * np.cos(O_1 - O_2)
        + O_2d**2 * l_1 * l_2 * m_2 * np.sin(2 * O_1 - 2 * O_2) / 2
        - g * l_1 * m_1 * np.sin(O_2) / 2
        + g * l_1 * m_1 * np.sin(2 * O_1 - O_2) / 2
        - g * l_1 * m_2 * np.sin(O_2) / 2
        + g * l_1 * m_2 * np.sin(2 * O_1 - O_2) / 2
    ) / (l_1 * l_2 * (m_1 - m_2 * np.cos(O_1 - O_2) ** 2 + m_2))
    return f


def double_pendulum_polar_to_cartesian(
    theta_1: np.ndarray, theta_2: np.ndarray, l_1: float, l_2: float
):
    x_1 = l_1 * np.sin(theta_1)
    y_1 = -l_1 * np.cos(theta_1)
    x_2 = x_1 + l_2 * np.sin(theta_2)
    y_2 = y_1 - l_2 * np.cos(theta_2)
    return x_1, y_1, x_2, y_2


def generate_double_pendulum_data(
    initial_conditions: np.ndarray,
    stop_at: int,
    dynamics_params: dict,
    sampling_rate: float = 1 / 30,
):
    parameterised_ODE = partial(double_pendulum_ODE, dynamics_params=dynamics_params)
    solution = solve_ivp(
        parameterised_ODE,
        [0, stop_at * sampling_rate],
        initial_conditions,
        t_eval=np.arange(0, stop_at * sampling_rate, sampling_rate),
    )
    theta_1, theta_1_dot, theta_2, theta_2_dot = solution.y
    x_1, y_1, x_2, y_2 = double_pendulum_polar_to_cartesian(
        theta_1, theta_2, dynamics_params["l_1"], dynamics_params["l_2"]
    )
    return np.stack([x_1, y_1, x_2, y_2], axis=1), solution.t


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    initial_conditions = np.array([np.pi / 2, 0, np.pi / 2, 0])
    stop_at = 100
    sampling_rate = 1 / 30
    dyn_params = {
        "l_1": 1,
        "l_2": 1,
        "m_1": 1,
        "m_2": 1,
        "k_1": 0.05,
        "g": 9.81,
    }
    data, time = generate_double_pendulum_data(
        initial_conditions, stop_at, dyn_params, sampling_rate
    )
    plt.plot(time, data[:, 0], label="x_1")
    plt.plot(time, data[:, 1], label="y_1")
    plt.plot(time, data[:, 2], label="x_2")
    plt.plot(time, data[:, 3], label="y_2")
    plt.legend()
    plt.show()
