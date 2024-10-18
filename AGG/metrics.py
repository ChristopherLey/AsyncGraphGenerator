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
from typing import Any
from typing import Optional
from typing import Sequence
from typing import Union

import torch
from torch import Tensor
from torch import tensor
from torchmetrics import Metric
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute
from torchmetrics.functional.regression.mae import _mean_absolute_error_update
from torchmetrics.utilities.plot import _AX_TYPE
from torchmetrics.utilities.plot import _PLOT_OUT_TYPE


class MeanRelativeError(Metric):
    r"""`Compute Mean Relative Error`_ (MRE).

    .. math:: \text{MRE} = \frac{\sum_i^N | y_i - \hat{y_i} |}{\sum_i^N | y_i |}

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a tensor of predictions.

    As input to ``forward`` and ``update`` the metric accepts the following input:

    - ``preds`` (:class:`~torch.Tensor`): Predictions from model
    - ``target`` (:class:`~torch.Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following output:

    - ``mean_relative_error`` (:class:`~torch.Tensor`): A tensor with the mean relative error over the state

    Args:
        kwargs: Additional keyword arguments, see :ref:`Metric kwargs` for more info.

    """
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False

    sum_abs_error: Tensor
    sum_abs_targets: Tensor

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.add_state("sum_abs_error", default=tensor(0.0), dist_reduce_fx="sum")
        self.add_state("sum_abs_targets", default=tensor(1e-6), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        sum_abs_error, _ = _mean_absolute_error_update(preds, target)
        sum_abs_target, _ = _mean_absolute_error_update(
            torch.zeros_like(target), target
        )

        self.sum_abs_error += sum_abs_error
        self.sum_abs_targets += sum_abs_target

    def compute(self) -> Tensor:
        """Compute mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.sum_abs_targets)

    def plot(
        self,
        val: Optional[Union[Tensor, Sequence[Tensor]]] = None,
        ax: Optional[_AX_TYPE] = None,
    ) -> _PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

        """
        return self._plot(val, ax)
