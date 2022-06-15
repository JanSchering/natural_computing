import torch as t

# ---------- Adjusted from https://github.com/rasmusbergpalm/vnca/blob/main/modules/residual.py

class Residual(t.nn.Module):
    """
    Residual Network layer.
    """
    def __init__(self, *args: t.nn.Module):
        super().__init__()
        self.delegate = t.nn.Sequential(*args)

    def forward(self, inputs:t.Tensor) -> t.Tensor:
        """
        Forward the input through the next layer and add the input (residual) on top of the result.
        inputs (t.Tensor): Network input.
        """
        return self.delegate(inputs) + inputs
