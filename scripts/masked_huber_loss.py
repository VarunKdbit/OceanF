"""
OceanEmbed Masked Huber Loss
============================

Loss function for subsurface ocean temperature reconstruction.

Purpose
-------
Calculate Huber loss only over valid GLORYS target cells.

Why masking is required
-----------------------
The GLORYS target contains missing values over land / unavailable
ocean cells.

The OceanEmbed Dataset intentionally keeps those target values as NaN
and provides a corresponding binary target-validity mask.

Therefore:

    target_mask = 1
        -> include pixel in loss

    target_mask = 0
        -> exclude pixel from loss

The loss never uses invalid target cells.

Expected tensors
----------------
prediction:
    [B, 15, 32, 32]

target:
    [B, 15, 32, 32]

target_mask:
    [B, 15, 32, 32]

Output:
    scalar loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# MASKED HUBER LOSS
# ============================================================================


class MaskedHuberLoss(nn.Module):
    """
    Huber loss calculated only on valid target cells.

    The Huber loss is:

        0.5 * error²                    if |error| <= delta

        delta * (|error| - 0.5 * delta)
                                      otherwise

    where:

        error = prediction - target

    The final loss is the mean over valid cells only.
    """

    def __init__(
        self,
        delta: float = 1.0,
    ):
        super().__init__()

        if delta <= 0:

            raise ValueError(
                f"Huber delta must be > 0. "
                f"Received: {delta}"
            )

        self.delta = float(delta)

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate masked Huber loss.

        Parameters
        ----------
        prediction:
            Model prediction.

            Shape:
                [B, 15, 32, 32]

        target:
            Normalized GLORYS temperature.

            Shape:
                [B, 15, 32, 32]

            Invalid cells may contain NaN.

        target_mask:
            Binary validity mask.

            Shape:
                [B, 15, 32, 32]

            1 = valid
            0 = invalid

        Returns
        -------
        torch.Tensor
            Scalar masked Huber loss.
        """

        # --------------------------------------------------------------
        # Shape validation.
        # --------------------------------------------------------------

        if prediction.shape != target.shape:

            raise ValueError(
                "Prediction and target shapes must match.\n"
                f"Prediction: {tuple(prediction.shape)}\n"
                f"Target:     {tuple(target.shape)}"
            )

        if prediction.shape != target_mask.shape:

            raise ValueError(
                "Prediction and target_mask shapes must match.\n"
                f"Prediction:  {tuple(prediction.shape)}\n"
                f"Target mask: {tuple(target_mask.shape)}"
            )

        # --------------------------------------------------------------
        # Prediction must be finite.
        # --------------------------------------------------------------

        if not torch.isfinite(prediction).all():

            raise ValueError(
                "Prediction contains NaN or Inf values."
            )

        # --------------------------------------------------------------
        # Convert mask to boolean.
        # --------------------------------------------------------------

        valid_mask = target_mask > 0

        # --------------------------------------------------------------
        # If there are no valid target cells, the loss cannot be
        # calculated.
        # --------------------------------------------------------------

        valid_count = valid_mask.sum()

        if valid_count.item() == 0:

            raise ValueError(
                "No valid target cells were provided."
            )

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # We must never perform arithmetic involving invalid NaN
        # target values.
        #
        # Select only valid cells BEFORE calculating the error.
        # --------------------------------------------------------------

        prediction_valid = prediction[
            valid_mask
        ]

        target_valid = target[
            valid_mask
        ]

        # --------------------------------------------------------------
        # Verify valid target values.
        # --------------------------------------------------------------

        if not torch.isfinite(
            target_valid
        ).all():

            raise ValueError(
                "Valid target cells contain NaN or Inf."
            )

        # --------------------------------------------------------------
        # Error.
        # --------------------------------------------------------------

        error = (
            prediction_valid
            - target_valid
        )

        absolute_error = torch.abs(error)

        # --------------------------------------------------------------
        # Huber loss.
        # --------------------------------------------------------------

        quadratic = torch.minimum(
            absolute_error,
            torch.tensor(
                self.delta,
                dtype=absolute_error.dtype,
                device=absolute_error.device,
            ),
        )

        linear = (
            absolute_error
            - quadratic
        )

        loss = (
            0.5 * quadratic ** 2
            + self.delta * linear
        )

        # --------------------------------------------------------------
        # Mean over valid target cells only.
        # --------------------------------------------------------------

        return loss.mean()


# ============================================================================
# SIMPLE FUNCTION API
# ============================================================================


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    delta: float = 1.0,
) -> torch.Tensor:
    """
    Functional interface for MaskedHuberLoss.
    """

    criterion = MaskedHuberLoss(
        delta=delta
    )

    return criterion(
        prediction,
        target,
        target_mask,
    )


# ============================================================================
# LOSS SELF-TEST
# ============================================================================


def loss_self_test():
    """
    Verify the loss implementation with controlled tensors.

    This test checks:

        1. Loss is finite.
        2. Invalid NaN target values do not contaminate the loss.
        3. Perfect predictions produce zero loss.
        4. Gradients can flow through the loss.
    """

    print()
    print("=" * 72)
    print("OceanEmbed MASKED HUBER LOSS SELF-TEST")
    print("=" * 72)

    # --------------------------------------------------------------
    # Small controlled example.
    # --------------------------------------------------------------

    prediction = torch.tensor(
        [
            [
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [float("nan"), 5.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    target_mask = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    criterion = MaskedHuberLoss(
        delta=1.0
    )

    loss = criterion(
        prediction,
        target,
        target_mask,
    )

    print()
    print(
        f"Controlled loss: {float(loss):.6f}"
    )

    # --------------------------------------------------------------
    # Loss must be finite.
    # --------------------------------------------------------------

    assert torch.isfinite(loss)

    print(
        "PASS: loss is finite."
    )

    # --------------------------------------------------------------
    # Expected valid errors:
    #
    # prediction:
    #     1, 2, 4
    #
    # target:
    #     1, 1, 5
    #
    # errors:
    #     0, 1, -1
    #
    # Huber(delta=1):
    #     0, 0.5, 0.5
    #
    # Mean:
    #     1 / 3
    # --------------------------------------------------------------

    expected_loss = 1.0 / 3.0

    assert torch.isclose(
        loss,
        torch.tensor(
            expected_loss,
            dtype=torch.float32,
        ),
        atol=1e-6,
    )

    print(
        "PASS: controlled Huber calculation is correct."
    )

    # --------------------------------------------------------------
    # Perfect prediction test.
    # --------------------------------------------------------------

    perfect_prediction = torch.tensor(
        [
            [
                [
                    [10.0, 20.0],
                    [30.0, 40.0],
                ]
            ]
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    perfect_target = perfect_prediction.detach().clone()

    perfect_target[
        0,
        0,
        1,
        0,
    ] = float("nan")

    perfect_mask = torch.tensor(
        [
            [
                [
                    [1.0, 1.0],
                    [0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    perfect_loss = criterion(
        perfect_prediction,
        perfect_target,
        perfect_mask,
    )

    print()
    print(
        f"Perfect-prediction loss: "
        f"{float(perfect_loss):.6f}"
    )

    assert torch.isclose(
        perfect_loss,
        torch.tensor(
            0.0,
            dtype=torch.float32,
        ),
        atol=1e-7,
    )

    print(
        "PASS: perfect predictions produce zero loss."
    )

    # --------------------------------------------------------------
    # Backpropagation test.
    # --------------------------------------------------------------

    loss.backward()

    assert (
        prediction.grad is not None
    )

    assert torch.isfinite(
        prediction.grad
    ).all()

    print(
        "PASS: gradients flow through masked Huber loss."
    )

    # --------------------------------------------------------------
    # Final result.
    # --------------------------------------------------------------

    print()
    print("=" * 72)
    print("MASKED HUBER LOSS SELF-TEST PASSED")
    print("=" * 72)
    print()


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    loss_self_test()