"""
OceanEmbed-CNN
==============

OceanEmbed SIH 2026 Project

Purpose
-------
Reconstruct subsurface ocean temperature from a 7-day retrospective
window of satellite-derived surface observations.

Scientific input contract
-------------------------
7 surface variables:
    1. SST
    2. SSS
    3. SLA
    4. U current
    5. V current
    6. U wind
    7. V wind

7-day retrospective window:
    7 days × 7 variables = 49 input channels

Spatial input:
    64 × 64

Spatial output:
    32 × 32

Target:
    15 subsurface temperature depth channels

Target depths:
    0, 5, 10, 20, 30, 50, 75, 100, 125, 150,
    200, 300, 500, 700, 1000 m

Model contract
--------------
Input:
    [B, 49, 64, 64]

Output:
    [B, 15, 32, 32]

The network uses a CNN encoder to transform the 64×64 surface
representation into a 128-channel latent representation at 32×32,
followed by a decoder/head that reconstructs temperature at the
15 target depths.

Important
---------
This file contains ONLY the model architecture.

Training, loss, optimizer, validation and checkpointing are handled
by separate files.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================================
# MODEL CONSTANTS
# ============================================================================

INPUT_CHANNELS = 49
LATENT_CHANNELS = 128
OUTPUT_CHANNELS = 15

INPUT_SIZE = 64
OUTPUT_SIZE = 32


# ============================================================================
# RESIDUAL BLOCK
# ============================================================================


class ResidualBlock(nn.Module):
    """
    Basic convolutional residual block.

    Structure:

        Input
          │
          ├─────────────── skip connection
          │
          ▼
        Conv 3×3
          ↓
        BatchNorm
          ↓
        ReLU
          ↓
        Conv 3×3
          ↓
        BatchNorm
          │
          + <── skip
          ↓
        ReLU
          │
          ▼
        Output

    Residual connections help the network learn corrections to the
    existing feature representation instead of learning every feature
    transformation from scratch.
    """

    def __init__(
        self,
        channels: int,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(channels),
        )

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        residual = x

        x = self.block(x)

        x = x + residual

        x = self.activation(x)

        return x


# ============================================================================
# OCEANEMBED-CNN
# ============================================================================


class OceanEmbedCNN(nn.Module):
    """
    Primary OceanEmbed CNN.

    Input:
        [B, 49, 64, 64]

    Output:
        [B, 15, 32, 32]

    Architecture
    ------------
    Input
        ↓
    Initial feature extraction
        49 → 64 channels
        64×64
        ↓
    Residual feature extraction
        64 channels
        64×64
        ↓
    Spatial encoder
        64 → 128 channels
        64×64 → 32×32
        ↓
    Latent residual representation
        128 channels
        32×32
        ↓
    Reconstruction head
        128 → 64 channels
        32×32
        ↓
    Output head
        64 → 15 channels
        32×32
        ↓
    Subsurface temperature
    """

    def __init__(
        self,
        input_channels: int = INPUT_CHANNELS,
        latent_channels: int = LATENT_CHANNELS,
        output_channels: int = OUTPUT_CHANNELS,
    ):
        super().__init__()

        # ----------------------------------------------------------------
        # Store architecture parameters.
        # ----------------------------------------------------------------

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.output_channels = output_channels

        # ----------------------------------------------------------------
        # Input feature extraction.
        #
        # 49 input channels contain:
        #
        #     7 days × 7 surface variables
        #
        # Spatial resolution remains:
        #
        #     64 × 64
        # ----------------------------------------------------------------

        self.input_projection = nn.Sequential(

            nn.Conv2d(
                input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # High-resolution residual feature extraction.
        # ----------------------------------------------------------------

        self.high_resolution_features = nn.Sequential(

            ResidualBlock(64),

            ResidualBlock(64),
        )

        # ----------------------------------------------------------------
        # Spatial encoder.
        #
        # The stride-2 convolution reduces:
        #
        #     64 × 64
        #
        # to:
        #
        #     32 × 32
        #
        # while increasing the feature representation:
        #
        #     64 → 128 channels
        #
        # This 128-channel representation is our latent feature space.
        # ----------------------------------------------------------------

        self.encoder_downsample = nn.Sequential(

            nn.Conv2d(
                64,
                latent_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(latent_channels),

            nn.ReLU(inplace=True),
        )

        # ----------------------------------------------------------------
        # Latent feature processing.
        #
        # The spatial size remains 32 × 32 and the latent representation
        # remains 128 channels.
        # ----------------------------------------------------------------

        self.latent_features = nn.Sequential(

            ResidualBlock(latent_channels),

            ResidualBlock(latent_channels),
        )

        # ----------------------------------------------------------------
        # Reconstruction head.
        #
        # Convert the 128-channel latent representation into a smaller
        # feature representation before producing the 15 temperature
        # depth channels.
        # ----------------------------------------------------------------

        self.reconstruction_features = nn.Sequential(

            nn.Conv2d(
                latent_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            ResidualBlock(64),
        )

        # ----------------------------------------------------------------
        # Final temperature reconstruction layer.
        #
        # Output:
        #
        #     15 channels
        #
        # corresponding to:
        #
        #     0, 5, 10, 20, 30, 50, 75, 100,
        #     125, 150, 200, 300, 500, 700, 1000 m
        #
        # No activation function is applied because temperature is a
        # continuous regression target and can take values across the
        # normalized target range.
        # ----------------------------------------------------------------

        self.output_head = nn.Conv2d(
            64,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # ----------------------------------------------------------------
        # Initialize weights.
        # ----------------------------------------------------------------

        self._initialize_weights()

    # ==================================================================
    # WEIGHT INITIALIZATION
    # ==================================================================

    def _initialize_weights(self):

        for module in self.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

    # ==================================================================
    # FORWARD PASS
    # ==================================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor.

            Expected shape:
                [B, 49, 64, 64]

        Returns
        -------
        torch.Tensor

            Predicted subsurface temperature.

            Shape:
                [B, 15, 32, 32]
        """

        # --------------------------------------------------------------
        # Validate input dimensions.
        # --------------------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                "OceanEmbedCNN expects a 4D input tensor "
                "[B, C, H, W]. "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.shape[1] != self.input_channels:

            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"received {x.shape[1]}."
            )

        if x.shape[2] != INPUT_SIZE:

            raise ValueError(
                f"Expected input height {INPUT_SIZE}, "
                f"received {x.shape[2]}."
            )

        if x.shape[3] != INPUT_SIZE:

            raise ValueError(
                f"Expected input width {INPUT_SIZE}, "
                f"received {x.shape[3]}."
            )

        # --------------------------------------------------------------
        # High-resolution feature extraction.
        #
        # [B, 49, 64, 64]
        #        ↓
        # [B, 64, 64, 64]
        # --------------------------------------------------------------

        x = self.input_projection(x)

        x = self.high_resolution_features(x)

        # --------------------------------------------------------------
        # Spatial encoding.
        #
        # [B, 64, 64, 64]
        #        ↓
        # [B, 128, 32, 32]
        # --------------------------------------------------------------

        x = self.encoder_downsample(x)

        # --------------------------------------------------------------
        # Latent representation.
        #
        # [B, 128, 32, 32]
        #        ↓
        # [B, 128, 32, 32]
        # --------------------------------------------------------------

        x = self.latent_features(x)

        # --------------------------------------------------------------
        # Reconstruction feature extraction.
        #
        # [B, 128, 32, 32]
        #        ↓
        # [B, 64, 32, 32]
        # --------------------------------------------------------------

        x = self.reconstruction_features(x)

        # --------------------------------------------------------------
        # Temperature output.
        #
        # [B, 64, 32, 32]
        #        ↓
        # [B, 15, 32, 32]
        # --------------------------------------------------------------

        x = self.output_head(x)

        return x


# ============================================================================
# MODEL FACTORY
# ============================================================================


def create_oceanembed_model() -> OceanEmbedCNN:
    """
    Create the standard OceanEmbed-CNN.

    Returns
    -------
    OceanEmbedCNN
        Model configured according to the OceanEmbed V1 contract.
    """

    return OceanEmbedCNN(
        input_channels=INPUT_CHANNELS,
        latent_channels=LATENT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
    )


# ============================================================================
# PARAMETER COUNT
# ============================================================================


def count_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable parameters.
    """

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================================
# ARCHITECTURE SELF-CHECK
# ============================================================================


def architecture_check():
    """
    Basic architecture-only check.

    This does NOT use the OceanEmbed dataset.

    It verifies:

        [2, 49, 64, 64]
                    ↓
              OceanEmbed-CNN
                    ↓
        [2, 15, 32, 32]
    """

    print()
    print("=" * 72)
    print("OceanEmbed-CNN ARCHITECTURE CHECK")
    print("=" * 72)

    # --------------------------------------------------------------
    # Create model.
    # --------------------------------------------------------------

    model = create_oceanembed_model()

    print()
    print("Model:")
    print(model)

    # --------------------------------------------------------------
    # Count parameters.
    # --------------------------------------------------------------

    parameters = count_parameters(model)

    print()
    print(
        f"Trainable parameters: {parameters:,}"
    )

    # --------------------------------------------------------------
    # Dummy input.
    # --------------------------------------------------------------

    batch_size = 2

    x = torch.randn(
        batch_size,
        INPUT_CHANNELS,
        INPUT_SIZE,
        INPUT_SIZE,
        dtype=torch.float32,
    )

    print()
    print(
        f"Input shape: {tuple(x.shape)}"
    )

    # --------------------------------------------------------------
    # Forward pass.
    # --------------------------------------------------------------

    model.eval()

    with torch.no_grad():

        y = model(x)

    print(
        f"Output shape: {tuple(y.shape)}"
    )

    # --------------------------------------------------------------
    # Output validation.
    # --------------------------------------------------------------

    expected_shape = (
        batch_size,
        OUTPUT_CHANNELS,
        OUTPUT_SIZE,
        OUTPUT_SIZE,
    )

    if tuple(y.shape) != expected_shape:

        raise AssertionError(
            f"Output shape mismatch.\n"
            f"Expected: {expected_shape}\n"
            f"Received: {tuple(y.shape)}"
        )

    print(
        "PASS: output shape is exactly "
        "[B, 15, 32, 32]."
    )

    # --------------------------------------------------------------
    # Numerical validation.
    # --------------------------------------------------------------

    if not torch.isfinite(y).all():

        raise AssertionError(
            "Model output contains NaN or Inf values."
        )

    print(
        "PASS: output contains no NaN or Inf values."
    )

    print()
    print("=" * 72)
    print("OceanEmbed-CNN ARCHITECTURE CHECK PASSED")
    print("=" * 72)
    print()


# ============================================================================
# MAIN
# ============================================================================


if __name__ == "__main__":
    architecture_check()