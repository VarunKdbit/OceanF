from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import torch


# =====================================================================
# PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPTS_DIR = PROJECT_ROOT / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from oceanembed_model import OceanEmbedCNN


# =====================================================================
# ML PATHS
# =====================================================================

ML_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ML"
)

# IMPORTANT:
# The existing V1 checkpoints were trained using ml_config.json.
# Therefore this file is the authoritative normalization source
# for the current V1 inference pipeline.

V1_CONFIG_PATH = (
    ML_DIR
    / "ml_config.json"
)

CHECKPOINT_DIR = (
    ML_DIR
    / "checkpoints"
)


# =====================================================================
# FROZEN OCEANEMBED CONTRACT
# =====================================================================

DEPTHS_M = [
    0,
    5,
    10,
    20,
    30,
    50,
    75,
    100,
    125,
    150,
    200,
    300,
    500,
    700,
    1000,
]

SEEDS = [
    42,
    123,
    2024,
]

CHECKPOINT_NAMES = {
    42: "oceanembed_e2_seed42.pt",
    123: "oceanembed_e2_seed123.pt",
    2024: "oceanembed_e2_seed2024.pt",
}

INPUT_FEATURES = [
    "sst",
    "sss",
    "sla",
    "uo",
    "vo",
    "u_wind",
    "v_wind",
]

HISTORY_DAYS = 7

INPUT_CHANNELS = 49

OUTPUT_CHANNELS = 15

LATENT_CHANNELS = 128

INPUT_HEIGHT = 64
INPUT_WIDTH = 64

OUTPUT_HEIGHT = 32
OUTPUT_WIDTH = 32


# =====================================================================
# HELPERS
# =====================================================================

def load_json(path: Path) -> dict:

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


def validate_finite(
    tensor: torch.Tensor,
    name: str,
) -> None:

    if not torch.isfinite(
        tensor
    ).all().item():

        raise ValueError(
            f"{name} contains non-finite values."
        )


def get_stat_value(
    stats: dict,
    key: str,
    context: str,
) -> float:

    if key not in stats:

        raise ValueError(
            f"Missing '{key}' in {context}."
        )

    return float(stats[key])


def find_depth_statistics(
    target_statistics: dict,
    depth: int,
) -> dict:

    # Support both possible JSON representations:
    #
    # "0"
    # "0.0"
    #
    # without assuming one exact serialization.

    candidates = [
        str(depth),
        str(float(depth)),
    ]

    for candidate in candidates:

        if candidate in target_statistics:

            return target_statistics[candidate]

    raise ValueError(
        f"Missing target statistics for depth {depth}m."
    )


# =====================================================================
# OCEANEMBED V1 ENSEMBLE
# =====================================================================

class OceanEmbedEnsemble:

    def __init__(
        self,
        device: str | torch.device = "cuda",
    ) -> None:

        self.device = torch.device(device)

        # -------------------------------------------------------------
        # Load the ORIGINAL V1 configuration.
        # -------------------------------------------------------------

        self.config = load_json(
            V1_CONFIG_PATH
        )

        self.models: Dict[
            int,
            OceanEmbedCNN,
        ] = {}

        self.input_means: Dict[
            str,
            float,
        ] = {}

        self.input_stds: Dict[
            str,
            float,
        ] = {}

        self.target_means: List[
            float
        ] = []

        self.target_stds: List[
            float
        ] = []

        self._validate_configuration()

        self._load_input_statistics()

        self._load_target_statistics()

        self._load_models()


    # -----------------------------------------------------------------
    # Validate V1 configuration
    # -----------------------------------------------------------------

    def _validate_configuration(
        self,
    ) -> None:

        if self.config.get(
            "project"
        ) != "OceanF":

            raise ValueError(
                "Unexpected project in ml_config.json."
            )

        # -------------------------------------------------------------
        # Input features
        # -------------------------------------------------------------

        features = self.config.get(
            "input_features"
        )

        if features != INPUT_FEATURES:

            raise ValueError(
                "Input features in ml_config.json do not match "
                "the frozen OceanEmbed contract."
            )

        # -------------------------------------------------------------
        # Target variable
        # -------------------------------------------------------------

        target_variable = self.config.get(
            "target_variable"
        )

        if target_variable != "thetao":

            raise ValueError(
                "Unexpected target variable in ml_config.json."
            )

        # -------------------------------------------------------------
        # Target depths
        # -------------------------------------------------------------

        depths = self.config.get(
            "target_depths_m"
        )

        if depths != DEPTHS_M:

            raise ValueError(
                "Target depths in ml_config.json do not match "
                "the frozen OceanEmbed contract."
            )

        # -------------------------------------------------------------
        # Domain
        # -------------------------------------------------------------

        domain = self.config.get(
            "domain",
            {},
        )

        if float(
            domain.get("latitude_min")
        ) != 5.0:

            raise ValueError(
                "Unexpected latitude_min in ml_config.json."
            )

        if float(
            domain.get("latitude_max")
        ) != 30.0:

            raise ValueError(
                "Unexpected latitude_max in ml_config.json."
            )

        if float(
            domain.get("longitude_min")
        ) != 45.0:

            raise ValueError(
                "Unexpected longitude_min in ml_config.json."
            )

        if float(
            domain.get("longitude_max")
        ) != 105.0:

            raise ValueError(
                "Unexpected longitude_max in ml_config.json."
            )

        if float(
            domain.get("resolution")
        ) != 0.25:

            raise ValueError(
                "Unexpected spatial resolution in ml_config.json."
            )

        # -------------------------------------------------------------
        # Time configuration
        # -------------------------------------------------------------

        time_config = self.config.get(
            "time",
            {},
        )

        if time_config.get(
            "train_start"
        ) != "2025-07-01":

            raise ValueError(
                "Unexpected V1 training start date."
            )

        if time_config.get(
            "train_end"
        ) != "2025-10-31":

            raise ValueError(
                "Unexpected V1 training end date."
            )

        if time_config.get(
            "validation_start"
        ) != "2025-11-01":

            raise ValueError(
                "Unexpected V1 validation start date."
            )

        if time_config.get(
            "validation_end"
        ) != "2025-11-30":

            raise ValueError(
                "Unexpected V1 validation end date."
            )

        if time_config.get(
            "test_start"
        ) != "2025-12-01":

            raise ValueError(
                "Unexpected V1 test start date."
            )

        if time_config.get(
            "test_end"
        ) != "2025-12-31":

            raise ValueError(
                "Unexpected V1 test end date."
            )

        # -------------------------------------------------------------
        # Normalization declaration
        # -------------------------------------------------------------

        normalization = self.config.get(
            "normalization",
            {},
        )

        if normalization.get(
            "inputs"
        ) != "training_period_mean_std":

            raise ValueError(
                "Unexpected input normalization definition."
            )

        if normalization.get(
            "target"
        ) != "training_period_mean_std_by_depth":

            raise ValueError(
                "Unexpected target normalization definition."
            )


    # -----------------------------------------------------------------
    # Load V1 input statistics
    # -----------------------------------------------------------------

    def _load_input_statistics(
        self,
    ) -> None:

        input_statistics = self.config.get(
            "input_statistics"
        )

        if input_statistics is None:

            raise ValueError(
                "input_statistics are missing from ml_config.json."
            )

        for feature in INPUT_FEATURES:

            if feature not in input_statistics:

                raise ValueError(
                    f"Missing input statistics for {feature}."
                )

            stats = input_statistics[
                feature
            ]

            mean = get_stat_value(
                stats,
                "mean",
                f"input_statistics[{feature}]",
            )

            std = get_stat_value(
                stats,
                "std",
                f"input_statistics[{feature}]",
            )

            if std <= 0:

                raise ValueError(
                    f"Invalid standard deviation for "
                    f"{feature}: {std}"
                )

            self.input_means[
                feature
            ] = mean

            self.input_stds[
                feature
            ] = std


    # -----------------------------------------------------------------
    # Load V1 target statistics
    # -----------------------------------------------------------------

    def _load_target_statistics(
        self,
    ) -> None:

        target_statistics = self.config.get(
            "target_statistics"
        )

        if target_statistics is None:

            raise ValueError(
                "target_statistics are missing from ml_config.json."
            )

        for depth in DEPTHS_M:

            stats = find_depth_statistics(
                target_statistics,
                depth,
            )

            mean = get_stat_value(
                stats,
                "mean",
                f"target_statistics[{depth}m]",
            )

            std = get_stat_value(
                stats,
                "std",
                f"target_statistics[{depth}m]",
            )

            if std <= 0:

                raise ValueError(
                    f"Invalid target standard deviation "
                    f"at {depth}m: {std}"
                )

            self.target_means.append(
                mean
            )

            self.target_stds.append(
                std
            )


    # -----------------------------------------------------------------
    # Load three V1 trained models
    # -----------------------------------------------------------------

    def _load_models(
        self,
    ) -> None:

        for seed in SEEDS:

            checkpoint_path = (
                CHECKPOINT_DIR
                / CHECKPOINT_NAMES[seed]
            )

            if not checkpoint_path.exists():

                raise FileNotFoundError(
                    f"Checkpoint not found for seed "
                    f"{seed}: {checkpoint_path}"
                )

            model = OceanEmbedCNN(
                input_channels=INPUT_CHANNELS,
                latent_channels=LATENT_CHANNELS,
                output_channels=OUTPUT_CHANNELS,
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            if "model_state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            elif "state_dict" in checkpoint:

                model.load_state_dict(
                    checkpoint[
                        "state_dict"
                    ]
                )

            else:

                raise ValueError(
                    f"Checkpoint for seed {seed} does not contain "
                    "model_state_dict or state_dict."
                )

            model.to(
                self.device
            )

            model.eval()

            self.models[
                seed
            ] = model

            print(
                f"Loaded seed {seed}: "
                f"{checkpoint_path.name}"
            )


    # -----------------------------------------------------------------
    # Normalize one raw feature
    # -----------------------------------------------------------------

    def normalize_feature(
        self,
        values: torch.Tensor,
        feature: str,
    ) -> torch.Tensor:

        if feature not in self.input_means:

            raise ValueError(
                f"Unknown input feature: {feature}"
            )

        mean = self.input_means[
            feature
        ]

        std = self.input_stds[
            feature
        ]

        result = (
            values - mean
        ) / std

        # Match the existing training pipeline:
        #
        # missing input values are replaced with zero
        # AFTER normalization.

        result = torch.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return result


    # -----------------------------------------------------------------
    # Normalize complete seven-day input window
    # -----------------------------------------------------------------

    def normalize_input_window(
        self,
        feature_arrays: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        """
        Convert seven days of raw/harmonized observations
        into the normalized V1 model input.

        Expected feature shape:

            [7, 64, 64]

        Expected output:

            [49, 64, 64]
        """

        normalized_channels = []

        for feature in INPUT_FEATURES:

            if feature not in feature_arrays:

                raise ValueError(
                    f"Missing required input feature: "
                    f"{feature}"
                )

            values = torch.as_tensor(
                feature_arrays[feature],
                dtype=torch.float32,
            )

            expected_shape = (
                HISTORY_DAYS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
            )

            if tuple(values.shape) != expected_shape:

                raise ValueError(
                    f"{feature} has shape "
                    f"{tuple(values.shape)}. "
                    f"Expected {expected_shape}."
                )

            normalized = (
                self.normalize_feature(
                    values,
                    feature,
                )
            )

            normalized_channels.append(
                normalized
            )

        # Feature-major ordering.
        #
        # SST day1...day7
        # SSS day1...day7
        # SLA day1...day7
        # UO day1...day7
        # VO day1...day7
        # U Wind day1...day7
        # V Wind day1...day7

        x = torch.cat(
            normalized_channels,
            dim=0,
        )

        expected_shape = (
            INPUT_CHANNELS,
            INPUT_HEIGHT,
            INPUT_WIDTH,
        )

        if tuple(x.shape) != expected_shape:

            raise RuntimeError(
                "Normalized input window produced "
                f"unexpected shape: {tuple(x.shape)}"
            )

        validate_finite(
            x,
            "Normalized V1 input window",
        )

        return x


    # -----------------------------------------------------------------
    # Validate normalized model input
    # -----------------------------------------------------------------

    def _validate_model_input(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if not isinstance(
            x,
            torch.Tensor,
        ):

            x = torch.as_tensor(
                x,
                dtype=torch.float32,
            )

        x = x.float()

        if x.ndim == 3:

            expected = (
                INPUT_CHANNELS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
            )

            if tuple(x.shape) != expected:

                raise ValueError(
                    f"Expected input shape {expected}, "
                    f"received {tuple(x.shape)}."
                )

            x = x.unsqueeze(0)

        elif x.ndim == 4:

            expected = (
                INPUT_CHANNELS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
            )

            if tuple(x.shape[1:]) != expected:

                raise ValueError(
                    "Expected batched input shape "
                    "[N,49,64,64], "
                    f"received {tuple(x.shape)}."
                )

        else:

            raise ValueError(
                "Input must have shape "
                "[49,64,64] or [N,49,64,64]."
            )

        validate_finite(
            x,
            "V1 model input",
        )

        return x.to(
            self.device
        )


    # -----------------------------------------------------------------
    # Ensemble prediction in normalized target space
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_normalized(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self._validate_model_input(
            x
        )

        predictions = []

        for seed in SEEDS:

            model = self.models[
                seed
            ]

            prediction = model(
                x
            )

            expected_shape = (
                x.shape[0],
                OUTPUT_CHANNELS,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
            )

            if tuple(
                prediction.shape
            ) != expected_shape:

                raise RuntimeError(
                    f"Seed {seed} produced unexpected "
                    f"shape: {tuple(prediction.shape)}"
                )

            validate_finite(
                prediction,
                f"Prediction from seed {seed}",
            )

            predictions.append(
                prediction
            )

        ensemble_prediction = (
            torch.stack(
                predictions,
                dim=0,
            ).mean(
                dim=0
            )
        )

        validate_finite(
            ensemble_prediction,
            "V1 ensemble normalized prediction",
        )

        return ensemble_prediction


    # -----------------------------------------------------------------
    # Denormalize target to degrees Celsius
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_temperature(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        normalized_prediction = (
            self.predict_normalized(
                x
            )
        )

        means = torch.tensor(
            self.target_means,
            dtype=normalized_prediction.dtype,
            device=normalized_prediction.device,
        ).view(
            1,
            OUTPUT_CHANNELS,
            1,
            1,
        )

        stds = torch.tensor(
            self.target_stds,
            dtype=normalized_prediction.dtype,
            device=normalized_prediction.device,
        ).view(
            1,
            OUTPUT_CHANNELS,
            1,
            1,
        )

        temperature = (
            normalized_prediction * stds
            + means
        )

        validate_finite(
            temperature,
            "V1 temperature prediction",
        )

        return temperature


    # -----------------------------------------------------------------
    # Complete raw-window inference
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_from_raw_window(
        self,
        feature_arrays: Dict[
            str,
            torch.Tensor,
        ],
    ) -> torch.Tensor:

        """
        Complete V1 preprocessing + inference.

        Input:

            seven days of raw/harmonized observations
            for all seven features.

        Output:

            [15, 32, 32]

        Units:

            degrees Celsius
        """

        normalized_input = (
            self.normalize_input_window(
                feature_arrays
            )
        )

        prediction = (
            self.predict_temperature(
                normalized_input
            )
        )

        if prediction.shape[0] != 1:

            raise RuntimeError(
                "Unexpected batch dimension "
                "in single prediction."
            )

        return prediction[0]


    # -----------------------------------------------------------------
    # Single normalized-input prediction
    # -----------------------------------------------------------------

    @torch.no_grad()
    def predict_single(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        temperature = (
            self.predict_temperature(
                x
            )
        )

        if temperature.shape[0] != 1:

            raise ValueError(
                "predict_single expects exactly one sample."
            )

        return temperature[0]


    # -----------------------------------------------------------------
    # Backend information
    # -----------------------------------------------------------------

    def get_model_info(
        self,
    ) -> dict:

        return {
            "project": "OceanF",
            "experiment": "OceanEmbed-CNN",
            "experiment_variant": "E2_7day_retrospective",
            "checkpoint_version": "V1",
            "ensemble_type": "three_seed_mean",
            "seeds": SEEDS,
            "history_days": HISTORY_DAYS,
            "input_features": INPUT_FEATURES,
            "input_channels": INPUT_CHANNELS,
            "input_shape": [
                INPUT_CHANNELS,
                INPUT_HEIGHT,
                INPUT_WIDTH,
            ],
            "output_channels": OUTPUT_CHANNELS,
            "output_shape": [
                OUTPUT_CHANNELS,
                OUTPUT_HEIGHT,
                OUTPUT_WIDTH,
            ],
            "depths_m": DEPTHS_M,
            "temperature_unit": "degC",
            "normalization": {
                "source": "ml_config.json",
                "period": {
                    "start": "2025-07-01",
                    "end": "2025-10-31",
                },
            },
            "training_period": {
                "start": "2025-07-01",
                "end": "2025-10-31",
            },
            "validation_period": {
                "start": "2025-11-01",
                "end": "2025-11-30",
            },
            "held_out_test_period": {
                "start": "2025-12-01",
                "end": "2025-12-31",
            },
            "future_final_training_config": {
                "statistics_available": True,
                "used_by_current_v1_checkpoints": False,
            },
        }


# =====================================================================
# SELF TEST
# =====================================================================

def self_test() -> None:

    print()
    print("=" * 72)
    print("OceanEmbed V1 INFERENCE ENGINE SELF TEST")
    print("=" * 72)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print()
    print(
        f"Device: {device}"
    )

    print(
        "Normalization source: "
        "data/processed/ML/ml_config.json"
    )

    engine = OceanEmbedEnsemble(
        device=device
    )

    print()
    print(
        "PASS: V1 configuration and normalization "
        "statistics loaded"
    )

    # -------------------------------------------------------------
    # Verify normalization against the loaded V1 means.
    # -------------------------------------------------------------

    raw_window = {}

    for feature in INPUT_FEATURES:

        mean = engine.input_means[
            feature
        ]

        raw_window[feature] = (
            torch.full(
                (
                    HISTORY_DAYS,
                    INPUT_HEIGHT,
                    INPUT_WIDTH,
                ),
                float(mean),
                dtype=torch.float32,
            )
        )

    normalized = (
        engine.normalize_input_window(
            raw_window
        )
    )

    expected_input_shape = (
        INPUT_CHANNELS,
        INPUT_HEIGHT,
        INPUT_WIDTH,
    )

    if tuple(
        normalized.shape
    ) != expected_input_shape:

        raise RuntimeError(
            "Unexpected normalized input shape: "
            f"{tuple(normalized.shape)}"
        )

    print(
        "PASS: V1 raw 7-day window normalized to "
        f"{list(normalized.shape)}"
    )

    max_abs_normalized = (
        normalized.abs().max().item()
    )

    if max_abs_normalized > 1e-5:

        raise RuntimeError(
            "V1 normalization self-test failed. "
            f"Maximum absolute normalized value: "
            f"{max_abs_normalized}"
        )

    print(
        "PASS: V1 normalization is correct"
    )

    # -------------------------------------------------------------
    # Run ensemble.
    # -------------------------------------------------------------

    prediction = (
        engine.predict_temperature(
            normalized
        )
    )

    expected_output_shape = (
        1,
        OUTPUT_CHANNELS,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    )

    if tuple(
        prediction.shape
    ) != expected_output_shape:

        raise RuntimeError(
            "Unexpected temperature prediction shape: "
            f"{tuple(prediction.shape)}"
        )

    print(
        "PASS: V1 ensemble output shape = "
        f"{list(prediction.shape)}"
    )

    validate_finite(
        prediction,
        "V1 self-test temperature output",
    )

    print(
        "PASS: V1 temperature output is finite"
    )

    # -------------------------------------------------------------
    # Test complete raw-window API.
    # -------------------------------------------------------------

    raw_prediction = (
        engine.predict_from_raw_window(
            raw_window
        )
    )

    expected_single_shape = (
        OUTPUT_CHANNELS,
        OUTPUT_HEIGHT,
        OUTPUT_WIDTH,
    )

    if tuple(
        raw_prediction.shape
    ) != expected_single_shape:

        raise RuntimeError(
            "Unexpected raw-window prediction shape: "
            f"{tuple(raw_prediction.shape)}"
        )

    print(
        "PASS: V1 raw-window prediction shape = "
        f"{list(raw_prediction.shape)}"
    )

    validate_finite(
        raw_prediction,
        "V1 raw-window prediction",
    )

    print(
        "PASS: V1 raw-window prediction is finite"
    )

    print()
    print(
        "Target depths:"
    )

    print(
        DEPTHS_M
    )

    print()
    print(
        "Self-test temperature range: "
        f"{raw_prediction.min().item():.3f} °C "
        f"to "
        f"{raw_prediction.max().item():.3f} °C"
    )

    print()
    print("=" * 72)
    print(
        "OCEANEMBED V1 INFERENCE ENGINE SELF TEST PASSED"
    )
    print("=" * 72)


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":
    self_test()