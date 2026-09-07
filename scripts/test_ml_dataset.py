import sys
from pathlib import Path

import torch


# ============================================================
# Make scripts/ml_dataset.py importable
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SCRIPT_DIR),
    )


from ml_dataset import OceanEmbedDataset


# ============================================================
# TEST
# ============================================================

print("=" * 80)
print("OCEANEMBED ML DATASET SMOKE TEST")
print("=" * 80)


dataset = OceanEmbedDataset()


# ============================================================
# BASIC DATASET CHECK
# ============================================================

print()
print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print()
print("Dataset length:")
print(len(dataset))

print()
print("Latitude size:")
print(dataset.num_lat)

print()
print("Longitude size:")
print(dataset.num_lon)

print()
print("First date:")
print(dataset.dates[0])

print()
print("Last date:")
print(dataset.dates[-1])

print()
print("Actual selected depths:")
print(dataset.actual_depths)


# ============================================================
# FIRST SAMPLE
# ============================================================

print()
print("=" * 80)
print("FIRST SAMPLE")
print("=" * 80)

sample = dataset[0]

print()
print("Sample date:")
print(sample["date"])

print()
print("Input shape:")
print(sample["inputs"].shape)

print()
print("Target shape:")
print(sample["target"].shape)

print()
print("Input mask shape:")
print(sample["input_mask"].shape)

print()
print("Target mask shape:")
print(sample["target_mask"].shape)


# ============================================================
# DATA TYPES
# ============================================================

print()
print("=" * 80)
print("DATA TYPES")
print("=" * 80)

print()
print("Input dtype:")
print(sample["inputs"].dtype)

print()
print("Target dtype:")
print(sample["target"].dtype)

print()
print("Input mask dtype:")
print(sample["input_mask"].dtype)

print()
print("Target mask dtype:")
print(sample["target_mask"].dtype)


# ============================================================
# FINITE CHECK
# ============================================================

print()
print("=" * 80)
print("FINITE VALUE CHECK")
print("=" * 80)

input_finite = torch.isfinite(
    sample["inputs"]
).all().item()

target_finite = torch.isfinite(
    sample["target"]
).all().item()

print()
print("Input finite:")
print(input_finite)

print()
print("Target finite:")
print(target_finite)


# ============================================================
# MASK CHECK
# ============================================================

print()
print("=" * 80)
print("MASK CHECK")
print("=" * 80)

input_valid = (
    sample["input_mask"]
    .sum()
    .item()
)

target_valid = (
    sample["target_mask"]
    .sum()
    .item()
)

print()
print("Input valid points:")
print(input_valid)

print()
print("Target valid points:")
print(target_valid)


# ============================================================
# VALUE CHECK
# ============================================================

print()
print("=" * 80)
print("VALUE CHECK")
print("=" * 80)

print()
print("Input minimum:")
print(sample["inputs"].min().item())

print()
print("Input maximum:")
print(sample["inputs"].max().item())

print()
print("Target minimum:")
print(sample["target"].min().item())

print()
print("Target maximum:")
print(sample["target"].max().item())


# ============================================================
# SECOND SAMPLE
# ============================================================

print()
print("=" * 80)
print("SECOND SAMPLE CHECK")
print("=" * 80)

sample_2 = dataset[1]

print()
print("Second sample date:")
print(sample_2["date"])

print()
print("Second sample input shape:")
print(sample_2["inputs"].shape)

print()
print("Second sample target shape:")
print(sample_2["target"].shape)


# ============================================================
# ASSERTIONS
# ============================================================

expected_input_shape = (
    7,
    101,
    241,
)

expected_target_shape = (
    15,
    101,
    241,
)


if len(dataset) != 184:

    raise AssertionError(
        f"Expected dataset length 184, "
        f"got {len(dataset)}"
    )


if dataset.dates[0] != "2025-07-01":

    raise AssertionError(
        f"Unexpected first date: "
        f"{dataset.dates[0]}"
    )


if dataset.dates[-1] != "2025-12-31":

    raise AssertionError(
        f"Unexpected last date: "
        f"{dataset.dates[-1]}"
    )


if tuple(sample["inputs"].shape) != expected_input_shape:

    raise AssertionError(
        f"Unexpected input shape: "
        f"{tuple(sample['inputs'].shape)}"
    )


if tuple(sample["target"].shape) != expected_target_shape:

    raise AssertionError(
        f"Unexpected target shape: "
        f"{tuple(sample['target'].shape)}"
    )


if tuple(sample["input_mask"].shape) != expected_input_shape:

    raise AssertionError(
        f"Unexpected input mask shape: "
        f"{tuple(sample['input_mask'].shape)}"
    )


if tuple(sample["target_mask"].shape) != expected_target_shape:

    raise AssertionError(
        f"Unexpected target mask shape: "
        f"{tuple(sample['target_mask'].shape)}"
    )


# ============================================================
# SUCCESS
# ============================================================

print()
print("=" * 80)
print("STEP 1 — ML DATASET LOADER: PASS")
print("=" * 80)


dataset.close()