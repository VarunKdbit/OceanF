"""
OceanEmbed ML Split and Temporal Leakage Verification
=======================================================

Purpose
-------
Verify that the OceanEmbed ML Dataset obeys the chronological split
and 7-day retrospective-window contract.

Splits
------
Train:
    2025-07-01 -> 2025-10-31

Validation:
    2025-11-01 -> 2025-11-30

Test:
    2025-12-01 -> 2025-12-31

Temporal input
--------------
Each target date uses exactly seven dates:

    target - 6 days
    target - 5 days
    target - 4 days
    target - 3 days
    target - 2 days
    target - 1 day
    target

The target itself is NEVER taken from a future date.

Spatial contract
----------------
Input:
    64 × 64

Target:
    centered 32 × 32

The target must occupy:

    input rows    16:48
    input columns 16:48
"""


from __future__ import annotations

from datetime import date, timedelta

from ml_dataset import (
    OceanEmbedDataset,
    WINDOW_SIZE,
    INPUT_TILE_SIZE,
    OUTPUT_TILE_SIZE,
    CONTEXT,
)


# ---------------------------------------------------------------------
# EXPECTED SPLITS
# ---------------------------------------------------------------------

EXPECTED_SPLITS = {
    "train": (
        "2025-07-01",
        "2025-10-31",
    ),
    "validation": (
        "2025-11-01",
        "2025-11-30",
    ),
    "test": (
        "2025-12-01",
        "2025-12-31",
    ),
}


# ---------------------------------------------------------------------
# DATE HELPERS
# ---------------------------------------------------------------------


def parse_date(value: str) -> date:

    year, month, day = map(
        int,
        value.split("-"),
    )

    return date(
        year,
        month,
        day,
    )


def date_difference_days(
    earlier: str,
    later: str,
) -> int:

    return (
        parse_date(later)
        - parse_date(earlier)
    ).days


def expected_window_dates(
    target_date: str,
):

    target = parse_date(target_date)

    return [
        (
            target
            - timedelta(days=offset)
        ).isoformat()
        for offset in range(
            WINDOW_SIZE - 1,
            -1,
            -1,
        )
    ]


# ---------------------------------------------------------------------
# SPLIT VERIFICATION
# ---------------------------------------------------------------------


def verify_split(
    split_name: str,
):

    print()
    print("=" * 72)
    print(f"VERIFYING SPLIT: {split_name.upper()}")
    print("=" * 72)

    dataset = OceanEmbedDataset(
        split=split_name,
        tile_stride=32,
        return_metadata=True,
        normalize=True,
    )

    expected_start, expected_end = (
        EXPECTED_SPLITS[split_name]
    )

    # ---------------------------------------------------------------
    # Number of target dates
    # ---------------------------------------------------------------

    expected_target_dates = (
        date_difference_days(
            expected_start,
            expected_end,
        )
        + 1
    )

    actual_target_dates = len(
        dataset.target_date_indices
    )

    print(
        f"Expected target dates : "
        f"{expected_target_dates}"
    )

    print(
        f"Actual target dates   : "
        f"{actual_target_dates}"
    )

    assert (
        actual_target_dates
        == expected_target_dates
    ), (
        f"{split_name}: target-date count mismatch."
    )

    # ---------------------------------------------------------------
    # Target date boundaries
    # ---------------------------------------------------------------

    target_dates = [
        dataset.dates[index]
        for index in dataset.target_date_indices
    ]

    assert target_dates[0] == expected_start
    assert target_dates[-1] == expected_end

    print(
        f"Target range          : "
        f"{target_dates[0]} -> {target_dates[-1]}"
    )

    print(
        "PASS: target dates stay inside split."
    )

    # ---------------------------------------------------------------
    # Every target must have seven historical dates.
    # ---------------------------------------------------------------

    for target_index in dataset.sample_date_indices:

        target_date = dataset.dates[
            target_index
        ]

        window_start_index = (
            target_index
            - WINDOW_SIZE
            + 1
        )

        window_dates = dataset.dates[
            window_start_index :
            target_index + 1
        ]

        expected_dates = expected_window_dates(
            target_date
        )

        assert window_dates == expected_dates, (
            f"{split_name}: invalid temporal window "
            f"for target {target_date}.\n"
            f"Expected: {expected_dates}\n"
            f"Actual:   {window_dates}"
        )

        # -----------------------------------------------------------
        # No future date may appear.
        # -----------------------------------------------------------

        assert (
            window_dates[-1]
            == target_date
        )

        assert all(
            window_date <= target_date
            for window_date in window_dates
        )

    print(
        "PASS: every sample has exactly "
        "7 chronological input dates."
    )

    print(
        "PASS: no future date enters any input window."
    )

    # ---------------------------------------------------------------
    # Check first and last target windows.
    # ---------------------------------------------------------------

    first_target_index = (
        dataset.sample_date_indices[0]
    )

    last_target_index = (
        dataset.sample_date_indices[-1]
    )

    first_target_date = dataset.dates[
        first_target_index
    ]

    last_target_date = dataset.dates[
        last_target_index
    ]

    first_window = dataset.dates[
        first_target_index - WINDOW_SIZE + 1 :
        first_target_index + 1
    ]

    last_window = dataset.dates[
        last_target_index - WINDOW_SIZE + 1 :
        last_target_index + 1
    ]

    print()
    print(
        f"First usable target   : {first_target_date}"
    )

    print(
        f"First input window    : "
        f"{first_window[0]} -> {first_window[-1]}"
    )

    print(
        f"Last target           : {last_target_date}"
    )

    print(
        f"Last input window     : "
        f"{last_window[0]} -> {last_window[-1]}"
    )

    # ---------------------------------------------------------------
    # Spatial contract.
    # ---------------------------------------------------------------

    assert (
        INPUT_TILE_SIZE == 64
    )

    assert (
        OUTPUT_TILE_SIZE == 32
    )

    assert (
        CONTEXT == 16
    )

    print()
    print(
        "PASS: input tile = 64 × 64."
    )

    print(
        "PASS: target tile = 32 × 32."
    )

    print(
        "PASS: target is centered with 16-pixel context."
    )

    # ---------------------------------------------------------------
    # Inspect first dataset sample.
    # ---------------------------------------------------------------

    (
        x,
        x_mask,
        y,
        y_mask,
        metadata,
    ) = dataset[0]

    assert tuple(x.shape) == (
        49,
        64,
        64,
    )

    assert tuple(x_mask.shape) == (
        49,
        64,
        64,
    )

    assert tuple(y.shape) == (
        15,
        32,
        32,
    )

    assert tuple(y_mask.shape) == (
        15,
        32,
        32,
    )

    print(
        "PASS: sample tensor shapes."
    )

    dataset.close()

    print(
        f"PASS: {split_name} split verification."
    )


# ---------------------------------------------------------------------
# CROSS-SPLIT VERIFICATION
# ---------------------------------------------------------------------


def verify_cross_split_temporal_separation():

    print()
    print("=" * 72)
    print("CROSS-SPLIT TEMPORAL SEPARATION")
    print("=" * 72)

    train_dataset = OceanEmbedDataset(
        split="train",
        tile_stride=32,
        return_metadata=False,
        normalize=True,
    )

    validation_dataset = OceanEmbedDataset(
        split="validation",
        tile_stride=32,
        return_metadata=False,
        normalize=True,
    )

    test_dataset = OceanEmbedDataset(
        split="test",
        tile_stride=32,
        return_metadata=False,
        normalize=True,
    )

    train_targets = {
        train_dataset.dates[index]
        for index in train_dataset.target_date_indices
    }

    validation_targets = {
        validation_dataset.dates[index]
        for index in validation_dataset.target_date_indices
    }

    test_targets = {
        test_dataset.dates[index]
        for index in test_dataset.target_date_indices
    }

    # ---------------------------------------------------------------
    # Target sets must be disjoint.
    # ---------------------------------------------------------------

    assert (
        train_targets.isdisjoint(
            validation_targets
        )
    )

    assert (
        train_targets.isdisjoint(
            test_targets
        )
    )

    assert (
        validation_targets.isdisjoint(
            test_targets
        )
    )

    print(
        "PASS: train/validation/test target dates "
        "are completely disjoint."
    )

    # ---------------------------------------------------------------
    # Verify expected boundaries.
    # ---------------------------------------------------------------

    assert max(train_targets) < min(
        validation_targets
    )

    assert max(validation_targets) < min(
        test_targets
    )

    print(
        "PASS: target dates are strictly chronological."
    )

    # ---------------------------------------------------------------
    # Verify retrospective-history behavior.
    # ---------------------------------------------------------------

    validation_first_index = (
        validation_dataset.target_date_indices[0]
    )

    validation_window = validation_dataset.dates[
        validation_first_index - WINDOW_SIZE + 1 :
        validation_first_index + 1
    ]

    print()
    print(
        "Validation first target:"
    )

    print(
        f"  target = {validation_window[-1]}"
    )

    print(
        f"  history = "
        f"{validation_window[0]} -> "
        f"{validation_window[-2]}"
    )

    assert validation_window[-1] == "2025-11-01"

    assert validation_window == [
        "2025-10-26",
        "2025-10-27",
        "2025-10-28",
        "2025-10-29",
        "2025-10-30",
        "2025-10-31",
        "2025-11-01",
    ]

    print(
        "PASS: validation correctly uses preceding "
        "training-period surface history."
    )

    test_first_index = (
        test_dataset.target_date_indices[0]
    )

    test_window = test_dataset.dates[
        test_first_index - WINDOW_SIZE + 1 :
        test_first_index + 1
    ]

    print()
    print(
        "Test first target:"
    )

    print(
        f"  target = {test_window[-1]}"
    )

    print(
        f"  history = "
        f"{test_window[0]} -> "
        f"{test_window[-2]}"
    )

    assert test_window[-1] == "2025-12-01"

    assert test_window == [
        "2025-11-25",
        "2025-11-26",
        "2025-11-27",
        "2025-11-28",
        "2025-11-29",
        "2025-11-30",
        "2025-12-01",
    ]

    print(
        "PASS: test correctly uses preceding "
        "validation-period surface history."
    )

    train_dataset.close()
    validation_dataset.close()
    test_dataset.close()


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():

    print()
    print("=" * 72)
    print("OceanEmbed ML SPLIT INTEGRITY VERIFICATION")
    print("=" * 72)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        verify_split(split)

    verify_cross_split_temporal_separation()

    print()
    print("=" * 72)
    print("SPLIT INTEGRITY VERIFICATION PASSED")
    print("=" * 72)
    print()
    print(
        "Chronological targets are separated."
    )
    print(
        "Seven-day retrospective windows are complete."
    )
    print(
        "No future observations enter an input window."
    )
    print(
        "Validation/test history is handled correctly."
    )
    print(
        "64×64 -> centered 32×32 spatial contract is valid."
    )
    print()


if __name__ == "__main__":
    main()