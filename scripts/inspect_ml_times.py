import netCDF4


# ============================================================
# OceanEmbed Dataset Time Inspection
# Step 1: Inspect actual temporal coverage
# ============================================================


DATA_FILES = {
    "sst": "data/processed/SST/SST_processed.nc",
    "sss": "data/processed/SSS/SSS_processed.nc",
    "sla": "data/processed/SSA/SSA_processed.nc",
    "currents": "data/processed/Currents/Currents_processed.nc",
    "winds": "data/processed/Winds/Winds_processed.nc",
    "thetao": "data/processed/SubsurfaceTemp/SubsurfaceTemp_processed.nc",
}


def decode_time(dataset, variable_name):

    variable = dataset.variables[variable_name]

    values = variable[:]

    units = variable.units

    calendar = getattr(
        variable,
        "calendar",
        "standard",
    )

    dates = netCDF4.num2date(
        values,
        units=units,
        calendar=calendar,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    )

    return dates


print("=" * 80)
print("OCEANEMBED DATASET TIME INSPECTION")
print("=" * 80)


all_dates = {}


for name, file_path in DATA_FILES.items():

    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    dataset = netCDF4.Dataset(
        file_path,
        mode="r",
    )

    print("File:")
    print(file_path)

    print()
    print("Dimensions:")

    for dimension_name, dimension in dataset.dimensions.items():

        print(
            f"  {dimension_name}: {len(dimension)}"
        )

    print()
    print("Time-like variables:")

    for variable_name, variable in dataset.variables.items():

        if (
            "time" in variable_name.lower()
            or variable_name in ["time", "valid_time"]
        ):

            print(
                f"  {variable_name}: "
                f"shape={variable.shape}"
            )

    # --------------------------------------------------------
    # Select time coordinate
    # --------------------------------------------------------

    if "time" in dataset.variables:

        time_variable_name = "time"

    elif "valid_time" in dataset.variables:

        time_variable_name = "valid_time"

    else:

        print()
        print("[FAIL] No time coordinate found")

        dataset.close()

        continue

    # --------------------------------------------------------
    # Decode dates
    # --------------------------------------------------------

    dates = decode_time(
        dataset,
        time_variable_name,
    )

    dates = list(dates)

    all_dates[name] = dates

    print()
    print("Time variable:")
    print(time_variable_name)

    print()
    print("Time units:")
    print(
        dataset.variables[
            time_variable_name
        ].units
    )

    print()
    print("Calendar:")
    print(
        getattr(
            dataset.variables[
                time_variable_name
            ],
            "calendar",
            "standard",
        )
    )

    print()
    print("Number of dates:")
    print(len(dates))

    if len(dates) > 0:

        print()
        print("First date:")
        print(dates[0])

        print()
        print("Last date:")
        print(dates[-1])

        print()
        print("First 5 dates:")

        for date in dates[:5]:
            print(" ", date)

        print()
        print("Last 5 dates:")

        for date in dates[-5:]:
            print(" ", date)

    dataset.close()


# ============================================================
# Common date intersection
# ============================================================

print()
print("=" * 80)
print("COMMON DATE ANALYSIS")
print("=" * 80)


date_sets = {}

for name, dates in all_dates.items():

    date_sets[name] = {
        date.strftime("%Y-%m-%d")
        for date in dates
    }


if len(date_sets) == len(DATA_FILES):

    common_dates = set.intersection(
        *date_sets.values()
    )

    common_dates = sorted(
        common_dates
    )

    print()
    print("Common dates across ALL datasets:")
    print(len(common_dates))

    if common_dates:

        print()
        print("Common start:")
        print(common_dates[0])

        print()
        print("Common end:")
        print(common_dates[-1])

        print()
        print("First 10 common dates:")

        for date in common_dates[:10]:
            print(" ", date)

        print()
        print("Last 10 common dates:")

        for date in common_dates[-10:]:
            print(" ", date)

    # --------------------------------------------------------
    # Expected OceanEmbed study period
    # --------------------------------------------------------

    expected_start = "2025-07-01"
    expected_end = "2025-12-31"

    study_dates = [
        date
        for date in common_dates
        if expected_start <= date <= expected_end
    ]

    print()
    print("=" * 80)
    print("OCEANEMBED STUDY PERIOD")
    print("=" * 80)

    print()
    print("Requested:")
    print(
        expected_start,
        "to",
        expected_end,
    )

    print()
    print("Common dates inside requested period:")
    print(len(study_dates))

    if study_dates:

        print()
        print("Actual common study start:")
        print(study_dates[0])

        print()
        print("Actual common study end:")
        print(study_dates[-1])

else:

    print()
    print(
        "[FAIL] Could not obtain time coordinates "
        "from all datasets."
    )


print()
print("=" * 80)
print("TIME INSPECTION COMPLETE")
print("=" * 80)