import netCDF4
import numpy as np


# ============================================================
# OceanEmbed Spatial Grid Inspection
# Step 1: Inspect latitude/longitude grids
# ============================================================


DATA_FILES = {
    "sst": "data/processed/SST/SST_processed.nc",
    "sss": "data/processed/SSS/SSS_processed.nc",
    "sla": "data/processed/SSA/SSA_processed.nc",
    "currents": "data/processed/Currents/Currents_processed.nc",
    "winds": "data/processed/Winds/Winds_processed.nc",
    "thetao": "data/processed/SubsurfaceTemp/SubsurfaceTemp_processed.nc",
}


def inspect_coordinate(dataset, variable_name):

    coordinate = np.asarray(
        dataset.variables[variable_name][:],
        dtype=np.float64,
    )

    print(f"{variable_name} size: {len(coordinate)}")

    if len(coordinate) > 0:

        print(
            f"{variable_name} first: "
            f"{coordinate[0]}"
        )

        print(
            f"{variable_name} last: "
            f"{coordinate[-1]}"
        )

        print(
            f"{variable_name} min: "
            f"{np.nanmin(coordinate)}"
        )

        print(
            f"{variable_name} max: "
            f"{np.nanmax(coordinate)}"
        )

        if len(coordinate) > 1:

            differences = np.diff(coordinate)

            print(
                f"{variable_name} mean spacing: "
                f"{np.mean(differences)}"
            )

            print(
                f"{variable_name} minimum spacing: "
                f"{np.min(differences)}"
            )

            print(
                f"{variable_name} maximum spacing: "
                f"{np.max(differences)}"
            )

            if np.all(differences > 0):

                print(
                    f"{variable_name} ordering: "
                    "INCREASING"
                )

            elif np.all(differences < 0):

                print(
                    f"{variable_name} ordering: "
                    "DECREASING"
                )

            else:

                print(
                    f"{variable_name} ordering: "
                    "NON-MONOTONIC"
                )


print("=" * 80)
print("OCEANEMBED SPATIAL GRID INSPECTION")
print("=" * 80)


for name, file_path in DATA_FILES.items():

    print()
    print("=" * 80)
    print(name.upper())
    print("=" * 80)

    dataset = netCDF4.Dataset(
        file_path,
        mode="r",
    )

    print()
    print("File:")
    print(file_path)

    print()
    print("Dimensions:")

    for dimension_name, dimension in dataset.dimensions.items():

        print(
            f"  {dimension_name}: "
            f"{len(dimension)}"
        )

    print()
    print("LATITUDE")

    inspect_coordinate(
        dataset,
        "latitude",
    )

    print()
    print("LONGITUDE")

    inspect_coordinate(
        dataset,
        "longitude",
    )

    dataset.close()


print()
print("=" * 80)
print("TARGET OCEANEMBED GRID")
print("=" * 80)

print()
print("Expected latitude:")
print("5.0°N to 30.0°N")

print()
print("Expected longitude:")
print("45.0°E to 105.0°E")

print()
print("Expected resolution:")
print("0.25° × 0.25°")

print()
print("Expected shape:")
print("101 × 241")

print()
print("=" * 80)
print("SPATIAL INSPECTION COMPLETE")
print("=" * 80)