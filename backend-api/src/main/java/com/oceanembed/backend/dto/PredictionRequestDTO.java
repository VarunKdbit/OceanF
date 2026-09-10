package com.oceanembed.backend.dto;

import jakarta.validation.constraints.*;
import java.time.LocalDate;
import java.util.List;
import java.util.Set;

public class PredictionRequestDTO {

    private static final Set<Integer> SUPPORTED_DEPTHS = Set.of(
        0, 5, 10, 20, 30, 50, 75, 100,
        125, 150, 200, 300, 500, 700, 1000
    );

    @NotNull
    @DecimalMin(value = "-90.0")
    @DecimalMax(value = "90.0")
    private Double latitude;

    @NotNull
    @DecimalMin(value = "-180.0")
    @DecimalMax(value = "180.0")
    private Double longitude;

    @NotNull
    private LocalDate date;

    private String regionName;

    /**
     * Optional legacy surface metadata accepted from older frontend clients.
     * It is NOT used as the OceanEmbed model input.
     */
    private SurfaceVariablesDTO surface;

    private List<Integer> depths;

    public boolean hasSupportedDepths() {
        return depths == null || depths.stream().allMatch(SUPPORTED_DEPTHS::contains);
    }

    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }
    public String getRegionName() { return regionName; }
    public void setRegionName(String regionName) { this.regionName = regionName; }
    public SurfaceVariablesDTO getSurface() { return surface; }
    public void setSurface(SurfaceVariablesDTO surface) { this.surface = surface; }
    public List<Integer> getDepths() { return depths; }
    public void setDepths(List<Integer> depths) { this.depths = depths; }
}
