package com.oceanembed.backend.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import java.time.LocalDate;
import java.util.List;

/**
 * Contract the frontend (React) sends to Spring Boot's /api/v1/predictions endpoint.
 */
public class PredictionRequestDTO {

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

    /** Optional human-readable region label, e.g. "Arabian Sea" */
    private String regionName;

    @NotNull
    @Valid
    private SurfaceVariablesDTO surface;

    /** Requested depth levels in meters. Defaults applied server-side if omitted. */
    private List<@Min(0) @Max(2000) Integer> depths;

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
