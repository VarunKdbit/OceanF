package com.oceanembed.backend.dto;

import java.time.LocalDate;
import java.util.List;

/** Mirrors ml-service/app/schemas.py:PredictionRequest exactly (field names must match). */
public class MlPredictionRequest {
    private Double latitude;
    private Double longitude;
    private LocalDate date;
    private MlSurfaceVariables surface;
    private List<Integer> depths;
    private String model_version;

    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }
    public MlSurfaceVariables getSurface() { return surface; }
    public void setSurface(MlSurfaceVariables surface) { this.surface = surface; }
    public List<Integer> getDepths() { return depths; }
    public void setDepths(List<Integer> depths) { this.depths = depths; }
    public String getModel_version() { return model_version; }
    public void setModel_version(String model_version) { this.model_version = model_version; }
}
