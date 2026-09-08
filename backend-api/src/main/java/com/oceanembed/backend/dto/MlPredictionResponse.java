package com.oceanembed.backend.dto;

import java.time.LocalDate;
import java.util.List;

/** Mirrors ml-service/app/schemas.py:PredictionResponse exactly. */
public class MlPredictionResponse {
    private Double latitude;
    private Double longitude;
    private LocalDate date;
    private String model_version;
    private Double grid_resolution_deg;
    private List<MlDepthPrediction> predictions;

    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }
    public String getModel_version() { return model_version; }
    public void setModel_version(String model_version) { this.model_version = model_version; }
    public Double getGrid_resolution_deg() { return grid_resolution_deg; }
    public void setGrid_resolution_deg(Double grid_resolution_deg) { this.grid_resolution_deg = grid_resolution_deg; }
    public List<MlDepthPrediction> getPredictions() { return predictions; }
    public void setPredictions(List<MlDepthPrediction> predictions) { this.predictions = predictions; }
}
