package com.oceanembed.backend.dto;

import java.time.Instant;
import java.time.LocalDate;
import java.util.List;

public class PredictionResponseDTO {

    private Long jobId;
    private String status;
    private Double latitude;
    private Double longitude;
    private LocalDate date;
    private String modelVersion;
    private Double gridResolutionDeg;
    private Instant createdAt;
    private Instant completedAt;
    private String errorMessage;
    private List<DepthPredictionDTO> predictions;

    public Long getJobId() {
        return jobId;
    }

    public void setJobId(Long jobId) {
        this.jobId = jobId;
    }

    public String getStatus() {
        return status;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public Double getLatitude() {
        return latitude;
    }

    public void setLatitude(Double latitude) {
        this.latitude = latitude;
    }

    public Double getLongitude() {
        return longitude;
    }

    public void setLongitude(Double longitude) {
        this.longitude = longitude;
    }

    public LocalDate getDate() {
        return date;
    }

    public void setDate(LocalDate date) {
        this.date = date;
    }

    public String getModelVersion() {
        return modelVersion;
    }

    public void setModelVersion(String modelVersion) {
        this.modelVersion = modelVersion;
    }

    public Double getGridResolutionDeg() {
        return gridResolutionDeg;
    }

    public void setGridResolutionDeg(Double gridResolutionDeg) {
        this.gridResolutionDeg = gridResolutionDeg;
    }

    public Instant getCreatedAt() {
        return createdAt;
    }

    public void setCreatedAt(Instant createdAt) {
        this.createdAt = createdAt;
    }

    public Instant getCompletedAt() {
        return completedAt;
    }

    public void setCompletedAt(Instant completedAt) {
        this.completedAt = completedAt;
    }

    public String getErrorMessage() {
        return errorMessage;
    }

    public void setErrorMessage(String errorMessage) {
        this.errorMessage = errorMessage;
    }

    public List<DepthPredictionDTO> getPredictions() {
        return predictions;
    }

    public void setPredictions(List<DepthPredictionDTO> predictions) {
        this.predictions = predictions;
    }
}
