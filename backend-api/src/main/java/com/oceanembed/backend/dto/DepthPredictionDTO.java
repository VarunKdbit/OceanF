package com.oceanembed.backend.dto;

public class DepthPredictionDTO {
    private Integer depthM;
    private Double temperatureC;
    private Double uncertaintyC;

    public DepthPredictionDTO() {}

    public DepthPredictionDTO(Integer depthM, Double temperatureC, Double uncertaintyC) {
        this.depthM = depthM;
        this.temperatureC = temperatureC;
        this.uncertaintyC = uncertaintyC;
    }

    public Integer getDepthM() { return depthM; }
    public void setDepthM(Integer depthM) { this.depthM = depthM; }
    public Double getTemperatureC() { return temperatureC; }
    public void setTemperatureC(Double temperatureC) { this.temperatureC = temperatureC; }
    public Double getUncertaintyC() { return uncertaintyC; }
    public void setUncertaintyC(Double uncertaintyC) { this.uncertaintyC = uncertaintyC; }
}
