package com.oceanembed.backend.dto;

public class MlDepthPrediction {
    private Integer depth_m;
    private Double temperature_c;
    private Double uncertainty_c;

    public Integer getDepth_m() { return depth_m; }
    public void setDepth_m(Integer depth_m) { this.depth_m = depth_m; }
    public Double getTemperature_c() { return temperature_c; }
    public void setTemperature_c(Double temperature_c) { this.temperature_c = temperature_c; }
    public Double getUncertainty_c() { return uncertainty_c; }
    public void setUncertainty_c(Double uncertainty_c) { this.uncertainty_c = uncertainty_c; }
}
