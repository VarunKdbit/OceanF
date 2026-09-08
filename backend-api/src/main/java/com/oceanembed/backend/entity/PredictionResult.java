package com.oceanembed.backend.entity;

import jakarta.persistence.*;

@Entity
@Table(name = "prediction_result")
public class PredictionResult {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "job_id", nullable = false)
    private PredictionJob job;

    private Integer depthM;
    private Double temperatureC;
    private Double uncertaintyC;

    public PredictionResult() {}

    public PredictionResult(PredictionJob job, Integer depthM, Double temperatureC, Double uncertaintyC) {
        this.job = job;
        this.depthM = depthM;
        this.temperatureC = temperatureC;
        this.uncertaintyC = uncertaintyC;
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public PredictionJob getJob() { return job; }
    public void setJob(PredictionJob job) { this.job = job; }
    public Integer getDepthM() { return depthM; }
    public void setDepthM(Integer depthM) { this.depthM = depthM; }
    public Double getTemperatureC() { return temperatureC; }
    public void setTemperatureC(Double temperatureC) { this.temperatureC = temperatureC; }
    public Double getUncertaintyC() { return uncertaintyC; }
    public void setUncertaintyC(Double uncertaintyC) { this.uncertaintyC = uncertaintyC; }
}
