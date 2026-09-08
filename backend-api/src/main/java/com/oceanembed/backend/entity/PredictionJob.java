package com.oceanembed.backend.entity;

import jakarta.persistence.*;
import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;

@Entity
@Table(name = "prediction_job")
public class PredictionJob {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private Double latitude;
    private Double longitude;
    private LocalDate requestDate;

    // Surface variables snapshot, stored for traceability/reproducibility
    private Double sst;
    private Double sss;
    private Double ssh;
    private Double windU;
    private Double windV;

    @Column(name = "region_name")
    private String regionName;

    @Enumerated(EnumType.STRING)
    private JobStatus status;

    private String modelVersion;
    private String errorMessage;

    private Instant createdAt;
    private Instant completedAt;

    @OneToMany(mappedBy = "job", cascade = CascadeType.ALL, orphanRemoval = true)
    private List<PredictionResult> results = new ArrayList<>();

    public enum JobStatus { PENDING, SUCCESS, FAILED }

    @PrePersist
    protected void onCreate() {
        this.createdAt = Instant.now();
        if (this.status == null) {
            this.status = JobStatus.PENDING;
        }
    }

    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    public Double getLatitude() { return latitude; }
    public void setLatitude(Double latitude) { this.latitude = latitude; }
    public Double getLongitude() { return longitude; }
    public void setLongitude(Double longitude) { this.longitude = longitude; }
    public LocalDate getRequestDate() { return requestDate; }
    public void setRequestDate(LocalDate requestDate) { this.requestDate = requestDate; }
    public Double getSst() { return sst; }
    public void setSst(Double sst) { this.sst = sst; }
    public Double getSss() { return sss; }
    public void setSss(Double sss) { this.sss = sss; }
    public Double getSsh() { return ssh; }
    public void setSsh(Double ssh) { this.ssh = ssh; }
    public Double getWindU() { return windU; }
    public void setWindU(Double windU) { this.windU = windU; }
    public Double getWindV() { return windV; }
    public void setWindV(Double windV) { this.windV = windV; }
    public String getRegionName() { return regionName; }
    public void setRegionName(String regionName) { this.regionName = regionName; }
    public JobStatus getStatus() { return status; }
    public void setStatus(JobStatus status) { this.status = status; }
    public String getModelVersion() { return modelVersion; }
    public void setModelVersion(String modelVersion) { this.modelVersion = modelVersion; }
    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }
    public Instant getCreatedAt() { return createdAt; }
    public void setCreatedAt(Instant createdAt) { this.createdAt = createdAt; }
    public Instant getCompletedAt() { return completedAt; }
    public void setCompletedAt(Instant completedAt) { this.completedAt = completedAt; }
    public List<PredictionResult> getResults() { return results; }
    public void setResults(List<PredictionResult> results) { this.results = results; }
}
