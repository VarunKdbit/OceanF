package com.oceanembed.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDate;
import java.util.List;

public class SurfaceCoverageDTO {
    private boolean ready;
    private double latitude;
    private double longitude;
    @JsonProperty("snappedLatitude")
    private Double snappedLatitude;
    @JsonProperty("snappedLongitude")
    private Double snappedLongitude;
    private LocalDate date;
    @JsonProperty("targetDate")
    private LocalDate targetDate;
    @JsonProperty("windowStart")
    private LocalDate windowStart;
    @JsonProperty("windowEnd")
    private LocalDate windowEnd;
    @JsonProperty("availableDates")
    private List<String> availableDates;
    @JsonProperty("missingDates")
    private List<String> missingDates;
    @JsonProperty("requiredVariables")
    private int requiredVariables;
    @JsonProperty("variablesReady")
    private int variablesReady;
    @JsonProperty("readyVariables")
    private List<String> readyVariables;
    @JsonProperty("missingVariables")
    private List<String> missingVariables;
    private String message;

    public boolean isReady() { return ready; }
    public void setReady(boolean ready) { this.ready = ready; }
    public double getLatitude() { return latitude; }
    public void setLatitude(double latitude) { this.latitude = latitude; }
    public double getLongitude() { return longitude; }
    public void setLongitude(double longitude) { this.longitude = longitude; }
    public Double getSnappedLatitude() { return snappedLatitude; }
    public void setSnappedLatitude(Double snappedLatitude) { this.snappedLatitude = snappedLatitude; }
    public Double getSnappedLongitude() { return snappedLongitude; }
    public void setSnappedLongitude(Double snappedLongitude) { this.snappedLongitude = snappedLongitude; }
    public LocalDate getDate() { return date; }
    public void setDate(LocalDate date) { this.date = date; }
    public LocalDate getTargetDate() { return targetDate; }
    public void setTargetDate(LocalDate targetDate) { this.targetDate = targetDate; }
    public LocalDate getWindowStart() { return windowStart; }
    public void setWindowStart(LocalDate windowStart) { this.windowStart = windowStart; }
    public LocalDate getWindowEnd() { return windowEnd; }
    public void setWindowEnd(LocalDate windowEnd) { this.windowEnd = windowEnd; }
    public List<String> getAvailableDates() { return availableDates; }
    public void setAvailableDates(List<String> availableDates) { this.availableDates = availableDates; }
    public List<String> getMissingDates() { return missingDates; }
    public void setMissingDates(List<String> missingDates) { this.missingDates = missingDates; }
    public int getRequiredVariables() { return requiredVariables; }
    public void setRequiredVariables(int requiredVariables) { this.requiredVariables = requiredVariables; }
    public int getVariablesReady() { return variablesReady; }
    public void setVariablesReady(int variablesReady) { this.variablesReady = variablesReady; }
    public List<String> getReadyVariables() { return readyVariables; }
    public void setReadyVariables(List<String> readyVariables) { this.readyVariables = readyVariables; }
    public List<String> getMissingVariables() { return missingVariables; }
    public void setMissingVariables(List<String> missingVariables) { this.missingVariables = missingVariables; }
    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }
}
