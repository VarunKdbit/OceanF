package com.oceanembed.backend.dto;

import jakarta.validation.constraints.NotNull;

/**
 * Satellite-observed surface inputs required by the OceanEmbed model:
 * SST (Sea Surface Temperature), SSS (Sea Surface Salinity),
 * SSH/SLA (Sea Surface Height / Sea Level Anomaly), and surface wind.
 */
public class SurfaceVariablesDTO {

    @NotNull
    private Double sst;

    @NotNull
    private Double sss;

    @NotNull
    private Double ssh;

    private Double windU;
    private Double windV;

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
}
