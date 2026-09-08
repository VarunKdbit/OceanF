package com.oceanembed.backend.dto;

public class MlSurfaceVariables {
    private Double sst;
    private Double sss;
    private Double ssh;
    private Double wind_u;
    private Double wind_v;

    public MlSurfaceVariables() {}

    public MlSurfaceVariables(Double sst, Double sss, Double ssh, Double windU, Double windV) {
        this.sst = sst;
        this.sss = sss;
        this.ssh = ssh;
        this.wind_u = windU;
        this.wind_v = windV;
    }

    public Double getSst() { return sst; }
    public void setSst(Double sst) { this.sst = sst; }
    public Double getSss() { return sss; }
    public void setSss(Double sss) { this.sss = sss; }
    public Double getSsh() { return ssh; }
    public void setSsh(Double ssh) { this.ssh = ssh; }
    public Double getWind_u() { return wind_u; }
    public void setWind_u(Double wind_u) { this.wind_u = wind_u; }
    public Double getWind_v() { return wind_v; }
    public void setWind_v(Double wind_v) { this.wind_v = wind_v; }
}
