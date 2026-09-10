package com.oceanembed.backend.dto;

/**
 * Optional metadata representation of the seven trained OceanEmbed inputs.
 * The ML service does not use these values to build the tensor; it loads the
 * authoritative spatial window from the harmonized datasets.
 */
public class MlSurfaceVariables {
    private Double sst;
    private Double sss;
    private Double sla;
    private Double uo;
    private Double vo;
    private Double u_wind;
    private Double v_wind;

    public MlSurfaceVariables() {}

    public Double getSst() { return sst; }
    public void setSst(Double sst) { this.sst = sst; }
    public Double getSss() { return sss; }
    public void setSss(Double sss) { this.sss = sss; }
    public Double getSla() { return sla; }
    public void setSla(Double sla) { this.sla = sla; }
    public Double getUo() { return uo; }
    public void setUo(Double uo) { this.uo = uo; }
    public Double getVo() { return vo; }
    public void setVo(Double vo) { this.vo = vo; }
    public Double getWind_u() { return u_wind; }
    public void setWind_u(Double wind_u) { this.u_wind = wind_u; }
    public Double getWind_v() { return v_wind; }
    public void setWind_v(Double wind_v) { this.v_wind = wind_v; }
}
