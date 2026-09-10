package com.oceanembed.backend.dto;

/**
 * Optional frontend metadata only. These values are never treated as the
 * complete OceanEmbed model input.
 */
public class SurfaceVariablesDTO {
    private Double sst;
    private Double sss;
    private Double sla;
    private Double uo;
    private Double vo;
    private Double windU;
    private Double windV;

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
    public Double getWindU() { return windU; }
    public void setWindU(Double windU) { this.windU = windU; }
    public Double getWindV() { return windV; }
    public void setWindV(Double windV) { this.windV = windV; }
}
