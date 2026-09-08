package com.oceanembed.backend.controller;

import com.oceanembed.backend.service.FastApiClient;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
public class HealthController {

    private final FastApiClient fastApiClient;

    public HealthController(FastApiClient fastApiClient) {
        this.fastApiClient = fastApiClient;
    }

    @GetMapping("/api/v1/health")
    public Map<String, Object> health() {
        boolean mlHealthy = fastApiClient.isHealthy();
        return Map.of(
                "status", "ok",
                "ml_service_reachable", mlHealthy
        );
    }
}
