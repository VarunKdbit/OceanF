package com.oceanembed.backend.controller;

import com.oceanembed.backend.dto.PredictionRequestDTO;
import com.oceanembed.backend.dto.PredictionResponseDTO;
import com.oceanembed.backend.service.PredictionService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDate;

@RestController
@RequestMapping("/api/v1/predictions")
public class PredictionController {

    private final PredictionService predictionService;

    public PredictionController(PredictionService predictionService) {
        this.predictionService = predictionService;
    }

    /**
     * Create a subsurface temperature prediction.
     * Frontend sends lat/lon/date/surface-variables/depths; this synchronously
     * calls the ML service and returns the stored, completed result.
     */
    @PostMapping
    public ResponseEntity<PredictionResponseDTO> create(@Valid @RequestBody PredictionRequestDTO request) {
        PredictionResponseDTO response = predictionService.createPrediction(request);
        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    @GetMapping("/coverage")
    public ResponseEntity<com.oceanembed.backend.dto.SurfaceCoverageDTO> coverage(
            @RequestParam double latitude,
            @RequestParam double longitude,
            @RequestParam LocalDate date) {
        return ResponseEntity.ok(
                predictionService.getCoverage(latitude, longitude, date));
    }

    /** Retrieve a previously computed prediction job by id. */
    @GetMapping("/{id}")
    public ResponseEntity<PredictionResponseDTO> get(@PathVariable Long id) {
        return ResponseEntity.ok(predictionService.getPrediction(id));
    }
}
