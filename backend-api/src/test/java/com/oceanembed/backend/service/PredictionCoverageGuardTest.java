package com.oceanembed.backend.service;

import com.oceanembed.backend.dto.PredictionRequestDTO;
import com.oceanembed.backend.dto.SurfaceCoverageDTO;
import com.oceanembed.backend.entity.PredictionJob;
import com.oceanembed.backend.exception.ModelServiceException;
import com.oceanembed.backend.repository.PredictionJobRepository;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.InjectMocks;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;

import java.time.LocalDate;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

@ExtendWith(MockitoExtension.class)
class PredictionCoverageGuardTest {

    @Mock
    private PredictionJobRepository jobRepository;

    @Mock
    private FastApiClient fastApiClient;

    @InjectMocks
    private PredictionService predictionService;

    @Test
    void incompleteWindowIsRejectedBeforeCreatingPredictionJob() {
        PredictionRequestDTO request = new PredictionRequestDTO();
        request.setLatitude(15.0);
        request.setLongitude(70.0);
        request.setDate(LocalDate.of(2026, 9, 21));
        request.setDepths(List.of(0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000));

        SurfaceCoverageDTO coverage = new SurfaceCoverageDTO();
        coverage.setReady(false);
        coverage.setVariablesReady(0);
        coverage.setRequiredVariables(7);
        coverage.setMissingVariables(List.of("sst", "sss", "sla", "uo", "vo", "u_wind", "v_wind"));
        coverage.setMissingDates(List.of("2026-09-21"));
        coverage.setMessage("Requested target-date input window is not available.");
        when(fastApiClient.checkCoverage(15.0, 70.0, request.getDate()))
                .thenReturn(coverage);

        ModelServiceException error = assertThrows(
                ModelServiceException.class,
                () -> predictionService.createPrediction(request));

        assertEquals(422, error.getStatusCode());
        verify(jobRepository, never()).save(any(PredictionJob.class));
        verify(fastApiClient, never()).predict(any());
    }
}
