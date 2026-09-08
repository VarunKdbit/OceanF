package com.oceanembed.backend.service;

import com.oceanembed.backend.dto.*;
import com.oceanembed.backend.entity.PredictionJob;
import com.oceanembed.backend.entity.PredictionResult;
import com.oceanembed.backend.exception.ModelServiceException;
import com.oceanembed.backend.exception.ResourceNotFoundException;
import com.oceanembed.backend.repository.PredictionJobRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

/**
 * Orchestrates a prediction: validates + persists the request, calls the
 * FastAPI ML service for inference, stores the result, and returns it in
 * the frontend-facing contract.
 */
@Service
public class PredictionService {

    private final PredictionJobRepository jobRepository;
    private final FastApiClient fastApiClient;

    public PredictionService(PredictionJobRepository jobRepository, FastApiClient fastApiClient) {
        this.jobRepository = jobRepository;
        this.fastApiClient = fastApiClient;
    }

    @Transactional
    public PredictionResponseDTO createPrediction(PredictionRequestDTO requestDto) {
        PredictionJob job = new PredictionJob();
        job.setLatitude(requestDto.getLatitude());
        job.setLongitude(requestDto.getLongitude());
        job.setRequestDate(requestDto.getDate());
        job.setRegionName(requestDto.getRegionName());
        job.setSst(requestDto.getSurface().getSst());
        job.setSss(requestDto.getSurface().getSss());
        job.setSsh(requestDto.getSurface().getSsh());
        job.setWindU(requestDto.getSurface().getWindU());
        job.setWindV(requestDto.getSurface().getWindV());
        job.setStatus(PredictionJob.JobStatus.PENDING);
        job = jobRepository.save(job);

        try {
            MlPredictionRequest mlRequest = toMlRequest(requestDto);
            MlPredictionResponse mlResponse = fastApiClient.predict(mlRequest);

            job.setModelVersion(mlResponse.getModel_version());
            job.setStatus(PredictionJob.JobStatus.SUCCESS);
            job.setCompletedAt(Instant.now());

            List<PredictionResult> results = new ArrayList<>();
            for (MlDepthPrediction p : mlResponse.getPredictions()) {
                results.add(new PredictionResult(job, p.getDepth_m(), p.getTemperature_c(), p.getUncertainty_c()));
            }
            job.setResults(results);
            job = jobRepository.save(job);

            return toResponseDto(job);

        } catch (ModelServiceException ex) {
            job.setStatus(PredictionJob.JobStatus.FAILED);
            job.setErrorMessage(ex.getMessage());
            job.setCompletedAt(Instant.now());
            jobRepository.save(job);
            throw ex;
        }
    }

    @Transactional(readOnly = true)
    public PredictionResponseDTO getPrediction(Long jobId) {
        PredictionJob job = jobRepository.findById(jobId)
                .orElseThrow(() -> new ResourceNotFoundException("Prediction job " + jobId + " not found"));
        return toResponseDto(job);
    }

    private MlPredictionRequest toMlRequest(PredictionRequestDTO dto) {
        MlPredictionRequest req = new MlPredictionRequest();
        req.setLatitude(dto.getLatitude());
        req.setLongitude(dto.getLongitude());
        req.setDate(dto.getDate());
        req.setSurface(new MlSurfaceVariables(
                dto.getSurface().getSst(),
                dto.getSurface().getSss(),
                dto.getSurface().getSsh(),
                dto.getSurface().getWindU(),
                dto.getSurface().getWindV()
        ));
        if (dto.getDepths() != null && !dto.getDepths().isEmpty()) {
            req.setDepths(dto.getDepths());
        } else {
            req.setDepths(List.of(0, 10, 20, 50, 100, 200, 500));
        }
        return req;
    }

    private PredictionResponseDTO toResponseDto(PredictionJob job) {
        PredictionResponseDTO dto = new PredictionResponseDTO();
        dto.setJobId(job.getId());
        dto.setStatus(job.getStatus().name());
        dto.setLatitude(job.getLatitude());
        dto.setLongitude(job.getLongitude());
        dto.setDate(job.getRequestDate());
        dto.setModelVersion(job.getModelVersion());
        dto.setGridResolutionDeg(0.25);
        dto.setCreatedAt(job.getCreatedAt());
        dto.setCompletedAt(job.getCompletedAt());
        dto.setErrorMessage(job.getErrorMessage());

        List<DepthPredictionDTO> preds = new ArrayList<>();
        for (PredictionResult r : job.getResults()) {
            preds.add(new DepthPredictionDTO(r.getDepthM(), r.getTemperatureC(), r.getUncertaintyC()));
        }
        dto.setPredictions(preds);
        return dto;
    }
}
