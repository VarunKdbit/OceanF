package com.oceanembed.backend.service;

import com.oceanembed.backend.dto.*;
import com.oceanembed.backend.entity.PredictionJob;
import com.oceanembed.backend.entity.PredictionResult;
import com.oceanembed.backend.entity.SurfaceObservationResult;
import com.oceanembed.backend.exception.ModelServiceException;
import com.oceanembed.backend.exception.ResourceNotFoundException;
import com.oceanembed.backend.repository.PredictionJobRepository;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

@Service
public class PredictionService {

    private static final Set<Integer> SUPPORTED_DEPTHS = Set.of(
        0, 5, 10, 20, 30, 50, 75, 100,
        125, 150, 200, 300, 500, 700, 1000
    );

    private final PredictionJobRepository jobRepository;
    private final FastApiClient fastApiClient;

    public PredictionService(
            PredictionJobRepository jobRepository,
            FastApiClient fastApiClient) {
        this.jobRepository = jobRepository;
        this.fastApiClient = fastApiClient;
    }

    @Transactional
    public PredictionResponseDTO createPrediction(PredictionRequestDTO requestDto) {
        validateDepths(requestDto.getDepths());
        SurfaceCoverageDTO coverage = fastApiClient.checkCoverage(
                requestDto.getLatitude(),
                requestDto.getLongitude(),
                requestDto.getDate());
        if (!coverage.isReady()) {
            List<String> missingDates = coverage.getMissingDates() == null
                    ? List.of()
                    : coverage.getMissingDates();
            throw new ModelServiceException(
                    "INCOMPLETE_DATA: " + coverage.getMessage()
                            + "; missing variables: "
                            + String.join(", ", coverage.getMissingVariables())
                            + (missingDates.isEmpty()
                                    ? ""
                                    : "; missing dates: " + String.join(", ", missingDates)),
                    422
            );
        }

        LocalDate inputWindowStart = requestDto.getDate().minusDays(6);
        LocalDate inputWindowEnd = requestDto.getDate();

        PredictionJob job = new PredictionJob();
        job.setLatitude(requestDto.getLatitude());
        job.setLongitude(requestDto.getLongitude());
        job.setRequestDate(requestDto.getDate());
        job.setInputWindowStart(inputWindowStart);
        job.setInputWindowEnd(inputWindowEnd);
        job.setRegionName(requestDto.getRegionName());

        // Keep legacy surface fields only as request metadata if supplied.
        if (requestDto.getSurface() != null) {
            job.setSst(requestDto.getSurface().getSst());
            job.setSss(requestDto.getSurface().getSss());
            job.setSsh(requestDto.getSurface().getSla());
            job.setWindU(requestDto.getSurface().getWindU());
            job.setWindV(requestDto.getSurface().getWindV());
        }

        job.setStatus(PredictionJob.JobStatus.PENDING);
        job = jobRepository.save(job);

        try {
            MlPredictionRequest mlRequest = toMlRequest(requestDto);
            MlPredictionResponse mlResponse = fastApiClient.predict(mlRequest);

            job.setModelVersion(mlResponse.getModelVersion());
            if (mlResponse.getInputWindowStart() != null) {
                job.setInputWindowStart(mlResponse.getInputWindowStart());
            }
            if (mlResponse.getInputWindowEnd() != null) {
                job.setInputWindowEnd(mlResponse.getInputWindowEnd());
            }
            job.setStatus(PredictionJob.JobStatus.SUCCESS);
            job.setCompletedAt(Instant.now());

            List<PredictionResult> results = new ArrayList<>();
            for (MlDepthPrediction p : mlResponse.getPredictions()) {
                results.add(new PredictionResult(
                    job,
                    p.getDepth_m(),
                    p.getTemperature_c(),
                    p.getUncertainty_c()
                ));
            }

            job.setResults(results);
            List<SurfaceObservationResult> surfaceObservations = new ArrayList<>();
            if (mlResponse.getSurfaceObservations() != null) {
                for (MlSurfaceObservation observation : mlResponse.getSurfaceObservations()) {
                    surfaceObservations.add(new SurfaceObservationResult(
                        job,
                        observation.getVariable(),
                        observation.getValue(),
                        observation.getUnit()
                    ));
                }
            }
            job.setSurfaceObservations(surfaceObservations);
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

    public SurfaceCoverageDTO getCoverage(
            double latitude,
            double longitude,
            LocalDate date) {
        return fastApiClient.checkCoverage(latitude, longitude, date);
    }

    @Transactional(readOnly = true)
    public PredictionResponseDTO getPrediction(Long jobId) {
        PredictionJob job = jobRepository.findById(jobId)
            .orElseThrow(() ->
                new ResourceNotFoundException("Prediction job " + jobId + " not found"));
        return toResponseDto(job);
    }

    private void validateDepths(List<Integer> depths) {
        if (depths == null || depths.isEmpty()) {
            return;
        }
        List<Integer> unsupported = depths.stream()
            .filter(d -> !SUPPORTED_DEPTHS.contains(d))
            .distinct()
            .sorted()
            .toList();

        if (!unsupported.isEmpty()) {
            throw new IllegalArgumentException(
                "Unsupported depth(s): " + unsupported +
                ". Supported depths: " + SUPPORTED_DEPTHS
            );
        }
    }

    private MlPredictionRequest toMlRequest(PredictionRequestDTO dto) {
        MlPredictionRequest req = new MlPredictionRequest();
        req.setLatitude(dto.getLatitude());
        req.setLongitude(dto.getLongitude());
        req.setDate(dto.getDate());

        if (dto.getSurface() != null) {
            MlSurfaceVariables surface = new MlSurfaceVariables();
            surface.setSst(dto.getSurface().getSst());
            surface.setSss(dto.getSurface().getSss());
            surface.setSla(dto.getSurface().getSla());
            surface.setUo(dto.getSurface().getUo());
            surface.setVo(dto.getSurface().getVo());
            surface.setWind_u(dto.getSurface().getWindU());
            surface.setWind_v(dto.getSurface().getWindV());
            req.setSurface(surface);
        }

        if (dto.getDepths() != null && !dto.getDepths().isEmpty()) {
            req.setDepths(dto.getDepths());
        } else {
            req.setDepths(List.of(
                0, 5, 10, 20, 30, 50, 75, 100,
                125, 150, 200, 300, 500, 700, 1000
            ));
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
        dto.setInputWindowStart(job.getInputWindowStart());
        dto.setInputWindowEnd(job.getInputWindowEnd());
        dto.setModelVersion(job.getModelVersion());
        dto.setGridResolutionDeg(0.25);
        dto.setCreatedAt(job.getCreatedAt());
        dto.setCompletedAt(job.getCompletedAt());
        dto.setErrorMessage(job.getErrorMessage());

        List<DepthPredictionDTO> preds = new ArrayList<>();
        for (PredictionResult r : job.getResults()) {
            preds.add(new DepthPredictionDTO(
                r.getDepthM(),
                r.getTemperatureC(),
                r.getUncertaintyC()
            ));
        }
        dto.setPredictions(preds);

        List<SurfaceObservationDTO> surfaceObservations = new ArrayList<>();
        for (SurfaceObservationResult observation : job.getSurfaceObservations()) {
            surfaceObservations.add(new SurfaceObservationDTO(
                observation.getVariable(),
                observation.getValue(),
                observation.getUnit()
            ));
        }
        dto.setSurfaceObservations(surfaceObservations);
        return dto;
    }
}
