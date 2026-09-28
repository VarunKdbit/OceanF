package com.oceanembed.backend.service;

import com.oceanembed.backend.dto.MlPredictionRequest;
import com.oceanembed.backend.dto.MlPredictionResponse;
import com.oceanembed.backend.dto.SurfaceCoverageDTO;
import com.oceanembed.backend.exception.ModelServiceException;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestClientResponseException;
import org.springframework.web.client.RestTemplate;
import org.springframework.web.util.UriComponentsBuilder;

import java.net.URI;
import java.time.LocalDate;

/**
 * Talks to the FastAPI ML service ("ml-service").
 * This is the ONLY class in backend-api that knows the ML service's HTTP contract.
 */
@Component
public class FastApiClient {

    private final RestTemplate restTemplate;
    private final String baseUrl;

    public FastApiClient(
            RestTemplate restTemplate,
            @Value("${ml-service.base-url}") String baseUrl) {
        this.restTemplate = restTemplate;
        this.baseUrl = baseUrl;
    }

    public MlPredictionResponse predict(MlPredictionRequest request) {
        try {
            MlPredictionResponse response = restTemplate.postForObject(
                    baseUrl + "/predict",
                    request,
                    MlPredictionResponse.class
            );

            if (response == null) {
                throw new ModelServiceException(
                        "ML service returned an empty response"
                );
            }

            return response;

        } catch (RestClientResponseException ex) {
            int statusCode = ex.getStatusCode().value();

            String responseBody = ex.getResponseBodyAsString();

            String message;

            if (responseBody != null && !responseBody.isBlank()) {
                message = responseBody;
            } else {
                message = ex.getMessage();
            }

            throw new ModelServiceException(
                    "OceanEmbed ML service returned HTTP "
                            + statusCode + ": " + message,
                    statusCode
            );

        } catch (RestClientException ex) {
            throw new ModelServiceException(
                    "Failed to reach OceanEmbed ML service: "
                            + ex.getMessage(),
                    ex
            );
        }
    }

    public SurfaceCoverageDTO checkCoverage(
            double latitude,
            double longitude,
            LocalDate date) {
        URI uri = UriComponentsBuilder.fromHttpUrl(baseUrl)
                .path("/coverage")
                .queryParam("latitude", latitude)
                .queryParam("longitude", longitude)
                .queryParam("date", date)
                .build()
                .encode()
                .toUri();
        try {
            SurfaceCoverageDTO response =
                    restTemplate.getForObject(uri, SurfaceCoverageDTO.class);
            if (response == null) {
                throw new ModelServiceException(
                        "ML service returned an empty coverage response");
            }
            return response;
        } catch (RestClientResponseException ex) {
            int statusCode = ex.getStatusCode().value();
            String responseBody = ex.getResponseBodyAsString();
            String message = responseBody == null || responseBody.isBlank()
                    ? ex.getMessage()
                    : responseBody;
            throw new ModelServiceException(
                    "OceanEmbed ML coverage check returned HTTP "
                            + statusCode + ": " + message,
                    statusCode
            );
        } catch (RestClientException ex) {
            throw new ModelServiceException(
                    "Failed to reach OceanEmbed ML coverage endpoint: "
                            + ex.getMessage(),
                    ex
            );
        }
    }

    public boolean isHealthy() {
        try {
            restTemplate.getForObject(
                    baseUrl + "/health",
                    String.class
            );
            return true;

        } catch (RestClientException ex) {
            return false;
        }
    }
}