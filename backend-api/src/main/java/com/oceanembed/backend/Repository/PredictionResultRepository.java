package com.oceanembed.backend.repository;

import com.oceanembed.backend.entity.PredictionResult;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PredictionResultRepository extends JpaRepository<PredictionResult, Long> {
}
