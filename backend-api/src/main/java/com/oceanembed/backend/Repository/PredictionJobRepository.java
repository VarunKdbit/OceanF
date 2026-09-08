package com.oceanembed.backend.repository;

import com.oceanembed.backend.entity.PredictionJob;
import org.springframework.data.jpa.repository.JpaRepository;

public interface PredictionJobRepository extends JpaRepository<PredictionJob, Long> {
}
