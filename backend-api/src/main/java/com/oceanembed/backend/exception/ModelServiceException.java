package com.oceanembed.backend.exception;

/** Thrown when the FastAPI ML service is unreachable or returns an error/invalid response. */
public class ModelServiceException extends RuntimeException {
    public ModelServiceException(String message) {
        super(message);
    }

    public ModelServiceException(String message, Throwable cause) {
        super(message, cause);
    }
}
