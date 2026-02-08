package com.hong.thebaker.dto;

import jakarta.validation.constraints.*;
import lombok.Data;

import java.math.BigDecimal;

@Data
public class AddPointsRequest {
    @NotBlank(message = "Phone number is required")
    private String phoneNumber;

    @NotNull(message = "Amount is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "Amount must be positive")
    @DecimalMax(value = "500000", message = "Amount cannot exceed 500,000")
    private BigDecimal amount;
}
