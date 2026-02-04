package com.hong.thebaker.dto;

import com.hong.thebaker.entity.PaymentMethod;
import jakarta.validation.constraints.*;
import lombok.Data;

import java.math.BigDecimal;

@Data
public class PaymentRequest {
    @NotBlank(message = "Phone number is required")
    private String phoneNumber;

    @NotNull(message = "Total amount is required")
    @DecimalMin(value = "0.0", inclusive = false, message = "Amount must be positive")
    @DecimalMax(value = "500000", message = "Amount cannot exceed 500,000")
    private BigDecimal totalAmount;

    @NotNull(message = "Payment method is required")
    private PaymentMethod paymentMethod;

    @Min(value = 0, message = "Points cannot be negative")
    private int pointsToUse;
}
