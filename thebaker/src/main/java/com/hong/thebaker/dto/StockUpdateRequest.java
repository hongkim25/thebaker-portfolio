package com.hong.thebaker.dto;

import jakarta.validation.constraints.*;
import lombok.Data;

@Data
public class StockUpdateRequest {
    @NotNull(message = "Product ID is required")
    private Long productId;

    @Min(value = 0, message = "Quantity cannot be negative")
    private int quantity;
}
