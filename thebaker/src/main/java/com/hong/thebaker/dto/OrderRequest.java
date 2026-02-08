package com.hong.thebaker.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.hong.thebaker.entity.PaymentMethod;
import jakarta.validation.Valid;
import jakarta.validation.constraints.*;
import lombok.Data;

import java.util.List;

@Data
public class OrderRequest {
    private Long customerId;
    private String customerName;

    @NotBlank(message = "Phone number is required")
    @Pattern(regexp = "^[0-9]{10,11}$", message = "Invalid phone number format")
    private String phoneNumber;

    @Size(max = 500, message = "Memo cannot exceed 500 characters")
    private String memo;

    @NotEmpty(message = "Order must contain at least one item")
    @Valid
    private List<OrderItemRequest> items;

    private boolean marketingConsent;

    @Min(value = 0, message = "Points cannot be negative")
    private int pointsToUse;

    private String pickupTime;
    private PaymentMethod paymentMethod;

    @JsonProperty("Takeaway")
    private boolean isTakeaway;

    @JsonProperty("wantsCut")
    private boolean wantsCut;

    @Data
    public static class OrderItemRequest {
        @NotNull(message = "Product ID is required")
        private Long productId;

        @Min(value = 1, message = "Quantity must be at least 1")
        private int quantity;
    }
}
