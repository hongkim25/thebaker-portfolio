package com.hong.thebaker.controller;

import com.hong.thebaker.dto.PaymentRequest;
import com.hong.thebaker.entity.Customer;
import com.hong.thebaker.entity.Order;
import com.hong.thebaker.entity.OrderStatus;
import com.hong.thebaker.repository.CustomerRepository;
import com.hong.thebaker.repository.OrderRepository;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.LocalDateTime;

@Slf4j
@RestController
@RequestMapping("/api/points")
@RequiredArgsConstructor
public class PointController {

    private final CustomerRepository customerRepo;
    private final OrderRepository orderRepo;

    @PostMapping("/pay")
    public ResponseEntity<?> processPayment(@Valid @RequestBody PaymentRequest request) {
        log.info("Processing payment for phone: {}", request.getPhoneNumber());

        String phone = request.getPhoneNumber();
        BigDecimal totalAmount = request.getTotalAmount();
        int pointsToUse = request.getPointsToUse();

        Customer customer = customerRepo.findByPhone(phone)
                .orElseGet(() -> {
                    Customer newCustomer = new Customer();
                    String lastFour = phone.length() > 4 ? phone.substring(phone.length() - 4) : phone;
                    newCustomer.setName("Guest " + lastFour);
                    newCustomer.setPhone(phone);
                    newCustomer.setPoints(0);
                    log.info("Created new customer: {}", phone);
                    return customerRepo.save(newCustomer);
                });

        if (pointsToUse > 0 && customer.getPoints() < pointsToUse) {
            log.warn("Insufficient points for customer {}: requested {}, available {}",
                    phone, pointsToUse, customer.getPoints());
            return ResponseEntity.badRequest().body("포인트 부족! (보유: " + customer.getPoints() + "P)");
        }

        BigDecimal realPaidAmount = totalAmount.subtract(BigDecimal.valueOf(pointsToUse));
        if (realPaidAmount.compareTo(BigDecimal.ZERO) < 0) {
            realPaidAmount = BigDecimal.ZERO;
        }

        int pointsToAdd = request.getPaymentMethod().calculatePoints(realPaidAmount);

        int newBalance = customer.getPoints() - pointsToUse + pointsToAdd;
        customer.setPoints(newBalance);
        customerRepo.save(customer);

        Order order = new Order();
        order.setCustomer(customer);
        order.setOrderDate(LocalDateTime.now());
        order.setTotalAmount(totalAmount);
        order.setPointsUsed(pointsToUse);
        order.setPointsEarned(pointsToAdd);
        order.setStatus(OrderStatus.COMPLETED);
        orderRepo.save(order);

        log.info("Payment completed for {}: used {}P, earned {}P, balance {}P",
                phone, pointsToUse, pointsToAdd, newBalance);

        return ResponseEntity.ok(String.format("사용: %dP | 적립: %dP | 잔액: %dP",
                pointsToUse, pointsToAdd, newBalance));
    }
}
