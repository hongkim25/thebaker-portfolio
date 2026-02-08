package com.hong.thebaker.controller;

import com.hong.thebaker.dto.OrderRequest;
import com.hong.thebaker.entity.Order;
import com.hong.thebaker.service.OrderService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/orders")
@RequiredArgsConstructor
public class OrderController {

    private final OrderService orderService;

    @PostMapping
    public Order createOrder(@Valid @RequestBody OrderRequest request) {
        log.info("Creating order for phone: {}", request.getPhoneNumber());
        return orderService.createOrder(request);
    }

    @GetMapping
    public List<Order> getAllOrders() {
        return orderService.getAllOrders();
    }

    @PutMapping("/{id}/archive")
    public ResponseEntity<Void> archiveOrder(@PathVariable Long id) {
        log.info("Archiving order: {}", id);
        orderService.archiveOrder(id);
        return ResponseEntity.ok().build();
    }

    @PostMapping("/{id}/cancel")
    public ResponseEntity<String> cancelOrder(@PathVariable Long id) {
        log.info("Cancelling order: {}", id);
        try {
            orderService.cancelOrder(id);
            return ResponseEntity.ok("Order cancelled and stock restored");
        } catch (Exception e) {
            log.error("Failed to cancel order {}: {}", id, e.getMessage());
            return ResponseEntity.badRequest().body(e.getMessage());
        }
    }

    @GetMapping("/search")
    public List<Order> findMyOrders(@RequestParam String phone) {
        return orderService.findMyOrders(phone);
    }

    @PutMapping("/{id}/confirm")
    public ResponseEntity<Void> confirmOrder(@PathVariable Long id) {
        log.info("Confirming order: {}", id);
        orderService.confirmOrder(id);
        return ResponseEntity.ok().build();
    }

    @PutMapping("/{id}/complete")
    public ResponseEntity<Void> completeOrder(@PathVariable Long id) {
        log.info("Completing order: {}", id);
        orderService.completeOrder(id);
        return ResponseEntity.ok().build();
    }

    @GetMapping("/pending-count")
    public ResponseEntity<Long> getPendingCount() {
        return ResponseEntity.ok(orderService.countPendingOrders());
    }

    @GetMapping("/{id}/status")
    public ResponseEntity<String> getOrderStatus(@PathVariable Long id) {
        return ResponseEntity.ok(orderService.getOrderStatus(id));
    }
}
