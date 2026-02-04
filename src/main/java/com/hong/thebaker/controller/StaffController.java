package com.hong.thebaker.controller;

import com.hong.thebaker.dto.AddPointsRequest;
import com.hong.thebaker.dto.ShopStatusRequest;
import com.hong.thebaker.dto.StockUpdateRequest;
import com.hong.thebaker.entity.Customer;
import com.hong.thebaker.entity.Order;
import com.hong.thebaker.entity.OrderStatus;
import com.hong.thebaker.repository.CustomerRepository;
import com.hong.thebaker.repository.OrderRepository;
import com.hong.thebaker.repository.ProductRepository;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Sort;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.math.BigDecimal;
import java.time.DayOfWeek;
import java.time.LocalDateTime;
import java.time.LocalTime;
import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@RestController
@RequestMapping("/api/staff")
@RequiredArgsConstructor
public class StaffController {

    private final CustomerRepository customerRepo;
    private final OrderRepository orderRepo;
    private final ProductRepository productRepo;

    // Shop configuration (in production, this should be in database)
    private static final LocalTime OPEN_TIME = LocalTime.of(9, 0);
    private static final LocalTime CLOSE_TIME = LocalTime.of(17, 0);
    private static boolean forceOpen = false;
    private static boolean forceClosed = false;
    private static boolean reservationAvailable = false;

    @GetMapping("/status")
    public ResponseEntity<?> getStatus() {
        boolean isShopOpen;
        if (forceOpen) isShopOpen = true;
        else if (forceClosed) isShopOpen = false;
        else isShopOpen = checkSchedule();

        return ResponseEntity.ok(Map.of("open", isShopOpen, "reservationOpen", reservationAvailable));
    }

    @PostMapping("/status")
    public ResponseEntity<?> toggleStatus(@Valid @RequestBody ShopStatusRequest request) {
        log.info("Shop status changed to: {}", request.isOpen());
        if (request.isOpen()) {
            forceOpen = true;
            forceClosed = false;
        } else {
            forceOpen = false;
            forceClosed = true;
        }
        return ResponseEntity.ok(Map.of("message", "Status updated", "open", request.isOpen()));
    }

    @PostMapping("/reservation-status")
    public ResponseEntity<?> toggleReservation(@Valid @RequestBody ShopStatusRequest request) {
        log.info("Reservation status changed to: {}", request.isOpen());
        reservationAvailable = request.isOpen();
        return ResponseEntity.ok(Map.of("message", "Reservation updated", "reservationOpen", reservationAvailable));
    }

    private boolean checkSchedule() {
        LocalDateTime now = LocalDateTime.now();
        LocalTime time = now.toLocalTime();
        if (now.getDayOfWeek() == DayOfWeek.TUESDAY) return false;
        return time.isAfter(OPEN_TIME) && time.isBefore(CLOSE_TIME);
    }

    @GetMapping("/search")
    public List<Order> findMyOrders(@RequestParam(required = false) String phone) {
        if (phone != null && !phone.isEmpty()) {
            return orderRepo.findByCustomerPhoneOrderByOrderDateDesc(phone);
        }
        return orderRepo.findAll(Sort.by(Sort.Direction.DESC, "orderDate"));
    }

    @PostMapping("/points")
    public ResponseEntity<?> addPointsManually(@Valid @RequestBody AddPointsRequest request) {
        log.info("Adding points manually for phone: {}", request.getPhoneNumber());

        Customer customer = customerRepo.findByPhone(request.getPhoneNumber()).orElseGet(() -> {
            Customer newCustomer = new Customer();
            newCustomer.setName("Walk-in " + request.getPhoneNumber());
            newCustomer.setPhone(request.getPhoneNumber());
            newCustomer.setPoints(0);
            return customerRepo.save(newCustomer);
        });

        Order order = new Order();
        order.setCustomer(customer);
        order.setOrderDate(LocalDateTime.now());
        order.setStatus(OrderStatus.COMPLETED);
        order.setTotalAmount(request.getAmount());

        int pointsToAdd = request.getAmount().multiply(BigDecimal.valueOf(0.05)).intValue();
        order.setPointsEarned(pointsToAdd);

        customer.setPoints(customer.getPoints() + pointsToAdd);
        customerRepo.save(customer);
        orderRepo.save(order);

        log.info("Points added for {}: {} points, new balance: {}",
                request.getPhoneNumber(), pointsToAdd, customer.getPoints());

        return ResponseEntity.ok(Map.of("message", "Points Added", "currentPoints", customer.getPoints()));
    }

    @PostMapping("/stock")
    public ResponseEntity<?> updateStock(@Valid @RequestBody StockUpdateRequest request) {
        log.info("Updating stock for product {}: quantity {}", request.getProductId(), request.getQuantity());

        return productRepo.findById(request.getProductId()).map(product -> {
            product.setStockQuantity(request.getQuantity());
            productRepo.save(product);
            return ResponseEntity.ok(Map.of("message", "Stock updated"));
        }).orElse(ResponseEntity.badRequest().body(Map.of("error", "Product not found")));
    }

    @GetMapping("/history")
    public List<Order> getStaffHistory(@RequestParam String phone) {
        return orderRepo.findByCustomerPhoneOrderByOrderDateDesc(phone);
    }

    @PostMapping("/orders/{id}/revert")
    public ResponseEntity<?> revertMistake(@PathVariable Long id) {
        log.info("Reverting order: {}", id);

        return orderRepo.findById(id).map(order -> {
            Customer customer = order.getCustomer();
            customer.setPoints(Math.max(0, customer.getPoints() - order.getPointsEarned()));
            customerRepo.save(customer);
            order.setStatus(OrderStatus.CANCELLED);
            orderRepo.save(order);

            log.info("Order {} reverted, customer {} points adjusted", id, customer.getPhone());
            return ResponseEntity.ok(Map.of("message", "Reverted"));
        }).orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/customer/my-points")
    public ResponseEntity<?> getCustomerPoints(@RequestParam String phone) {
        return customerRepo.findByPhone(phone).map(customer -> {
            List<Map<String, Object>> history = orderRepo.findByCustomerPhoneOrderByOrderDateDesc(phone)
                    .stream()
                    .filter(o -> o.getStatus() != OrderStatus.CANCELLED)
                    .limit(5)
                    .map(o -> {
                        Map<String, Object> map = new HashMap<>();
                        map.put("date", o.getOrderDate());
                        map.put("points", o.getPointsEarned());
                        map.put("used", o.getPointsUsed());
                        return map;
                    }).collect(Collectors.toList());

            return ResponseEntity.ok(Map.of("totalPoints", customer.getPoints(), "history", history));
        }).orElse(ResponseEntity.badRequest().body(Map.of("error", "Customer not found")));
    }
}
