package com.hong.thebaker.service;

import com.hong.thebaker.dto.OrderRequest;
import com.hong.thebaker.entity.*;
import com.hong.thebaker.repository.CustomerRepository;
import com.hong.thebaker.repository.OrderRepository;
import com.hong.thebaker.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.math.BigDecimal;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@Transactional
@RequiredArgsConstructor
public class OrderService {

    private final ProductRepository productRepository;
    private final OrderRepository orderRepository;
    private final CustomerRepository customerRepository;

    public Order createOrder(OrderRequest request) {
        log.info("Creating order for customer: {}", request.getPhoneNumber());

        Customer customer = customerRepository.findByPhone(request.getPhoneNumber())
                .orElseGet(() -> {
                    Customer newCustomer = new Customer();
                    newCustomer.setPhone(request.getPhoneNumber());
                    newCustomer.setPoints(0);
                    return newCustomer;
                });

        // Update Name
        if (request.getCustomerName() != null && !request.getCustomerName().isEmpty()) {
            customer.setName(request.getCustomerName());
        } else if (customer.getName() == null) {
            customer.setName("Guest");
        }

        customer.setMarketingConsent(request.isMarketingConsent());
        customerRepository.save(customer);

        Order order = new Order();
        order.setCustomer(customer);
        // Seoul Time
        order.setOrderDate(java.time.ZonedDateTime.now(java.time.ZoneId.of("Asia/Seoul")).toLocalDateTime());

        order.setStatus(OrderStatus.PENDING);

        order.setMemo(request.getMemo());
        order.setPickupTime(request.getPickupTime());
        order.setTakeaway(request.isTakeaway());
        order.setWantsCut(request.isWantsCut());

        BigDecimal totalAmount = BigDecimal.ZERO;
        List<OrderItem> orderItems = new ArrayList<>();

        // 2. Loop through items & Reduce Stock
        for (OrderRequest.OrderItemRequest itemRequest : request.getItems()) {
            Product product = productRepository.findById(itemRequest.getProductId())
                    .orElseThrow(() -> new RuntimeException("상품을 찾을 수 없습니다."));

            if (product.getStockQuantity() < itemRequest.getQuantity()) {
                throw new RuntimeException("재고가 충분하지 않습니다: " + product.getName());
            }

            product.setStockQuantity(product.getStockQuantity() - itemRequest.getQuantity());
            productRepository.save(product);

            OrderItem orderItem = new OrderItem();
            orderItem.setOrder(order);
            orderItem.setProduct(product);
            orderItem.setQuantity(itemRequest.getQuantity());
            orderItem.setPriceAtPurchase(product.getPrice());

            orderItems.add(orderItem);

            BigDecimal lineItemTotal = product.getPrice().multiply(new BigDecimal(itemRequest.getQuantity()));
            totalAmount = totalAmount.add(lineItemTotal);
        }

        order.setItems(orderItems);
        order.setTotalAmount(totalAmount);

        // 3. Handle Point Usage
        int pointsToUse = request.getPointsToUse();
        if (pointsToUse > 0) {
            if (customer.getPoints() < pointsToUse) {
                throw new RuntimeException("포인트가 부족합니다. 보유 포인트: " + customer.getPoints());
            }
            customer.setPoints(customer.getPoints() - pointsToUse);
        }

        order.setPointsUsed(pointsToUse);
        Order savedOrder = orderRepository.save(order);
        log.info("Order created: id={}, total={}, items={}",
                savedOrder.getId(), totalAmount, orderItems.size());
        return savedOrder;
    }

    public void confirmOrder(Long orderId) {
        log.info("Confirming order: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new RuntimeException("Order not found"));

        if (order.getStatus() != OrderStatus.PENDING) {
            throw new RuntimeException("Only PENDING orders can be confirmed.");
        }

        order.setStatus(OrderStatus.PROCESSING);
        orderRepository.save(order);
    }

    public void completeOrder(Long orderId) {
        log.info("Completing order: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new RuntimeException("Order not found"));

        if (order.getStatus() == OrderStatus.COMPLETED) return;

        Customer customer = order.getCustomer();
        BigDecimal total = order.getTotalAmount();
        int pointsEarned = total.multiply(new BigDecimal("0.03")).intValue();

        order.setPointsEarned(pointsEarned);
        customer.setPoints(customer.getPoints() + pointsEarned);
        customerRepository.save(customer);

        order.setStatus(OrderStatus.COMPLETED);
        orderRepository.save(order);
    }

    public Long countPendingOrders() {
        return orderRepository.countByStatus(OrderStatus.PENDING);
    }

    @Transactional
    public void cancelOrder(Long orderId) {
        log.info("Cancelling order: {}", orderId);
        Order order = orderRepository.findById(orderId)
                .orElseThrow(() -> new RuntimeException("Order not found"));

        if (order.getStatus() == OrderStatus.CANCELLED) {
            throw new RuntimeException("Already cancelled");
        }

        for (OrderItem item : order.getItems()) {
            Product product = item.getProduct();
            product.setStockQuantity(product.getStockQuantity() + item.getQuantity());
            productRepository.save(product);
        }

        order.setStatus(OrderStatus.CANCELLED);
        order.setCancelledDate(LocalDateTime.now());
        orderRepository.save(order);
    }

    public List<Order> getAllOrders() {
        return orderRepository.findAll(Sort.by(Sort.Direction.DESC, "orderDate"));
    }

    public void processQuickPayment(com.hong.thebaker.dto.QuickPaymentRequest request) {
        Customer customer = customerRepository.findByPhone(request.getPhoneNumber())
                .orElseGet(() -> {
                    Customer newCustomer = new Customer();
                    newCustomer.setPhone(request.getPhoneNumber());
                    newCustomer.setName("Guest");
                    newCustomer.setPoints(0);
                    return customerRepository.save(newCustomer);
                });

        if (request.getPointsToUse() > 0) {
            if (customer.getPoints() < request.getPointsToUse()) {
                throw new RuntimeException("포인트 부족");
            }
            customer.setPoints(customer.getPoints() - request.getPointsToUse());
        }

        BigDecimal totalAmountBd = BigDecimal.valueOf(request.getTotalAmount());
        BigDecimal pointsUsedBd = BigDecimal.valueOf(request.getPointsToUse());
        BigDecimal netPayAmount = totalAmountBd.subtract(pointsUsedBd);

        PaymentMethod method = request.getPaymentMethod();
        if (method == null) method = PaymentMethod.CARD;

        int pointsEarned = method.calculatePoints(netPayAmount);

        customer.setPoints(customer.getPoints() + pointsEarned);
        customerRepository.save(customer);
    }

    public void archiveOrder(Long id) {
    }

    public List<Order> findMyOrders(String phone) {
        return orderRepository.findByCustomerPhoneOrderByOrderDateDesc(phone);
    }

    public String getOrderStatus(Long id) {
        return orderRepository.findById(id)
                .map(order -> order.getStatus().name())
                .orElse("UNKNOWN");
    }

    public Order getOrderById(Long id) {
        return orderRepository.findById(id)
                .orElseThrow(() -> new RuntimeException("Order not found with id: " + id));
    }
}