package com.hong.thebaker.repository;

import com.hong.thebaker.entity.Order;
import com.hong.thebaker.entity.OrderStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface OrderRepository extends JpaRepository<Order, Long> {

    // 1. By Customer ID
    List<Order> findByCustomerId(Long customerId);

    // 🔴 NO ARCHIVED METHOD HERE.
    // It is gone. StaffController now uses findAll() instead.

    // 2. By Phone
    List<Order> findByCustomerPhoneOrderByOrderDateDesc(String phone);

    // 3. For Alarm
    long countByStatus(OrderStatus status);
}