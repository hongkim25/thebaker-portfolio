package com.hong.thebaker.controller;

import com.hong.thebaker.repository.CustomerRepository;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/customers")
public class CustomerController {

    private final CustomerRepository customerRepository;

    public CustomerController(CustomerRepository customerRepository) {
        this.customerRepository = customerRepository;
    }

    @GetMapping("/{phone}")
    public ResponseEntity<?> getCustomerPoints(@PathVariable String phone) {
        return customerRepository.findByPhone(phone)
                .map(customer -> ResponseEntity.ok((Object) customer))
                .orElse(ResponseEntity.ok(Map.of("points", 0)));
    }
}