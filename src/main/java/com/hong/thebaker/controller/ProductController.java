package com.hong.thebaker.controller;

import com.hong.thebaker.entity.Product;
import com.hong.thebaker.repository.ProductRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.math.BigDecimal;
import java.time.DayOfWeek;
import java.time.LocalDate;
import java.util.Base64;
import java.util.List;
import java.util.stream.Collectors;

@Slf4j
@RestController
@RequestMapping("/api/products")
@RequiredArgsConstructor
public class ProductController {

    private final ProductRepository productRepository;

    @GetMapping
    public List<Product> getMenuByDate(@RequestParam(required = false) String date) {
        List<Product> allProducts = productRepository.findAll();

        LocalDate targetDate;
        if (date == null || date.isEmpty()) {
            targetDate = LocalDate.now();
        } else {
            try {
                targetDate = LocalDate.parse(date);
            } catch (Exception e) {
                log.warn("Invalid date format: {}, defaulting to today", date);
                targetDate = LocalDate.now();
            }
        }

        DayOfWeek dayOfWeek = targetDate.getDayOfWeek();

        return allProducts.stream()
                .filter(product -> isAvailableOnDay(product, dayOfWeek))
                .collect(Collectors.toList());
    }

    private boolean isAvailableOnDay(Product product, DayOfWeek day) {
        String type = product.getCategory();

        if (type == null || type.equals("ALL")) return true;

        // HARD BREAD: Thu, Fri, Sat
        if (type.equals("HARD")) {
            return day == DayOfWeek.THURSDAY ||
                    day == DayOfWeek.FRIDAY ||
                    day == DayOfWeek.SATURDAY;
        }

        // SOFT BREAD: Sun, Mon, Wed
        if (type.equals("SOFT")) {
            return day == DayOfWeek.SUNDAY ||
                    day == DayOfWeek.MONDAY ||
                    day == DayOfWeek.WEDNESDAY;
        }

        return false;
    }

    @PostMapping
    public ResponseEntity<Product> createProduct(
            @RequestParam("name") String name,
            @RequestParam("price") BigDecimal price,
            @RequestParam("category") String category,
            @RequestParam("stockQuantity") int stockQuantity,
            @RequestParam(value = "image", required = false) MultipartFile imageFile
    ) {
        log.info("Creating product: {}", name);
        try {
            Product product = new Product();
            product.setName(name);
            product.setPrice(price);
            product.setCategory(category);
            product.setStockQuantity(stockQuantity);

            if (imageFile != null && !imageFile.isEmpty()) {
                String base64Image = Base64.getEncoder().encodeToString(imageFile.getBytes());
                product.setImageBase64("data:image/jpeg;base64," + base64Image);
            }

            return ResponseEntity.ok(productRepository.save(product));
        } catch (Exception e) {
            log.error("Failed to create product: {}", e.getMessage());
            return ResponseEntity.internalServerError().build();
        }
    }

    @PutMapping("/{id}")
    public ResponseEntity<Product> updateProduct(
            @PathVariable Long id,
            @RequestParam("name") String name,
            @RequestParam("price") BigDecimal price,
            @RequestParam("category") String category,
            @RequestParam("stockQuantity") int stockQuantity,
            @RequestParam(value = "image", required = false) MultipartFile imageFile
    ) {
        log.info("Updating product: {}", id);
        return productRepository.findById(id).map(product -> {
            product.setName(name);
            product.setPrice(price);
            product.setCategory(category);
            product.setStockQuantity(stockQuantity);

            if (imageFile != null && !imageFile.isEmpty()) {
                try {
                    String base64Image = Base64.getEncoder().encodeToString(imageFile.getBytes());
                    product.setImageBase64("data:image/jpeg;base64," + base64Image);
                } catch (Exception e) {
                    log.error("Failed to process image for product {}: {}", id, e.getMessage());
                    throw new RuntimeException("Image upload failed");
                }
            }

            return ResponseEntity.ok(productRepository.save(product));
        }).orElse(ResponseEntity.notFound().build());
    }

    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteProduct(@PathVariable Long id) {
        log.info("Deleting product: {}", id);
        productRepository.deleteById(id);
        return ResponseEntity.ok().build();
    }
}
