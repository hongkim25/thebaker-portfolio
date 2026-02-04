package com.hong.thebaker.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.provisioning.InMemoryUserDetailsManager;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
@EnableWebSecurity
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .authorizeHttpRequests((requests) -> requests
                        // Static files and public pages
                        .requestMatchers("/login", "/css/**", "/js/**", "/images/**", "/v3/api-docs/**", "/swagger-ui/**", "/swagger-ui.html").permitAll()
                        .requestMatchers("/", "/privacy", "/about.html", "/index.html", "/menu.html", "/my-order.html", "/reservation.html").permitAll()
                        .requestMatchers("/manifest.json", "/sw.js", "/icon-*.png").permitAll()
                        .requestMatchers("/h2-console/**").permitAll()

                        // Public APIs (customer-facing)
                        .requestMatchers("/api/products/**").permitAll()
                        .requestMatchers("/api/shop/**").permitAll()
                        .requestMatchers("/api/orders", "/api/orders/search", "/api/orders/*/status").permitAll()

                        // Protected APIs (staff only)
                        .requestMatchers("/api/staff/**").hasRole("STAFF")
                        .requestMatchers("/api/customers/**").hasRole("STAFF")
                        .requestMatchers("/api/points/**").hasRole("STAFF")

                        // Staff dashboard
                        .requestMatchers("/staff/**").authenticated()
                        .anyRequest().authenticated()
                )
                .formLogin((form) -> form
                        .loginPage("/login")
                        .loginProcessingUrl("/login")
                        .defaultSuccessUrl("/staff", true)
                        .permitAll()
                )
                .logout((logout) -> logout.permitAll())
                .csrf((csrf) -> csrf.disable())
                .headers((headers) -> headers.frameOptions((frame) -> frame.sameOrigin()));

        return http.build();
    }

    @Value("${app.staff.username}")
    private String staffUsername;

    @Value("${app.staff.password}")
    private String staffPassword;

    @Bean
    public InMemoryUserDetailsManager userDetailsService() {
        UserDetails staff = User.withDefaultPasswordEncoder()
                .username(staffUsername)
                .password(staffPassword)
                .roles("STAFF")
                .build();
        return new InMemoryUserDetailsManager(staff);
    }
}