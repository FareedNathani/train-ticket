import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
MODELS_DIR = BASE_DIR / "models"
FRONTEND_DIR = BASE_DIR / "frontend"

for directory in [UPLOAD_DIR, PROCESSED_DIR, SYNTHETIC_DIR, MODELS_DIR, FRONTEND_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Train Ticket Architecture Services (FudanSELab Train Ticket ~41 microservices)
TRAIN_TICKET_SERVICES = [
    "ts-ui-dashboard",
    "ts-auth-service",
    "ts-user-service",
    "ts-verification-code-service",
    "ts-contacts-service",
    "ts-order-service",
    "ts-order-other-service",
    "ts-travel-service",
    "ts-travel2-service",
    "ts-travel-plan-service",
    "ts-ticketinfo-service",
    "ts-station-service",
    "ts-train-service",
    "ts-config-service",
    "ts-security-service",
    "ts-execute-service",
    "ts-payment-service",
    "ts-inside-payment-service",
    "ts-rebook-service",
    "ts-cancel-service",
    "ts-route-service",
    "ts-route-plan-service",
    "ts-seat-service",
    "ts-assurance-service",
    "ts-consign-service",
    "ts-consign-price-service",
    "ts-food-service",
    "ts-food-map-service",
    "ts-voucher-service",
    "ts-news-service",
    "ts-price-service",
    "ts-notification-service",
    "ts-admin-basic-info-service",
    "ts-admin-order-service",
    "ts-admin-route-service",
    "ts-admin-travel-service",
    "ts-admin-user-service",
    "ts-delivery-service",
    "ts-wait-order-service",
    "ts-basic-service",
    "ts-preserve-service",
    "ts-preserve-other-service",
    "mongo-db",
    "mysql-db"
]

# Service Dependency Graph (Key caller -> callee relationships in Train Ticket)
SERVICE_DEPENDENCIES = {
    "ts-ui-dashboard": ["ts-auth-service", "ts-order-service", "ts-travel-service", "ts-user-service"],
    "ts-preserve-service": ["ts-order-service", "ts-travel-service", "ts-seat-service", "ts-ticketinfo-service", "ts-user-service"],
    "ts-order-service": ["ts-order-other-service", "ts-station-service", "ts-inside-payment-service", "mongo-db"],
    "ts-order-other-service": ["ts-station-service", "mongo-db"],
    "ts-travel-service": ["ts-ticketinfo-service", "ts-train-service", "ts-route-service", "ts-order-service"],
    "ts-travel2-service": ["ts-ticketinfo-service", "ts-train-service", "ts-route-service"],
    "ts-travel-plan-service": ["ts-travel-service", "ts-travel2-service", "ts-station-service"],
    "ts-ticketinfo-service": ["ts-basic-service", "ts-price-service"],
    "ts-seat-service": ["ts-travel-service", "ts-order-service", "ts-config-service"],
    "ts-inside-payment-service": ["ts-payment-service", "ts-order-service"],
    "ts-rebook-service": ["ts-order-service", "ts-seat-service", "ts-travel-service"],
    "ts-cancel-service": ["ts-order-service", "ts-inside-payment-service"],
    "ts-security-service": ["ts-order-service"],
    "ts-execute-service": ["ts-order-service", "ts-order-other-service"],
    "ts-food-service": ["ts-food-map-service", "ts-travel-service", "ts-station-service"],
    "ts-consign-service": ["ts-consign-price-service"],
    "ts-user-service": ["ts-auth-service", "mongo-db"],
    "ts-auth-service": ["mongo-db"],
    "ts-station-service": ["mongo-db"],
    "ts-train-service": ["mongo-db"],
    "ts-route-service": ["mongo-db"],
    "ts-price-service": ["mongo-db"],
    "ts-basic-service": ["ts-station-service", "ts-train-service", "ts-route-service", "ts-price-service"],
}

# Telemetry categories
TELEMETRY_TYPES = ["logs", "metrics", "traces", "events"]

# Standardized Anomaly Categories
ANOMALY_CATEGORIES = [
    "NORMAL",
    "HIGH_LATENCY",
    "HIGH_CPU",
    "HIGH_MEMORY",
    "DATABASE_FAILURE",
    "SERVICE_FAILURE",
    "NETWORK_FAILURE",
    "HIGH_ERROR_RATE"
]

# Severity Levels
SEVERITY_LEVELS = ["NORMAL", "WARNING", "CRITICAL"]
