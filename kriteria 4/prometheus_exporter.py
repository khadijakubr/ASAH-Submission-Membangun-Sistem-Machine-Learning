from prometheus_client import start_http_server, Gauge, Counter
import time
import psutil
import random

# Custom metrics
data_drift_score = Gauge('data_drift_score', 'Data drift detection score')
model_version = Gauge('model_version', 'Current model version')
training_accuracy = Gauge('training_accuracy', 'Training accuracy')
validation_accuracy = Gauge('validation_accuracy', 'Validation accuracy')
total_data_processed = Counter('total_data_processed', 'Total data processed')

def update_metrics():
    while True:
        # Simulate metrics (ganti dengan real metrics di produksi)
        data_drift_score.set(random.uniform(0, 0.3))
        model_version.set(1.0)
        training_accuracy.set(0.92 + random.uniform(-0.02, 0.02))
        validation_accuracy.set(0.88 + random.uniform(-0.02, 0.02))
        total_data_processed.inc(random.randint(1, 10))
        
        time.sleep(10)

if __name__ == '__main__':
    # Start server on port 8000
    start_http_server(8000)
    print("Prometheus exporter running on port 8000")
    update_metrics()