from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import random

app = Flask(__name__)

# Prometheus Metrics
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
prediction_errors = Counter('prediction_errors_total', 'Prediction errors')
active_requests = Gauge('active_requests', 'Active requests')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('memory_usage_mb', 'Memory usage')
successful_predictions = Counter('successful_predictions_total', 'Successful predictions')

@app.route('/predict', methods=['POST'])
def predict():
    active_requests.inc()
    start_time = time.time()
    
    try:
        data = request.json
        features = data.get('features', [])
        
        # Simulate prediction
        prediction = random.uniform(0, 100)
        
        # Update metrics
        prediction_counter.inc()
        successful_predictions.inc()
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        model_accuracy.set(0.85 + random.uniform(-0.05, 0.05))
        
        return jsonify({
            'prediction': prediction,
            'latency': latency
        })
        
    except Exception as e:
        prediction_errors.inc()
        return jsonify({'error': str(e)}), 500
    finally:
        active_requests.dec()

@app.route('/metrics')
def metrics():
    # FIX: Return dengan Content-Type yang benar
    try:
        # Update system metrics
        cpu_usage.set(random.uniform(20, 80))
        memory_usage.set(random.uniform(100, 500))
        
        # Generate metrics dengan Content-Type yang benar
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        print(f"Error generating metrics: {e}")
        return Response(
            "# Error generating metrics\n",
            mimetype=CONTENT_TYPE_LATEST
        )

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

@app.route('/')
def home():
    return jsonify({
        'service': 'ML Model Serving',
        'endpoints': {
            'predict': '/predict (POST)',
            'metrics': '/metrics (GET)',
            'health': '/health (GET)'
        }
    })

if __name__ == '__main__':
    print("Metrics:  http://localhost:5001/metrics")
    print("Health:  http://localhost:5001/health")
    print("Predict:  http://localhost:5001/predict (POST)")
    app.run(host='0.0.0.0', port=5001, debug=False)