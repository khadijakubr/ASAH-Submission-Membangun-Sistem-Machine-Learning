from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import time
import mlflow
import mlflow.sklearn

app = Flask(__name__)

# --- Prometheus Metrics ---
prediction_counter = Counter('predictions_total', 'Total predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
prediction_errors = Counter('prediction_errors_total', 'Prediction errors')
active_requests = Gauge('active_requests', 'Active requests')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage')
memory_usage = Gauge('memory_usage_mb', 'Memory usage')
successful_predictions = Counter('successful_predictions_total', 'Successful predictions')

# --- Load model dari MLflow ---
MODEL_URI = "runs:/bf87b979a5264d078503bebe0b2abaf2/model"  
model = mlflow.sklearn.load_model(MODEL_URI)

@app.route('/predict', methods=['POST'])
def predict():
    active_requests.inc()
    start_time = time.time()
    
    try:
        data = request.json
        features = data.get('features', [])
        if not features:
            raise ValueError("Fitur tidak boleh kosong")

        # Prediksi dengan model nyata
        prediction = model.predict([features])[0]

        # Update metrics
        prediction_counter.inc()
        successful_predictions.inc()
        latency = time.time() - start_time
        prediction_latency.observe(latency)
        # Model accuracy bisa diupdate manual atau dari test set
        model_accuracy.set(0.85)

        return jsonify({
            'prediction': float(prediction),
            'latency': latency
        })
        
    except Exception as e:
        prediction_errors.inc()
        return jsonify({'error': str(e)}), 500
    finally:
        active_requests.dec()

@app.route('/metrics')
def metrics():
    try:
        cpu_usage.set(50)       # bisa diganti monitoring nyata
        memory_usage.set(200)   # bisa diganti monitoring nyata
        return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
    except Exception as e:
        return Response("# Error generating metrics\n", mimetype=CONTENT_TYPE_LATEST)

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
