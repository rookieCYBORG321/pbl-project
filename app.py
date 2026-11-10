#!/usr/bin/env python3
"""
CardioPulse AI - Web Application Server & REST API
==================================================
Zero-dependency Python HTTP server serving the interactive web application
and providing live clinical risk scoring, SHAP explainability, and batch evaluation.
"""

import os
import sys
import json
import csv
import io
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import model_engine

PORT = 8000
HOST = "0.0.0.0"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class CardioPulseHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler supporting Static Assets and JSON REST APIs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_json_response({"status": "healthy", "service": "CardioPulse AI Pipeline", "version": "3.0.0"})
            return

        if path == "/api/benchmarks":
            data = {
                "models": [
                    {"model": "STACKING ENSEMBLE (Default 0.50)", "accuracy": 90.76, "precision": 91.26, "recall": 92.16, "f1": 91.71, "auc": 0.9315, "tier": "Champion", "type": "Ensemble"},
                    {"model": "STACKING ENSEMBLE (Tuned Cutoff = 0.371)", "accuracy": 89.67, "precision": 88.07, "recall": 94.12, "f1": 90.99, "auc": 0.9315, "tier": "High Sensitivity", "type": "Ensemble"},
                    {"model": "ExtraTrees Classifier", "accuracy": 89.67, "precision": 88.07, "recall": 94.12, "f1": 90.99, "auc": 0.9328, "tier": "Base Learner", "type": "Tree Ensemble"},
                    {"model": "Random Forest", "accuracy": 89.13, "precision": 89.42, "recall": 91.18, "f1": 90.29, "auc": 0.9230, "tier": "Base Learner", "type": "Tree Ensemble"},
                    {"model": "CatBoost Classifier", "accuracy": 88.59, "precision": 90.91, "recall": 88.24, "f1": 89.55, "auc": 0.9296, "tier": "Base Learner", "type": "Gradient Boosting"},
                    {"model": "Support Vector Machine (RBF)", "accuracy": 88.59, "precision": 89.32, "recall": 90.20, "f1": 89.76, "auc": 0.9385, "tier": "Base Learner", "type": "Kernel SVM"},
                    {"model": "LightGBM Classifier", "accuracy": 88.59, "precision": 90.91, "recall": 88.24, "f1": 89.55, "auc": 0.9243, "tier": "Base Learner", "type": "Gradient Boosting"},
                    {"model": "Multi-Layer Perceptron (MLP)", "accuracy": 88.04, "precision": 88.46, "recall": 90.20, "f1": 89.32, "auc": 0.9310, "tier": "Base Learner", "type": "Neural Network"},
                    {"model": "XGBoost Classifier", "accuracy": 87.50, "precision": 89.11, "recall": 88.24, "f1": 88.67, "auc": 0.9237, "tier": "Base Learner", "type": "Gradient Boosting"},
                    {"model": "Logistic Regression (Baseline)", "accuracy": 89.13, "precision": 88.50, "recall": 90.20, "f1": 89.34, "auc": 0.9280, "tier": "Baseline", "type": "Linear"},
                    {"model": "K-Nearest Neighbors (KNN)", "accuracy": 89.13, "precision": 88.46, "recall": 90.20, "f1": 89.32, "auc": 0.9265, "tier": "Baseline", "type": "Instance"},
                    {"model": "Decision Tree (Baseline)", "accuracy": 85.87, "precision": 85.71, "recall": 88.24, "f1": 86.96, "auc": 0.8920, "tier": "Baseline", "type": "Single Tree"}
                ],
                "confusion_matrix": {
                    "tp": 94, "fn": 8, "tn": 73, "fp": 9,
                    "sensitivity": 0.9216, "specificity": 0.8902, "precision": 0.9126, "accuracy": 0.9076,
                    "test_size": 184
                },
                "cohort_stats": {
                    "total_patients": 918,
                    "hospitals": ["Cleveland", "Hungarian", "Swiss", "Long Beach VA", "St. George"],
                    "disease_prevalence": 55.34,
                    "features_count": 47,
                    "imputed_cholesterol_records": 172
                }
            }
            self.send_json_response(data)
            return

        if path == "/api/sample-patients":
            csv_path = os.path.join(BASE_DIR, "sample_patients.csv")
            patients = []
            if os.path.exists(csv_path):
                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        patients.append(r)
            self.send_json_response({"patients": patients, "count": len(patients)})
            return

        # Serve static file or fallback to index.html
        if path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except Exception as e:
            self.send_error_response(400, f"Malformed JSON: {str(e)}")
            return

        if path == "/api/predict":
            try:
                res = model_engine.evaluate_patient(payload)
                self.send_json_response(res)
            except Exception as e:
                self.send_error_response(500, f"Inference error: {str(e)}")
            return

        if path == "/api/batch":
            try:
                patients = payload.get("patients", [])
                results = []
                high_risk_count = 0
                critical_risk_count = 0

                for p in patients:
                    eval_res = model_engine.evaluate_patient(p)
                    pred = eval_res["prediction"]
                    if pred["risk_level"] in ["High Risk", "Critical Risk"]:
                        high_risk_count += 1
                    if pred["risk_level"] == "Critical Risk":
                        critical_risk_count += 1
                    results.append({
                        "input": p,
                        "risk_probability": pred["probability_percent"],
                        "risk_level": pred["risk_level"],
                        "risk_color": pred["risk_color"],
                        "prediction_default": pred["prediction_default_threshold"],
                        "prediction_tuned": pred["prediction_tuned_threshold"],
                        "ischemia_score": pred["derived_biomarkers"]["ischemia_severity_score"],
                        "chronotropic_incompetence": pred["derived_biomarkers"]["chronotropic_incompetence"],
                        "phenotype_cluster": pred["derived_biomarkers"]["phenotype_cluster_name"]
                    })

                summary = {
                    "total_evaluated": len(patients),
                    "flagged_high_or_critical": high_risk_count,
                    "flagged_critical": critical_risk_count,
                    "positivity_rate": round((high_risk_count / max(1, len(patients))) * 100, 1),
                    "results": results
                }
                self.send_json_response(summary)
            except Exception as e:
                self.send_error_response(500, f"Batch processing error: {str(e)}")
            return

        self.send_error_response(404, "Endpoint not found")

    def send_json_response(self, data, status_code=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, code, message):
        self.send_json_response({"error": message, "code": code}, status_code=code)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=PORT):
    server_address = (HOST, port)
    httpd = HTTPServer(server_address, CardioPulseHandler)
    print("=" * 72)
    print("  CARDIOPULSE AI - CLINICAL CARDIOVASCULAR PREDICTION PLATFORM")
    print("=" * 72)
    print(f"  * Web Dashboard  : http://localhost:{port}")
    print(f"  * REST API Engine: http://localhost:{port}/api/predict")
    print(f"  * Model Benchmark: http://localhost:{port}/api/benchmarks")
    print(f"  * Project Root   : {BASE_DIR}")
    print("=" * 72)
    print("  Serving web dashboard with orange and beige clinical design system.")
    print("  Press Ctrl+C to terminate the server.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nCardioPulse AI server stopped gracefully.")
        httpd.server_close()

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else PORT
    run_server(port)

