export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');

  const benchmarks = [
    { model: "Stacking Ensemble (Tuned 0.371)", accuracy: 0.8967, precision: 0.8807, recall: 0.9412, f1: 0.9099, roc_auc: 0.9315, latency_ms: 1.2 },
    { model: "Stacking Ensemble (Default 0.50)", accuracy: 0.9076, precision: 0.9126, recall: 0.9216, f1: 0.9171, roc_auc: 0.9315, latency_ms: 1.2 },
    { model: "ExtraTrees Classifier", accuracy: 0.8967, precision: 0.8807, recall: 0.9412, f1: 0.9099, roc_auc: 0.9328, latency_ms: 2.4 },
    { model: "Support Vector Machine (RBF)", accuracy: 0.8859, precision: 0.8932, recall: 0.9020, f1: 0.8976, roc_auc: 0.9385, latency_ms: 0.8 },
    { model: "Random Forest", accuracy: 0.8913, precision: 0.8942, recall: 0.9118, f1: 0.9029, roc_auc: 0.9230, latency_ms: 3.1 },
    { model: "CatBoost Classifier", accuracy: 0.8859, precision: 0.9091, recall: 0.8824, f1: 0.8955, roc_auc: 0.9296, latency_ms: 1.9 },
    { model: "LightGBM Classifier", accuracy: 0.8859, precision: 0.9091, recall: 0.8824, f1: 0.8955, roc_auc: 0.9243, latency_ms: 0.9 },
    { model: "Multi-Layer Perceptron (MLP)", accuracy: 0.8804, precision: 0.8846, recall: 0.9020, f1: 0.8932, roc_auc: 0.9310, latency_ms: 1.1 },
    { model: "XGBoost Classifier", accuracy: 0.8750, precision: 0.8911, recall: 0.8824, f1: 0.8867, roc_auc: 0.9237, latency_ms: 1.4 }
  ];

  res.status(200).json({
    status: "success",
    dataset: "Fedesoriano 5-Hospital Cohort (N=918)",
    cross_validation: "5-Fold Stratified CV",
    benchmarks
  });
}

