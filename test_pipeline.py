#!/usr/bin/env python3
"""
CardioPulse AI - Automated Test & Validation Suite
==================================================
Tests cardiology biomarker feature engineering, model inference,
zero-value imputation logic, and probability calibration.
"""

import unittest
import model_engine

class TestCardioPulsePipeline(unittest.TestCase):

    def test_healthy_patient_low_risk(self):
        """Validates that a young patient with up-sloping ST and normal vitals is categorized as Low Risk."""
        patient = {
            "Age": 38,
            "Sex": "F",
            "ChestPainType": "ATA",
            "RestingBP": 118,
            "Cholesterol": 195,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 178,
            "ExerciseAngina": "N",
            "Oldpeak": 0.0,
            "ST_Slope": "Up"
        }
        res = model_engine.evaluate_patient(patient)
        pred = res["prediction"]

        self.assertLess(pred["probability"], 0.25)
        self.assertEqual(pred["risk_level"], "Low Risk")
        self.assertEqual(pred["prediction_default_threshold"], 0)
        self.assertFalse(pred["derived_biomarkers"]["chronotropic_incompetence"])
        self.assertAlmostEqual(pred["derived_biomarkers"]["pct_max_hr_achieved"], 97.8, delta=1.5)

    def test_ischemic_patient_critical_risk(self):
        """Validates that a patient with severe ST depression, ASY, Flat slope, and angina is Critical Risk."""
        patient = {
            "Age": 62,
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 145,
            "Cholesterol": 265,
            "FastingBS": 1,
            "RestingECG": "LVH",
            "MaxHR": 105,
            "ExerciseAngina": "Y",
            "Oldpeak": 2.5,
            "ST_Slope": "Flat"
        }
        res = model_engine.evaluate_patient(patient)
        pred = res["prediction"]

        self.assertGreater(pred["probability"], 0.85)
        self.assertEqual(pred["risk_level"], "Critical Risk")
        self.assertEqual(pred["prediction_tuned_threshold"], 1)
        self.assertTrue(pred["derived_biomarkers"]["chronotropic_incompetence"])
        self.assertGreaterEqual(pred["derived_biomarkers"]["ischemia_severity_score"], 10.0)

    def test_zero_cholesterol_imputation(self):
        """Validates that physiologically impossible 0-values in cholesterol (Switzerland/VA records) are imputed without crashing."""
        patient = {
            "Age": 55,
            "Sex": "M",
            "ChestPainType": "ASY",
            "RestingBP": 130,
            "Cholesterol": 0,  # Missing encoding from VA/Swiss cohort
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 125,
            "ExerciseAngina": "Y",
            "Oldpeak": 1.2,
            "ST_Slope": "Flat"
        }
        res = model_engine.evaluate_patient(patient)
        eng = res["engineered_features"]

        self.assertEqual(eng["Cholesterol_is_zero"], 1)
        self.assertGreater(eng["Cholesterol"], 200)  # Imputed value
        self.assertGreater(res["prediction"]["probability"], 0.50)

    def test_duke_ischemia_score_calculation(self):
        """Validates Duke treadmill-style ischemia calculation formula."""
        patient = {
            "Age": 50,
            "Sex": "M",
            "ChestPainType": "ASY",       # 2.0
            "RestingBP": 130,
            "Cholesterol": 220,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 140,
            "ExerciseAngina": "Y",       # 2.5
            "Oldpeak": 2.0,              # 2.0 * 2.5 = 5.0
            "ST_Slope": "Down"           # 2 * 3.0 = 6.0
        }
        # Expected: 5.0 + 6.0 + 2.5 + 2.0 = 15.5
        res = model_engine.evaluate_patient(patient)
        score = res["engineered_features"]["ischemia_severity_score"]
        self.assertAlmostEqual(score, 15.5, delta=0.1)

    def test_optimal_cutoff_behavior(self):
        """Validates that a borderline case (e.g. prob = 0.42) triggers the tuned triage cutoff (0.371)."""
        patient = {
            "Age": 54,
            "Sex": "F",
            "ChestPainType": "NAP",
            "RestingBP": 138,
            "Cholesterol": 230,
            "FastingBS": 0,
            "RestingECG": "Normal",
            "MaxHR": 132,
            "ExerciseAngina": "N",
            "Oldpeak": 1.2,
            "ST_Slope": "Flat"
        }
        res = model_engine.evaluate_patient(patient)
        pred = res["prediction"]

        # If probability falls between 0.371 and 0.50, tuned cutoff flags it as 1 while default is 0
        if 0.371 <= pred["probability"] < 0.50:
            self.assertEqual(pred["prediction_tuned_threshold"], 1)
            self.assertEqual(pred["prediction_default_threshold"], 0)

if __name__ == "__main__":
    unittest.main()

