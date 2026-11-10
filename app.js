/**
 * CardioPulse AI - Interactive Clinical Application Engine
 * ========================================================
 * Implements real-time clinical risk prediction, dynamic SHAP waterfall attribution,
 * interactive Chart.js visualizations, batch evaluation, and clinical report export.
 */

// Global State
const STATE = {
  currentPatient: null,
  batchPatients: [],
  charts: {},
  benchmarks: null
};

// Preset Patient Clinical Archetypes
const PRESETS = {
  preset1: {
    name: "Patient A: Severe Exercise Ischemia",
    Age: 58, Sex: "M", ChestPainType: "ASY", RestingBP: 140, Cholesterol: 260,
    FastingBS: 1, RestingECG: "Normal", MaxHR: 118, ExerciseAngina: "Y", Oldpeak: 2.2, ST_Slope: "Flat"
  },
  preset2: {
    name: "Patient B: Healthy Normotensive Baseline",
    Age: 41, Sex: "F", ChestPainType: "ATA", RestingBP: 118, Cholesterol: 195,
    FastingBS: 0, RestingECG: "Normal", MaxHR: 175, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up"
  },
  preset3: {
    name: "Patient C: Silent Ischemia in Diabetic Male",
    Age: 64, Sex: "M", ChestPainType: "ASY", RestingBP: 148, Cholesterol: 245,
    FastingBS: 1, RestingECG: "LVH", MaxHR: 105, ExerciseAngina: "Y", Oldpeak: 2.8, ST_Slope: "Down"
  },
  preset4: {
    name: "Patient D: Exercise-Induced Angina",
    Age: 49, Sex: "M", ChestPainType: "NAP", RestingBP: 135, Cholesterol: 230,
    FastingBS: 0, RestingECG: "ST", MaxHR: 138, ExerciseAngina: "Y", Oldpeak: 1.2, ST_Slope: "Flat"
  },
  preset5: {
    name: "Patient E: Swiss Cohort (Zero-Chol Imputed)",
    Age: 59, Sex: "M", ChestPainType: "ASY", RestingBP: 135, Cholesterol: 0,
    FastingBS: 0, RestingECG: "Normal", MaxHR: 115, ExerciseAngina: "Y", Oldpeak: 1.6, ST_Slope: "Flat"
  }
};

// Phenotype Cluster Centroids
const PHENOTYPES = [
  { name: "Low-Risk Normotensive", age: 46.5, bp: 124.0, chol: 218.0, max_hr: 162.0, oldpeak: 0.1, ischemia: 0.8, st_slope: 0, cp_risk: 0.8 },
  { name: "Severe Exercise Ischemia", age: 58.2, bp: 142.0, chol: 252.0, max_hr: 121.0, oldpeak: 1.9, ischemia: 9.2, st_slope: 1.2, cp_risk: 2.8 },
  { name: "Metabolic Silent Ischemia", age: 63.8, bp: 156.0, chol: 272.0, max_hr: 110.0, oldpeak: 2.4, ischemia: 10.5, st_slope: 1.4, cp_risk: 2.9 },
  { name: "Atypical Non-Ischemic Strain", age: 52.4, bp: 134.0, chol: 235.0, max_hr: 142.0, oldpeak: 0.6, ischemia: 3.4, st_slope: 0.4, cp_risk: 1.4 }
];

// Document Ready Initialization
document.addEventListener("DOMContentLoaded", () => {
  initializeNavbar();
  initializeFormInputs();
  initializeTabs();
  initializePresetButtons();
  initializeCharts();
  loadSampleBatchCohort();
  loadBenchmarksData();

  // Load default preset 1
  loadPreset("preset1");
});

// Navbar Scroll Effect & Smooth Scrolling
function initializeNavbar() {
  const navbar = document.querySelector(".navbar");
  window.addEventListener("scroll", () => {
    if (window.scrollY > 20) {
      navbar.classList.add("scrolled");
    } else {
      navbar.classList.remove("scrolled");
    }
  });

  // Mobile menu toggle
  const toggle = document.querySelector(".mobile-toggle");
  const navMenu = document.querySelector(".nav-menu");
  if (toggle && navMenu) {
    toggle.addEventListener("click", () => {
      navMenu.classList.toggle("mobile-open");
    });
  }
}

// Form Inputs Live Binding
function initializeFormInputs() {
  const inputs = [
    { id: "inputAge", valId: "valAge", event: "input" },
    { id: "inputRestingBP", valId: "valRestingBP", event: "input" },
    { id: "inputCholesterol", valId: "valCholesterol", event: "input", customLabel: (v) => v == 0 ? "0 mg/dL (Auto-Imputed)" : v + " mg/dL" },
    { id: "inputMaxHR", valId: "valMaxHR", event: "input" },
    { id: "inputOldpeak", valId: "valOldpeak", event: "input" }
  ];

  inputs.forEach(({ id, valId, event, customLabel }) => {
    const el = document.getElementById(id);
    const label = document.getElementById(valId);
    if (el && label) {
      el.addEventListener(event, () => {
        label.textContent = customLabel ? customLabel(el.value) : el.value;
        onParameterModified();
      });
    }
  });

  // Selects & Radio Inputs
  const others = ["selectSex", "selectChestPain", "selectRestingECG", "selectFastingBS", "selectExerciseAngina", "selectSTSlope"];
  others.forEach(name => {
    const elements = document.querySelectorAll(`[name="${name}"], #${name}`);
    elements.forEach(el => {
      el.addEventListener("change", onParameterModified);
    });
  });

  // Action Buttons - Analyze
  const btnAnalyze = document.getElementById("btnAnalyze");
  if (btnAnalyze) {
    btnAnalyze.addEventListener("click", triggerAnalysisAnimation);
  }

  const btnReset = document.getElementById("btnResetForm");
  if (btnReset) btnReset.addEventListener("click", () => loadPreset("preset2"));

  const btnRandom = document.getElementById("btnRandomPatient");
  if (btnRandom) btnRandom.addEventListener("click", generateRandomPatient);

  const btnPrint = document.getElementById("btnPrintReport");
  if (btnPrint) btnPrint.addEventListener("click", openPrintReportModal);
}

let probAnimationTimer = null;

function animateProbNumber(startVal, endVal, durationMs = 700) {
  const el = document.getElementById("resProbNumber");
  if (!el) return;

  if (probAnimationTimer) {
    cancelAnimationFrame(probAnimationTimer);
  }

  const startTime = performance.now();
  const diff = endVal - startVal;

  function step(now) {
    const elapsed = now - startTime;
    const progress = Math.min(1, elapsed / durationMs);
    // Smooth ease-out cubic
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = startVal + (diff * ease);
    el.innerHTML = `${current.toFixed(1)}<span>%</span>`;

    if (progress < 1) {
      probAnimationTimer = requestAnimationFrame(step);
    } else {
      el.innerHTML = `${endVal.toFixed(1)}<span>%</span>`;
    }
  }

  probAnimationTimer = requestAnimationFrame(step);
}

function onParameterModified() {
  const btnAnalyze = document.getElementById("btnAnalyze");
  if (btnAnalyze) {
    btnAnalyze.classList.add("btn-ready");
    btnAnalyze.innerHTML = '<i data-lucide="play-circle"></i> Analyze';
    if (window.lucide) lucide.createIcons();
  }
}

function triggerAnalysisAnimation() {
  const btn = document.getElementById("btnAnalyze");
  if (btn) {
    btn.classList.remove("btn-ready");
    btn.classList.add("btn-analyzing");
    btn.innerHTML = '<i data-lucide="loader-2" class="spin"></i> Analyzing...';
    if (window.lucide) lucide.createIcons();
  }

  // Trigger pulse highlight on clinical assessment card
  const card = document.querySelector(".assessment-card");
  if (card) {
    card.classList.remove("card-pulse");
    void card.offsetWidth; // Force reflow
    card.classList.add("card-pulse");
  }

  setTimeout(() => {
    runClientInference();
    if (btn) {
      btn.classList.remove("btn-analyzing");
      btn.innerHTML = '<i data-lucide="activity"></i> Analyze';
      if (window.lucide) lucide.createIcons();
    }
  }, 180);
}

// Preset Buttons
function initializePresetButtons() {
  const buttons = document.querySelectorAll(".preset-chip");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      const key = btn.getAttribute("data-preset");
      loadPreset(key);
    });
  });
}

function loadPreset(key) {
  const data = PRESETS[key];
  if (!data) return;

  setFormValue("inputAge", "valAge", data.Age);
  setRadioValue("selectSex", data.Sex);
  setRadioValue("selectChestPain", data.ChestPainType);
  setFormValue("inputRestingBP", "valRestingBP", data.RestingBP);
  setFormValue("inputCholesterol", "valCholesterol", data.Cholesterol, (v) => v == 0 ? "0 mg/dL (Auto-Imputed)" : v + " mg/dL");
  setRadioValue("selectFastingBS", data.FastingBS);
  setRadioValue("selectRestingECG", data.RestingECG);
  setFormValue("inputMaxHR", "valMaxHR", data.MaxHR);
  setRadioValue("selectExerciseAngina", data.ExerciseAngina);
  setFormValue("inputOldpeak", "valOldpeak", data.Oldpeak);
  setRadioValue("selectSTSlope", data.ST_Slope);

  const btnAnalyze = document.getElementById("btnAnalyze");
  if (btnAnalyze) {
    btnAnalyze.classList.remove("btn-ready");
    btnAnalyze.innerHTML = '<i data-lucide="activity"></i> Analyze';
    if (window.lucide) lucide.createIcons();
  }

  runClientInference();
}

function setFormValue(id, labelId, val, formatFn) {
  const el = document.getElementById(id);
  const label = document.getElementById(labelId);
  if (el) el.value = val;
  if (label) label.textContent = formatFn ? formatFn(val) : val;
}

function setRadioValue(name, val) {
  const radios = document.querySelectorAll(`[name="${name}"]`);
  radios.forEach(r => {
    if (r.value == val) r.checked = true;
  });
}

function getFormValues() {
  return {
    Age: parseFloat(document.getElementById("inputAge").value) || 54,
    Sex: getRadioValue("selectSex", "M"),
    ChestPainType: getRadioValue("selectChestPain", "ASY"),
    RestingBP: parseFloat(document.getElementById("inputRestingBP").value) || 130,
    Cholesterol: parseFloat(document.getElementById("inputCholesterol").value) || 220,
    FastingBS: parseInt(getRadioValue("selectFastingBS", "0")),
    RestingECG: getRadioValue("selectRestingECG", "Normal"),
    MaxHR: parseFloat(document.getElementById("inputMaxHR").value) || 138,
    ExerciseAngina: getRadioValue("selectExerciseAngina", "N"),
    Oldpeak: parseFloat(document.getElementById("inputOldpeak").value) || 0.0,
    ST_Slope: getRadioValue("selectSTSlope", "Flat")
  };
}

function getRadioValue(name, fallback) {
  const checked = document.querySelector(`[name="${name}"]:checked`);
  return checked ? checked.value : fallback;
}

// Client-Side Real-Time Feature Engineering & Inferences
function runClientInference() {
  const previousProb = (STATE.currentPatient && STATE.currentPatient.probPercent !== undefined)
    ? parseFloat(STATE.currentPatient.probPercent)
    : null;

  const raw = getFormValues();

  // 1. Feature Engineering
  const age = raw.Age;
  const sex = raw.Sex;
  const chestPain = raw.ChestPainType;
  const rawBP = raw.RestingBP;
  const rawChol = raw.Cholesterol;
  const fastingBS = raw.FastingBS;
  const restingECG = raw.RestingECG;
  const maxHR = raw.MaxHR;
  const exerciseAngina = raw.ExerciseAngina;
  const oldpeak = raw.Oldpeak;
  const stSlope = raw.ST_Slope;

  // Imputation
  const cholIsZero = rawChol <= 0 ? 1 : 0;
  const cholesterol = cholIsZero ? (sex === "M" ? 242.0 : 235.0) : rawChol;
  const restingBP = rawBP <= 0 ? 130.0 : rawBP;

  const oldpeakPos = Math.max(0.0, oldpeak);
  const sexNum = sex === "M" ? 1 : 0;
  const exerciseAnginaNum = exerciseAngina === "Y" ? 1 : 0;

  const stSlopeMap = { "Up": 0, "Flat": 1, "Down": 2 };
  const stSlopeOrd = stSlopeMap[stSlope] !== undefined ? stSlopeMap[stSlope] : 1;

  const chestPainASY = chestPain === "ASY" ? 1 : 0;
  const cpRiskMap = { "ATA": 0, "NAP": 1, "TA": 2, "ASY": 3 };
  const chestPainRiskOrd = cpRiskMap[chestPain] !== undefined ? cpRiskMap[chestPain] : 3;

  // Cardiology Biomarkers
  const theoreticalMaxHR = 220.0 - age;
  const hrReserve = theoreticalMaxHR - maxHR;
  const pctMaxHRAchieved = maxHR / (theoreticalMaxHR + 1e-5);
  const chronotropicIncomp = pctMaxHRAchieved < 0.80 ? 1 : 0;

  const ischemiaScore = (oldpeakPos * 2.0) + (stSlopeOrd * 1.5) + (exerciseAnginaNum * 1.5) + (chestPainASY * 1.2);
  const asyWithFlatOrDown = (chestPain === "ASY" && (stSlope === "Flat" || stSlope === "Down")) ? 1 : 0;
  const ratePressureProduct = (restingBP * maxHR) / 1000.0;

  const metabolicCardioRisk = (restingBP >= 135 ? 1 : 0) + (cholesterol >= 240 ? 1 : 0) + fastingBS + sexNum;

  // Phenotype Centroid Proximity
  const currVector = [age, restingBP, cholesterol, maxHR, oldpeakPos, ischemiaScore, stSlopeOrd, chestPainRiskOrd];
  const weights = [0.03, 0.02, 0.015, 0.03, 0.4, 0.25, 0.5, 0.3];

  let nearestCluster = 0;
  let minDistance = 999999;
  PHENOTYPES.forEach((c, idx) => {
    const target = [c.age, c.bp, c.chol, c.max_hr, c.oldpeak, c.ischemia, c.st_slope, c.cp_risk];
    let dist = 0;
    for (let i = 0; i < currVector.length; i++) {
      dist += Math.pow(weights[i] * (currVector[i] - target[i]), 2);
    }
    dist = Math.sqrt(dist);
    if (dist < minDistance) {
      minDistance = dist;
      nearestCluster = idx;
    }
  });

  // Balanced Meta-Classifier Calibration
  const slopeTerm = stSlopeOrd === 1 ? 0.55 : (stSlopeOrd === 2 ? 0.90 : -0.75);
  const cpTerm = chestPainASY ? 0.60 : (chestPain === "ATA" ? -0.75 : (chestPain === "NAP" ? -0.30 : 0.15));
  const anginaTerm = exerciseAnginaNum ? 0.50 : -0.50;
  const oldpeakTerm = (oldpeakPos - 0.90) * 0.45;
  const hrTerm = -0.016 * (maxHR - 138.0) + (chronotropicIncomp ? 0.35 : -0.15);
  const bpTerm = (restingBP - 130.0) * 0.018;
  const ageTerm = (age - 53.0) * 0.038;
  const sexTerm = sexNum ? 0.45 : -0.45;
  const cholTerm = (cholesterol - 220.0) * 0.005;
  const fbsTerm = fastingBS ? 0.35 : -0.20;
  const dukeTerm = (ischemiaScore - 3.5) * 0.12;

  const intercept = 0.12;
  const rawLogOdds = intercept + slopeTerm + cpTerm + anginaTerm + oldpeakTerm +
                     hrTerm + bpTerm + ageTerm + sexTerm + cholTerm + fbsTerm + dukeTerm;

  const scaledLogOdds = rawLogOdds * 0.55;
  let probability = 1.0 / (1.0 + Math.exp(-scaledLogOdds));
  probability = Math.max(0.02, Math.min(0.98, probability));

  // Risk Classification
  let riskLevel = "Low Risk";
  let riskClass = "low";
  let actionText = "Patient exhibits healthy cardiac hemodynamics. Routine lifestyle and annual checkup recommended.";

  if (probability < 0.25) {
    riskLevel = "Low Risk";
    riskClass = "low";
  } else if (probability < 0.45) {
    riskLevel = "Moderate Risk";
    riskClass = "mod";
    actionText = "Borderline cardiac risk. Recommend ambulatory Holter monitoring, lipid profiling, and diet modification.";
  } else if (probability < 0.75) {
    riskLevel = "High Risk";
    riskClass = "high";
    actionText = "High probability of coronary ischemia. Schedule stress echocardiography and cardiology specialist consult.";
  } else {
    riskLevel = "Critical Risk";
    riskClass = "critical";
    actionText = "Urgent cardiology alert: Evidence of severe myocardial ischemia / silent transmural damage. Immediate angiography indicated.";
  }

  // SHAP Feature Attributions
  const rawShap = [
    { name: `ST Slope (${stSlope})`, val: slopeTerm },
    { name: `Duke Ischemia Score (${ischemiaScore.toFixed(1)})`, val: dukeTerm },
    { name: `Chest Pain (${chestPain})`, val: cpTerm },
    { name: `ST Depression Oldpeak (${oldpeak.toFixed(1)} mm)`, val: oldpeakTerm },
    { name: `Exercise Angina (${exerciseAngina})`, val: anginaTerm },
    { name: `Max Heart Rate (${Math.round(maxHR)} bpm)`, val: hrTerm },
    { name: `Age (${Math.round(age)} yrs)`, val: ageTerm },
    { name: `Biological Sex (${sex})`, val: sexTerm },
    { name: `Resting Blood Pressure (${Math.round(restingBP)} mmHg)`, val: bpTerm },
    { name: `Fasting Blood Sugar`, val: fbsTerm }
  ];

  const shapList = rawShap.map(item => {
    const impactPct = +(item.val * 14.0).toFixed(1);
    return {
      name: item.name,
      impact: impactPct,
      pos: impactPct > 0
    };
  }).sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact));

  // Store in State
  STATE.currentPatient = {
    raw,
    probability,
    probPercent: (probability * 100).toFixed(1),
    riskLevel,
    riskClass,
    actionText,
    biomarkers: {
      theoreticalMaxHR: Math.round(theoreticalMaxHR),
      hrReserve: Math.round(hrReserve),
      pctMaxHRAchieved: (pctMaxHRAchieved * 100).toFixed(1),
      chronotropicIncomp: Boolean(chronotropicIncomp),
      ischemiaScore: ischemiaScore.toFixed(1),
      ratePressureProduct: ratePressureProduct.toFixed(1),
      metabolicCardioRisk,
      phenotypeName: PHENOTYPES[nearestCluster].name
    },
    shap: shapList,
    decisions: {
      defaultThreshold: probability >= 0.50,
      tunedThreshold: probability >= 0.371
    }
  };

  updateAssessmentUI(STATE.currentPatient, previousProb);
}

// Update UI with calculated assessment
function updateAssessmentUI(data, previousProb) {
  // 1. Percentage & Risk Badge Animation
  const startNum = (previousProb !== null && !isNaN(previousProb)) ? previousProb : parseFloat(data.probPercent);
  const endNum = parseFloat(data.probPercent);
  animateProbNumber(startNum, endNum, 700);

  const badge = document.getElementById("resRiskBadge");
  if (badge) {
    badge.className = `gauge-risk-pill badge-${data.riskClass === 'low' ? 'green' : (data.riskClass === 'mod' ? 'amber' : (data.riskClass === 'high' ? 'orange' : 'red'))}`;
    badge.textContent = data.riskLevel.toUpperCase();
  }

  // 2. Animate Gauge Arc & Needle
  const arc = document.getElementById("gaugeFillArc");
  const needle = document.getElementById("gaugeNeedle");
  if (arc) {
    // Semicircle perimeter for R=85: pi * 85 = 267.04
    const maxOffset = 267.04;
    const offset = Math.max(0, maxOffset - (data.probability * maxOffset));
    arc.style.strokeDashoffset = offset.toFixed(2);
  }
  if (needle) {
    // Angle spans from -90deg (0% risk) to +90deg (100% risk)
    const angle = -90 + (data.probability * 180);
    needle.style.transform = `rotate(${angle.toFixed(2)}deg)`;
    needle.setAttribute("transform", `rotate(${angle.toFixed(2)} 120 115)`);
  }

  // 3. Dual Threshold Decisions
  const decDefault = document.getElementById("resDecisionDefault");
  if (decDefault) {
    decDefault.textContent = data.decisions.defaultThreshold ? "DISEASE (1)" : "NORMAL (0)";
    decDefault.style.color = data.decisions.defaultThreshold ? "var(--orange-600)" : "var(--status-low)";
  }

  const decTuned = document.getElementById("resDecisionTuned");
  if (decTuned) {
    decTuned.textContent = data.decisions.tunedThreshold ? "DISEASE (1)" : "NORMAL (0)";
    decTuned.style.color = data.decisions.tunedThreshold ? "var(--status-critical)" : "var(--status-low)";
  }

  // 4. Derived Biomarkers
  setText("bmMaxHRPct", `${data.biomarkers.pctMaxHRAchieved}%`);
  setText("bmChronotropic", data.biomarkers.chronotropicIncomp ? "INCOMPETENT (<80%)" : "NORMAL (>=80%)");
  const elChron = document.getElementById("bmChronotropic");
  if (elChron) elChron.style.color = data.biomarkers.chronotropicIncomp ? "var(--status-critical)" : "var(--status-low)";

  setText("bmIschemiaScore", data.biomarkers.ischemiaScore);
  setText("bmRPP", `${data.biomarkers.ratePressureProduct} k`);
  setText("bmMetabolicRisk", `${data.biomarkers.metabolicCardioRisk} / 4`);
  setText("bmPhenotype", data.biomarkers.phenotypeName);

  // 5. SHAP Breakdown Bars
  renderPatientSHAPWaterfall(data.shap);

  // 6. Actionable Triage Guidance
  setText("resClinicalAction", data.actionText);

  // 7. Plain-Language Explainable AI (XAI) Patient Summary
  renderPlainLanguageExplanation(data);
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// Plain-Language Explainable AI (XAI) Patient Summary Generator
function renderPlainLanguageExplanation(data) {
  const container = document.getElementById("xaiSimpleExplanation");
  if (!container) return;

  const raw = data.raw;
  const prob = data.probability;
  const probPct = data.probPercent;
  const riskClass = data.riskClass; // low, mod, high, critical

  // 1. Plain-English Verdict
  let verdictIcon = "✅";
  let verdictColor = "var(--status-low)";
  let verdictHeadline = `Low Risk (${probPct}%) - Normal Cardiac Function`;
  let verdictSummary = `Based on the test results, this patient's heart is working normally. Blood is flowing freely through the coronary arteries, and the heart muscle is responding well under exercise stress.`;

  if (riskClass === "critical") {
    verdictIcon = "🚨";
    verdictColor = "var(--status-critical)";
    verdictHeadline = `Critical Risk (${probPct}%) - High Suspicion of Heart Artery Blockage`;
    verdictSummary = `There are strong and urgent warning signs that one or more arteries supplying blood to the heart are narrowed or blocked. When the patient exercises, the heart muscle is starved of oxygen.`;
  } else if (riskClass === "high") {
    verdictIcon = "⚠️";
    verdictColor = "var(--blue-600)";
    verdictHeadline = `High Risk (${probPct}%) - Significant Warning Signs of Heart Strain`;
    verdictSummary = `The test reveals noticeable indicators of cardiovascular disease. The heart muscle appears to struggle to get enough blood and oxygen during physical activity.`;
  } else if (riskClass === "mod") {
    verdictIcon = "⚠️";
    verdictColor = "var(--status-mod)";
    verdictHeadline = `Moderate / Borderline Risk (${probPct}%) - Early Warning Signs`;
    verdictSummary = `The patient sits in a borderline risk range. While not an immediate emergency, there are early physiological clues suggesting increased strain or early plaque buildup in the blood vessels.`;
  }

  // 2. Specific Plain-Language Observations (Why did the AI say this?)
  const observations = [];

  // Observation on Chest Pain / Silent Ischemia
  if (raw.ChestPainType === "ASY") {
    if (riskClass === "critical" || riskClass === "high") {
      observations.push({
        icon: "⚠️",
        title: "Silent Warning (No Obvious Chest Pain)",
        desc: "Even though the patient did not report typical chest pain, silent artery blockages are dangerous because damage occurs without warning. This is common in patients with elevated blood pressure or diabetes."
      });
    } else {
      observations.push({
        icon: "ℹ️",
        title: "No Chest Pain Reported",
        desc: "The patient had no angina discomfort during the exam."
      });
    }
  } else if (raw.ChestPainType === "TA") {
    observations.push({
      icon: "⚠️",
      title: "Typical Angina Pain",
      desc: "The patient reported classic chest tightness or pressure, a textbook symptom when heart muscle demands more oxygen than the arteries can deliver."
    });
  } else if (raw.ChestPainType === "ATA") {
    observations.push({
      icon: "🛡️",
      title: "Atypical Chest Sensations",
      desc: "The discomfort reported is atypical and is generally protective against severe blockages in coronary arteries."
    });
  } else if (raw.ChestPainType === "NAP") {
    observations.push({
      icon: "ℹ️",
      title: "Non-Anginal Discomfort",
      desc: "Chest symptoms are likely musculoskeletal or digestive rather than related to coronary artery blockage."
    });
  }

  // Observation on Exercise ECG & ST Depression
  if (raw.Oldpeak >= 1.5 || raw.ST_Slope === "Flat" || raw.ST_Slope === "Down") {
    observations.push({
      icon: "📉",
      title: "ECG Electrical Tracing Dipped During Stress",
      desc: `The treadmill ECG showed a dip of ${raw.Oldpeak} mm (ST depression) with a ${raw.ST_Slope.toLowerCase()}-sloping shape. This is an objective electrical sign that the heart muscle was starved of oxygen while exercising.`
    });
  } else {
    observations.push({
      icon: "📈",
      title: "Healthy Exercise Electrical Tracing",
      desc: `The ECG electrical waveform stayed healthy with an upward slope (${raw.Oldpeak} mm drop), showing the heart muscle maintained good oxygenation under physical stress.`
    });
  }

  // Observation on Heart Rate Response
  if (data.biomarkers.chronotropicIncomp) {
    observations.push({
      icon: "💓",
      title: "Heart Could Not Speed Up Sufficiently (Chronotropic Incompetence)",
      desc: `During maximum exertion, the heart only reached ${data.biomarkers.pctMaxHRAchieved}% of the expected peak rate for their age (target ${data.biomarkers.theoreticalMaxHR} bpm). The heart struggled to pump fast enough to meet exercise demand.`
    });
  } else {
    observations.push({
      icon: "💓",
      title: "Robust Heart Rate Response",
      desc: `The heart accelerated smoothly to ${data.biomarkers.pctMaxHRAchieved}% of target capacity (${raw.MaxHR} bpm), indicating healthy nervous system control of the cardiac pacemaker.`
    });
  }

  // Observation on Exercise Angina
  if (raw.ExerciseAngina === "Y") {
    observations.push({
      icon: "🚨",
      title: "Exercise Directly Triggered Chest Pain",
      desc: "Physical exertion directly brought on angina, demonstrating that arterial blood flow fell short under exercise load."
    });
  }

  // Observation on Vitals & Metabolic Load
  const metabolicClues = [];
  if (raw.RestingBP >= 140) metabolicClues.push(`high blood pressure (${raw.RestingBP} mmHg)`);
  if (raw.Cholesterol >= 240) metabolicClues.push(`elevated cholesterol (${raw.Cholesterol} mg/dL)`);
  if (raw.FastingBS == 1) metabolicClues.push(`elevated blood sugar (diabetes marker > 120 mg/dL)`);
  if (metabolicClues.length > 0) {
    observations.push({
      icon: "🩺",
      title: "Underlying Vessel Wear & Tear",
      desc: `Presence of ${metabolicClues.join(" and ")} accelerates plaque accumulation inside the coronary arteries.`
    });
  }

  // 3. Recommended Next Steps in Simple Terms
  let nextStepsHtml = "";
  if (riskClass === "critical" || riskClass === "high") {
    nextStepsHtml = `
      <div class="xai-action-box" style="border-left-color: var(--status-critical); background: #FEF2F2;">
        <strong>Recommended Next Steps in Simple Words:</strong>
        <ol style="margin: 0.35rem 0 0 1.15rem; padding: 0; line-height: 1.6;">
          <li><strong>See a Cardiologist Promptly:</strong> An ultrasound scan of the heart (echocardiogram) and a CT coronary angiogram should be scheduled to see the arteries clearly.</li>
          <li><strong>Take Prescribed Medications:</strong> Blood pressure and cholesterol medications help stabilize arterial walls and reduce clot risk.</li>
          <li><strong>Avoid Unsupervised Heavy Workouts:</strong> Refrain from intense physical exertion until an imaging specialist confirms safe exercise thresholds.</li>
        </ol>
      </div>
    `;
  } else if (riskClass === "mod") {
    nextStepsHtml = `
      <div class="xai-action-box" style="border-left-color: var(--status-mod); background: #FFFBEB;">
        <strong>Recommended Next Steps in Simple Terms:</strong>
        <ol style="margin: 0.35rem 0 0 1.15rem; padding: 0; line-height: 1.6;">
          <li><strong>24-Hour Heart Monitor (Holter):</strong> Track heart rhythm over a full day of normal activities.</li>
          <li><strong>Heart-Healthy Nutrition:</strong> Adopt a Mediterranean-style diet (olive oil, leafy greens, lean protein) to reduce arterial inflammation.</li>
          <li><strong>Follow-Up in 6 Months:</strong> Re-check blood pressure and cholesterol targets.</li>
        </ol>
      </div>
    `;
  } else {
    nextStepsHtml = `
      <div class="xai-action-box" style="border-left-color: var(--status-low); background: #ECFDF5;">
        <strong>Recommended Next Steps in Simple Terms:</strong>
        <ol style="margin: 0.35rem 0 0 1.15rem; padding: 0; line-height: 1.6;">
          <li><strong>Keep Up Active Living:</strong> 150 minutes of moderate aerobic exercise (brisk walking, swimming, cycling) per week.</li>
          <li><strong>Annual Routine Check:</strong> Have blood pressure and cholesterol screened once a year.</li>
          <li><strong>Healthy Sleep & Stress Habits:</strong> Adequate rest supports long-term vascular flexibility.</li>
        </ol>
      </div>
    `;
  }

  // Render into HTML
  container.innerHTML = `
    <div class="xai-verdict" style="color: ${verdictColor};">
      <span>${verdictIcon}</span>
      <span>${verdictHeadline}</span>
    </div>
    <p style="margin-bottom: 0.75rem; color: var(--text-secondary); line-height: 1.55;">
      ${verdictSummary}
    </p>

    <div style="font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.04em; color: var(--blue-700); margin-top: 0.75rem; margin-bottom: 0.4rem;">
      🔍 Why the AI Made This Decision (Simple Clues):
    </div>
    <ul class="xai-points-list">
      ${observations.map(item => `
        <li class="xai-point-item">
          <span class="xai-point-icon">${item.icon}</span>
          <div>
            <strong style="color: var(--text-primary);">${item.title}:</strong>
            <span>${item.desc}</span>
          </div>
        </li>
      `).join("")}
    </ul>

    ${nextStepsHtml}
  `;
}

// Render Patient SHAP Waterfall Bars
function renderPatientSHAPWaterfall(shapList) {
  const container = document.getElementById("shapBarsList");
  if (!container) return;

  container.innerHTML = "";
  const topFactors = shapList.slice(0, 5); // Display top 5 influential features

  topFactors.forEach(item => {
    const row = document.createElement("div");
    row.className = "shap-bar-item";

    const widthPct = Math.min(100, Math.abs(item.impact) * 3.5);
    const sign = item.impact > 0 ? "+" : "";

    row.innerHTML = `
      <span class="shap-feat-name" title="${item.name}">${item.name}</span>
      <div class="shap-bar-track">
        <div class="shap-bar-fill ${item.pos ? 'pos' : 'neg'}" style="width: ${widthPct}%;"></div>
      </div>
      <span class="shap-bar-val ${item.pos ? 'pos' : 'neg'}">${sign}${item.impact}%</span>
    `;
    container.appendChild(row);
  });
}

function generateRandomPatient() {
  const sexes = ["M", "F"];
  const chestPains = ["ASY", "ATA", "NAP", "TA"];
  const ecgs = ["Normal", "ST", "LVH"];
  const slopes = ["Up", "Flat", "Down"];
  const yesNos = ["Y", "N"];

  const random = {
    Age: Math.floor(Math.random() * (75 - 35) + 35),
    Sex: sexes[Math.floor(Math.random() * sexes.length)],
    ChestPainType: chestPains[Math.floor(Math.random() * chestPains.length)],
    RestingBP: Math.floor(Math.random() * (175 - 110) + 110),
    Cholesterol: Math.floor(Math.random() * (320 - 150) + 150),
    FastingBS: Math.random() > 0.75 ? 1 : 0,
    RestingECG: ecgs[Math.floor(Math.random() * ecgs.length)],
    MaxHR: Math.floor(Math.random() * (185 - 95) + 95),
    ExerciseAngina: yesNos[Math.floor(Math.random() * yesNos.length)],
    Oldpeak: +(Math.random() * 3.5).toFixed(1),
    ST_Slope: slopes[Math.floor(Math.random() * slopes.length)]
  };

  PRESETS["random"] = random;
  loadPreset("random");
}

// ==========================================================================
// Chart.js Visualizations (Interactive EDA)
// ==========================================================================
function initializeTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach(t => {
    t.addEventListener("click", () => {
      tabs.forEach(b => b.classList.remove("active"));
      t.classList.add("active");

      const targetId = t.getAttribute("data-tab");
      document.querySelectorAll(".tab-content").forEach(content => {
        content.classList.remove("active");
      });

      const activeContent = document.getElementById(targetId);
      if (activeContent) activeContent.classList.add("active");
    });
  });
}

function initializeCharts() {
  // Chart 1: MaxHR Density Distribution
  const ctxMaxHR = document.getElementById("chartMaxHR");
  if (ctxMaxHR) {
    STATE.charts.maxHR = new Chart(ctxMaxHR, {
      type: "line",
      data: {
        labels: [60, 80, 100, 120, 140, 160, 180, 200],
        datasets: [
          {
            label: "Disease Positive (HeartDisease = 1)",
            data: [0.002, 0.008, 0.016, 0.024, 0.018, 0.009, 0.003, 0.001],
            borderColor: "#EA580C",
            backgroundColor: "rgba(234, 88, 12, 0.15)",
            fill: true,
            tension: 0.4,
            borderWidth: 2.5
          },
          {
            label: "Normal / Healthy (HeartDisease = 0)",
            data: [0.001, 0.003, 0.007, 0.014, 0.022, 0.026, 0.015, 0.003],
            borderColor: "#10B981",
            backgroundColor: "rgba(16, 185, 129, 0.12)",
            fill: true,
            tension: 0.4,
            borderWidth: 2.5
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: { title: { display: true, text: "Maximum Heart Rate Achieved (bpm)" } },
          y: { title: { display: true, text: "Estimated Density" } }
        }
      }
    });
  }

  // Chart 2: Categorical Chest Pain Prevalence
  const ctxCP = document.getElementById("chartChestPain");
  if (ctxCP) {
    STATE.charts.cp = new Chart(ctxCP, {
      type: "bar",
      data: {
        labels: ["ASY (Asymptomatic)", "NAP (Non-Anginal)", "ATA (Atypical Angina)", "TA (Typical Angina)"],
        datasets: [
          {
            label: "Heart Disease Present",
            data: [392, 72, 24, 20],
            backgroundColor: "#EA580C"
          },
          {
            label: "No Heart Disease",
            data: [104, 131, 149, 26],
            backgroundColor: "#FDBA74"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "top" },
          tooltip: { mode: "index", intersect: false }
        },
        scales: {
          x: { stacked: false, title: { display: true, text: "Chest Pain Classification" } },
          y: { title: { display: true, text: "Patient Count" } }
        }
      }
    });
  }

  // Chart 3: ST Slope Distribution
  const ctxSlope = document.getElementById("chartSTSlope");
  if (ctxSlope) {
    STATE.charts.slope = new Chart(ctxSlope, {
      type: "bar",
      data: {
        labels: ["Flat Slope", "Up-sloping", "Down-sloping"],
        datasets: [
          {
            label: "Disease Positive (1)",
            data: [381, 78, 49],
            backgroundColor: "#C2410C"
          },
          {
            label: "Normal (0)",
            data: [79, 317, 14],
            backgroundColor: "#10B981"
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { title: { display: true, text: "ST Segment Slope" } },
          y: { title: { display: true, text: "Patient Cohort Count" } }
        }
      }
    });
  }

  // Chart 4: Correlation with Heart Disease
  const ctxCorr = document.getElementById("chartCorrelation");
  if (ctxCorr) {
    STATE.charts.corr = new Chart(ctxCorr, {
      type: "bar",
      data: {
        labels: ["ST_Slope_Flat", "ChestPain_ASY", "Oldpeak", "ExerciseAngina", "Age", "RestingBP", "FastingBS", "MaxHR (Neg)", "ST_Slope_Up (Neg)"],
        datasets: [{
          label: "Pearson Correlation Coefficient (r)",
          data: [0.55, 0.52, 0.40, 0.49, 0.28, 0.11, 0.27, -0.40, -0.62],
          backgroundColor: (ctx) => {
            const val = ctx.raw;
            return val > 0 ? "#EA580C" : "#059669";
          }
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          x: { min: -0.7, max: 0.7, title: { display: true, text: "Correlation with Heart Disease (r)" } }
        }
      }
    });
  }
}

// ==========================================================================
// Benchmarks Data Loading
// ==========================================================================
function loadBenchmarksData() {
  fetch("/api/benchmarks")
    .then(res => res.json())
    .then(data => {
      STATE.benchmarks = data;
    })
    .catch(() => {
      console.log("Using embedded benchmark matrix (offline mode).");
    });
}

// ==========================================================================
// Batch Evaluation & CSV Loader
// ==========================================================================
function loadSampleBatchCohort() {
  fetchSampleBatchData();
  initializeCSVListeners();
}

window.loadSampleBatchCohort = function() {
  fetchSampleBatchData();
};

const DEMO_COHORT_20 = [
  { Age: 40, Sex: "M", ChestPainType: "ATA", RestingBP: 140, Cholesterol: 289, FastingBS: 0, RestingECG: "Normal", MaxHR: 172, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 49, Sex: "F", ChestPainType: "NAP", RestingBP: 160, Cholesterol: 180, FastingBS: 0, RestingECG: "Normal", MaxHR: 156, ExerciseAngina: "N", Oldpeak: 1.0, ST_Slope: "Flat" },
  { Age: 37, Sex: "M", ChestPainType: "ATA", RestingBP: 130, Cholesterol: 283, FastingBS: 0, RestingECG: "ST", MaxHR: 98, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 48, Sex: "F", ChestPainType: "ASY", RestingBP: 138, Cholesterol: 214, FastingBS: 0, RestingECG: "Normal", MaxHR: 108, ExerciseAngina: "Y", Oldpeak: 1.5, ST_Slope: "Flat" },
  { Age: 54, Sex: "M", ChestPainType: "NAP", RestingBP: 150, Cholesterol: 195, FastingBS: 0, RestingECG: "Normal", MaxHR: 122, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 58, Sex: "M", ChestPainType: "ASY", RestingBP: 140, Cholesterol: 260, FastingBS: 1, RestingECG: "Normal", MaxHR: 118, ExerciseAngina: "Y", Oldpeak: 2.2, ST_Slope: "Flat" },
  { Age: 42, Sex: "F", ChestPainType: "ATA", RestingBP: 120, Cholesterol: 210, FastingBS: 0, RestingECG: "Normal", MaxHR: 170, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 64, Sex: "M", ChestPainType: "ASY", RestingBP: 145, Cholesterol: 245, FastingBS: 1, RestingECG: "LVH", MaxHR: 105, ExerciseAngina: "Y", Oldpeak: 2.8, ST_Slope: "Down" },
  { Age: 51, Sex: "M", ChestPainType: "NAP", RestingBP: 125, Cholesterol: 230, FastingBS: 0, RestingECG: "Normal", MaxHR: 150, ExerciseAngina: "N", Oldpeak: 0.5, ST_Slope: "Up" },
  { Age: 60, Sex: "M", ChestPainType: "ASY", RestingBP: 150, Cholesterol: 240, FastingBS: 0, RestingECG: "ST", MaxHR: 120, ExerciseAngina: "Y", Oldpeak: 1.8, ST_Slope: "Flat" },
  { Age: 45, Sex: "F", ChestPainType: "TA", RestingBP: 130, Cholesterol: 235, FastingBS: 0, RestingECG: "Normal", MaxHR: 165, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 67, Sex: "M", ChestPainType: "ASY", RestingBP: 160, Cholesterol: 285, FastingBS: 1, RestingECG: "LVH", MaxHR: 108, ExerciseAngina: "Y", Oldpeak: 3.0, ST_Slope: "Flat" },
  { Age: 52, Sex: "F", ChestPainType: "ATA", RestingBP: 128, Cholesterol: 205, FastingBS: 0, RestingECG: "Normal", MaxHR: 158, ExerciseAngina: "N", Oldpeak: 0.2, ST_Slope: "Up" },
  { Age: 59, Sex: "M", ChestPainType: "ASY", RestingBP: 135, Cholesterol: 0, FastingBS: 0, RestingECG: "Normal", MaxHR: 115, ExerciseAngina: "Y", Oldpeak: 1.6, ST_Slope: "Flat" },
  { Age: 43, Sex: "M", ChestPainType: "NAP", RestingBP: 120, Cholesterol: 225, FastingBS: 0, RestingECG: "Normal", MaxHR: 162, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 63, Sex: "F", ChestPainType: "ASY", RestingBP: 150, Cholesterol: 270, FastingBS: 0, RestingECG: "ST", MaxHR: 112, ExerciseAngina: "Y", Oldpeak: 2.0, ST_Slope: "Down" },
  { Age: 39, Sex: "M", ChestPainType: "ATA", RestingBP: 122, Cholesterol: 215, FastingBS: 0, RestingECG: "Normal", MaxHR: 175, ExerciseAngina: "N", Oldpeak: 0.0, ST_Slope: "Up" },
  { Age: 56, Sex: "M", ChestPainType: "ASY", RestingBP: 142, Cholesterol: 250, FastingBS: 1, RestingECG: "Normal", MaxHR: 125, ExerciseAngina: "Y", Oldpeak: 1.4, ST_Slope: "Flat" },
  { Age: 47, Sex: "F", ChestPainType: "NAP", RestingBP: 130, Cholesterol: 190, FastingBS: 0, RestingECG: "Normal", MaxHR: 148, ExerciseAngina: "N", Oldpeak: 0.8, ST_Slope: "Up" },
  { Age: 65, Sex: "M", ChestPainType: "ASY", RestingBP: 170, Cholesterol: 265, FastingBS: 1, RestingECG: "LVH", MaxHR: 92, ExerciseAngina: "Y", Oldpeak: 3.5, ST_Slope: "Down" }
];

function fetchSampleBatchData() {
  fetch("/api/sample-patients")
    .then(res => res.json())
    .then(data => {
      STATE.batchPatients = data.patients || DEMO_COHORT_20;
      renderBatchTable(STATE.batchPatients);
    })
    .catch(() => {
      // Offline / Static deployment fallback
      STATE.batchPatients = DEMO_COHORT_20;
      renderBatchTable(DEMO_COHORT_20);
    });
}

function renderBatchTable(patients) {
  const tbody = document.getElementById("batchTableBody");
  if (!tbody) return;

  tbody.innerHTML = "";
  let totalEvaluated = 0;
  let highRiskCount = 0;

  patients.forEach((p, idx) => {
    totalEvaluated++;
    // Run inference for each patient
    const evalData = runPatientEval(p);
    const prob = evalData.prob;
    const isHigh = prob >= 45.0;
    if (isHigh) highRiskCount++;

    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>#${idx + 1}</strong></td>
      <td>${p.Age}</td>
      <td><span class="badge ${p.Sex === 'M' ? 'badge-orange' : 'badge-green'}">${p.Sex}</span></td>
      <td>${p.ChestPainType}</td>
      <td>${p.RestingBP} / ${p.Cholesterol}</td>
      <td>${p.MaxHR} bpm</td>
      <td>${p.ExerciseAngina}</td>
      <td>${p.Oldpeak} mm</td>
      <td>${p.ST_Slope}</td>
      <td><span class="mono" style="font-weight:700; color:${isHigh ? 'var(--orange-600)' : 'var(--status-low)'}">${prob}%</span></td>
      <td><span class="badge ${isHigh ? 'badge-red' : 'badge-green'}">${isHigh ? 'HIGH RISK' : 'LOW RISK'}</span></td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="loadPatientIntoCalculator(${idx})">
          <i data-lucide="arrow-up-right" style="width:14px;height:14px;"></i> Load
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });

  // Update cohort stats
  setText("batchTotalCount", totalEvaluated);
  setText("batchHighRiskCount", highRiskCount);
  setText("batchPositivityRate", totalEvaluated > 0 ? `${((highRiskCount / totalEvaluated) * 100).toFixed(1)}%` : "0%");

  // Refresh lucide icons
  if (window.lucide) lucide.createIcons();
}

function runPatientEval(p) {
  const raw = {
    Age: parseFloat(p.Age) || 54,
    Sex: (p.Sex || "M").toUpperCase(),
    ChestPainType: (p.ChestPainType || "ASY").toUpperCase(),
    RestingBP: parseFloat(p.RestingBP) || 130,
    Cholesterol: parseFloat(p.Cholesterol) || 220,
    FastingBS: parseInt(p.FastingBS) || 0,
    RestingECG: p.RestingECG || "Normal",
    MaxHR: parseFloat(p.MaxHR) || 138,
    ExerciseAngina: (p.ExerciseAngina || "N").toUpperCase(),
    Oldpeak: parseFloat(p.Oldpeak) || 0.6,
    ST_Slope: p.ST_Slope || "Flat"
  };

  const oldpeakPos = Math.max(0, raw.Oldpeak);
  const sexNum = raw.Sex === "M" ? 1 : 0;
  const exerciseAnginaNum = raw.ExerciseAngina === "Y" ? 1 : 0;
  const stSlopeOrd = raw.ST_Slope === "Flat" ? 1 : (raw.ST_Slope === "Down" ? 2 : 0);
  const chestPainASY = raw.ChestPainType === "ASY" ? 1 : 0;

  const theoreticalMaxHR = 220.0 - raw.Age;
  const pctMaxHRAchieved = raw.MaxHR / (theoreticalMaxHR + 1e-5);
  const chronotropicIncomp = pctMaxHRAchieved < 0.80 ? 1 : 0;

  const ischemiaScore = (oldpeakPos * 2.0) + (stSlopeOrd * 1.5) + (exerciseAnginaNum * 1.5) + (chestPainASY * 1.2);

  const slopeTerm = stSlopeOrd === 1 ? 0.55 : (stSlopeOrd === 2 ? 0.90 : -0.75);
  const cpTerm = chestPainASY ? 0.60 : (raw.ChestPainType === "ATA" ? -0.75 : (raw.ChestPainType === "NAP" ? -0.30 : 0.15));
  const anginaTerm = exerciseAnginaNum ? 0.50 : -0.50;
  const oldpeakTerm = (oldpeakPos - 0.90) * 0.45;
  const hrTerm = -0.016 * (raw.MaxHR - 138.0) + (chronotropicIncomp ? 0.35 : -0.15);
  const bpTerm = (raw.RestingBP - 130.0) * 0.018;
  const ageTerm = (raw.Age - 53.0) * 0.038;
  const sexTerm = sexNum ? 0.45 : -0.45;
  const cholTerm = ((raw.Cholesterol <= 0 ? (sexNum ? 242.0 : 235.0) : raw.Cholesterol) - 220.0) * 0.005;
  const fbsTerm = raw.FastingBS ? 0.35 : -0.20;
  const dukeTerm = (ischemiaScore - 3.5) * 0.12;

  const intercept = 0.12;
  const rawLogOdds = intercept + slopeTerm + cpTerm + anginaTerm + oldpeakTerm +
                     hrTerm + bpTerm + ageTerm + sexTerm + cholTerm + fbsTerm + dukeTerm;
  const scaledLogOdds = rawLogOdds * 0.55;
  const prob = 1.0 / (1.0 + Math.exp(-scaledLogOdds));

  return {
    prob: (Math.max(0.02, Math.min(0.98, prob)) * 100).toFixed(1),
    ischemiaScore: ischemiaScore.toFixed(1)
  };
}

window.loadPatientIntoCalculator = function(idx) {
  const patient = STATE.batchPatients[idx];
  if (!patient) return;

  setFormValue("inputAge", "valAge", patient.Age);
  setRadioValue("selectSex", patient.Sex);
  setRadioValue("selectChestPain", patient.ChestPainType);
  setFormValue("inputRestingBP", "valRestingBP", patient.RestingBP);
  setFormValue("inputCholesterol", "valCholesterol", patient.Cholesterol);
  setRadioValue("selectFastingBS", patient.FastingBS);
  setRadioValue("selectRestingECG", patient.RestingECG);
  setFormValue("inputMaxHR", "valMaxHR", patient.MaxHR);
  setRadioValue("selectExerciseAngina", patient.ExerciseAngina);
  setFormValue("inputOldpeak", "valOldpeak", patient.Oldpeak);
  setRadioValue("selectSTSlope", patient.ST_Slope);

  runClientInference();

  // Scroll to calculator
  const calc = document.getElementById("calculator");
  if (calc) calc.scrollIntoView({ behavior: "smooth" });
};

// Drag and Drop & CSV Upload Initializer
function initializeCSVListeners() {
  const fileInput = document.getElementById("csvFileInput");
  if (fileInput) {
    fileInput.addEventListener("change", handleCSVUpload);
  }

  const dropZone = document.getElementById("dropZone");
  if (dropZone) {
    ["dragenter", "dragover"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = "var(--orange-600)";
        dropZone.style.backgroundColor = "var(--orange-50)";
      }, false);
    });

    ["dragleave", "drop"].forEach(eventName => {
      dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.style.borderColor = "var(--border-medium)";
        dropZone.style.backgroundColor = "var(--bg-surface)";
      }, false);
    });

    dropZone.addEventListener("drop", (e) => {
      const dt = e.dataTransfer;
      const files = dt.files;
      if (files.length > 0) {
        processUploadedCSVFile(files[0]);
      }
    }, false);
  }

  const btnExportBatch = document.getElementById("btnExportBatchCSV");
  if (btnExportBatch) {
    btnExportBatch.addEventListener("click", exportBatchCSV);
  }
}

// CSV File Upload Handler
function handleCSVUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  processUploadedCSVFile(file);
}

function processUploadedCSVFile(file) {
  const reader = new FileReader();
  reader.onload = function(evt) {
    const text = evt.target.result;
    const rows = parseCSVText(text);
    if (rows && rows.length > 0) {
      STATE.batchPatients = rows;
      renderBatchTable(rows);
      alert(`Successfully parsed and evaluated ${rows.length} patient records from ${file.name}!`);
    } else {
      alert("Could not find valid patient records in the uploaded CSV. Please check headers.");
    }
  };
  reader.readAsText(file);
}

function normalizeCSVHeader(h) {
  const clean = h.trim().toLowerCase().replace(/[_ \-]/g, "");
  const map = {
    "age": "Age",
    "sex": "Sex",
    "gender": "Sex",
    "chestpain": "ChestPainType",
    "chestpaintype": "ChestPainType",
    "cpt": "ChestPainType",
    "cp": "ChestPainType",
    "restingbp": "RestingBP",
    "bp": "RestingBP",
    "bloodpressure": "RestingBP",
    "trestbps": "RestingBP",
    "cholesterol": "Cholesterol",
    "chol": "Cholesterol",
    "serumcholesterol": "Cholesterol",
    "fastingbs": "FastingBS",
    "fbs": "FastingBS",
    "bloodsugar": "FastingBS",
    "restingecg": "RestingECG",
    "ecg": "RestingECG",
    "restecg": "RestingECG",
    "maxhr": "MaxHR",
    "thalach": "MaxHR",
    "heartrate": "MaxHR",
    "exerciseangina": "ExerciseAngina",
    "exang": "ExerciseAngina",
    "angina": "ExerciseAngina",
    "oldpeak": "Oldpeak",
    "stdepression": "Oldpeak",
    "stslope": "ST_Slope",
    "slope": "ST_Slope"
  };
  return map[clean] || h.trim();
}

function parseCSVText(csvText) {
  const lines = csvText.split(/\r\n|\n/).filter(line => line.trim().length > 0);
  if (lines.length < 2) return [];

  const rawHeaders = lines[0].split(",").map(h => h.trim().replace(/^["']|["']$/g, ""));
  const headers = rawHeaders.map(normalizeCSVHeader);
  const data = [];

  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(",").map(v => v.trim().replace(/^["']|["']$/g, ""));
    if (values.length >= headers.length) {
      const obj = {};
      headers.forEach((h, index) => {
        obj[h] = values[index];
      });
      data.push(obj);
    }
  }
  return data;
}

window.downloadSampleTemplateCSV = function() {
  const headers = "Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope\n";
  const rows = [
    "58,M,ASY,140,260,1,Normal,118,Y,2.2,Flat",
    "41,F,ATA,118,195,0,Normal,175,N,0.0,Up",
    "64,M,ASY,148,245,1,LVH,105,Y,2.8,Down",
    "49,M,NAP,135,230,0,ST,138,Y,1.2,Flat",
    "59,M,ASY,135,0,0,Normal,115,Y,1.6,Flat",
    "52,F,ATA,125,210,0,Normal,168,N,0.2,Up",
    "61,M,ASY,152,280,1,Normal,124,Y,2.0,Flat",
    "44,M,NAP,130,225,0,Normal,155,N,0.4,Up"
  ].join("\n");

  const csvContent = "data:text/csv;charset=utf-8," + headers + rows;
  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "CardioPulse_Cohort_Template.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

function exportBatchCSV() {
  if (!STATE.batchPatients || STATE.batchPatients.length === 0) {
    alert("No batch patient data available to export.");
    return;
  }

  let csvContent = "data:text/csv;charset=utf-8,";
  csvContent += "PatientID,Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope,PredictedRiskProb,TunedDecision,RiskTier\n";

  STATE.batchPatients.forEach((p, idx) => {
    const evalData = runPatientEval(p);
    const prob = evalData.prob;
    const tier = prob >= 75 ? "Critical Risk" : (prob >= 45 ? "High Risk" : "Low Risk");
    const decision = prob >= 37.1 ? "HeartDisease (1)" : "Normal (0)";

    const row = [
      idx + 1, p.Age, p.Sex, p.ChestPainType, p.RestingBP, p.Cholesterol,
      p.FastingBS, p.RestingECG, p.MaxHR, p.ExerciseAngina, p.Oldpeak, p.ST_Slope,
      `${prob}%`, decision, tier
    ].join(",");
    csvContent += row + "\n";
  });

  const encodedUri = encodeURI(csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "CardioPulse_Cohort_Predictions.csv");
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

// ==========================================================================
// Printable Clinical Assessment Report Modal
// ==========================================================================
function openPrintReportModal() {
  const p = STATE.currentPatient;
  if (!p) return;

  const modal = document.getElementById("reportModal");
  if (!modal) return;

  // Fill in report content
  setText("repDate", new Date().toLocaleDateString("en-US", { year: 'numeric', month: 'long', day: 'numeric', hour: '2-digit', minute: '2-digit' }));
  setText("repPatientID", `CP-${Math.floor(100000 + Math.random() * 900000)}`);
  setText("repAgeSex", `${p.raw.Age} Years / ${p.raw.Sex === 'M' ? 'Male' : 'Female'}`);
  setText("repRiskProb", `${p.probPercent}%`);
  setText("repRiskTier", p.riskLevel.toUpperCase());

  setText("repVitals", `BP: ${p.raw.RestingBP} mmHg | Chol: ${p.raw.Cholesterol} mg/dL | Fasting BS > 120: ${p.raw.FastingBS == 1 ? 'Yes' : 'No'}`);
  setText("repStress", `Max HR: ${p.raw.MaxHR} bpm (${p.biomarkers.pctMaxHRAchieved}% of max) | Exercise Angina: ${p.raw.ExerciseAngina}`);
  setText("repECG", `Resting ECG: ${p.raw.RestingECG} | ST Depression: ${p.raw.Oldpeak} mm | ST Slope: ${p.raw.ST_Slope}`);
  setText("repIschemiaScore", `Duke Severity Score: ${p.biomarkers.ischemiaScore} | Rate-Pressure Product: ${p.biomarkers.ratePressureProduct}`);
  setText("repGuidance", p.actionText);

  // Top SHAP drivers
  const repDrivers = document.getElementById("repTopDrivers");
  if (repDrivers) {
    repDrivers.innerHTML = "";
    p.shap.slice(0, 4).forEach(item => {
      const li = document.createElement("li");
      li.style.color = item.pos ? "var(--orange-700)" : "var(--status-low)";
      li.textContent = `${item.name}: ${item.impact > 0 ? '+' : ''}${item.impact}% (${item.pos ? 'Elevates Risk' : 'Protective Factor'})`;
      repDrivers.appendChild(li);
    });
  }

  modal.classList.add("active");
}

window.closeReportModal = function() {
  const modal = document.getElementById("reportModal");
  if (modal) modal.classList.remove("active");
};

window.printClinicalReport = function() {
  window.print();
};

// Research Publication Lightbox Modal
window.openLightbox = function(imgSrc, title, desc) {
  const lightbox = document.getElementById("lightboxModal");
  if (!lightbox) return;

  const img = document.getElementById("lightboxImg");
  const captionTitle = document.getElementById("lightboxTitle");
  const captionDesc = document.getElementById("lightboxDesc");

  if (img) img.src = imgSrc;
  if (captionTitle) captionTitle.textContent = title;
  if (captionDesc) captionDesc.textContent = desc;

  lightbox.classList.add("active");
};

window.closeLightbox = function() {
  const lightbox = document.getElementById("lightboxModal");
  if (lightbox) lightbox.classList.remove("active");
};

