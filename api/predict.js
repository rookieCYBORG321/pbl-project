export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const data = req.method === 'POST' ? req.body : req.query;
  const p = typeof data === 'string' ? JSON.parse(data) : (data || {});

  const age = parseFloat(p.Age) || 54;
  const sex = (p.Sex || "M").toUpperCase();
  const chestPain = (p.ChestPainType || "ASY").toUpperCase();
  const rawBP = parseFloat(p.RestingBP) || 130;
  const rawChol = parseFloat(p.Cholesterol) || 220;
  const fastingBS = parseInt(p.FastingBS) || 0;
  const restingECG = p.RestingECG || "Normal";
  const maxHR = parseFloat(p.MaxHR) || 138;
  const exerciseAngina = (p.ExerciseAngina || "N").toUpperCase();
  const oldpeak = parseFloat(p.Oldpeak) || 0.0;
  const stSlope = p.ST_Slope || "Flat";

  const cholIsZero = rawChol <= 0 ? 1 : 0;
  const cholesterol = cholIsZero ? (sex === "M" ? 242.0 : 235.0) : rawChol;
  const restingBP = rawBP <= 0 ? 130.0 : rawBP;
  const oldpeakPos = Math.max(0.0, oldpeak);
  const sexNum = sex === "M" ? 1 : 0;
  const exerciseAnginaNum = exerciseAngina === "Y" ? 1 : 0;

  const stSlopeMap = { "Up": 0, "Flat": 1, "Down": 2 };
  const stSlopeOrd = stSlopeMap[stSlope] !== undefined ? stSlopeMap[stSlope] : 1;
  const chestPainASY = chestPain === "ASY" ? 1 : 0;

  const theoreticalMaxHR = 220.0 - age;
  const hrReserve = theoreticalMaxHR - maxHR;
  const pctMaxHRAchieved = maxHR / (theoreticalMaxHR + 1e-5);
  const chronotropicIncomp = pctMaxHRAchieved < 0.80 ? 1 : 0;

  const ischemiaScore = (oldpeakPos * 2.0) + (stSlopeOrd * 1.5) + (exerciseAnginaNum * 1.5) + (chestPainASY * 1.2);
  const ratePressureProduct = (restingBP * maxHR) / 1000.0;
  const metabolicCardioRisk = (restingBP >= 135 ? 1 : 0) + (cholesterol >= 240 ? 1 : 0) + fastingBS + sexNum;

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
  const probability = Math.max(0.02, Math.min(0.98, 1.0 / (1.0 + Math.exp(-scaledLogOdds))));

  const riskLevel = probability < 0.25 ? "Low Risk" : (probability < 0.45 ? "Moderate Risk" : (probability < 0.75 ? "High Risk" : "Critical Risk"));

  res.status(200).json({
    probability: +probability.toFixed(4),
    probability_percent: +(probability * 100).toFixed(1),
    risk_level: riskLevel,
    default_cutoff_decision: probability >= 0.50 ? 1 : 0,
    tuned_cutoff_decision: probability >= 0.371 ? 1 : 0,
    optimal_cutoff: 0.371,
    biomarkers: {
      theoretical_max_hr: Math.round(theoreticalMaxHR),
      hr_reserve: Math.round(hrReserve),
      pct_max_hr_achieved: +(pctMaxHRAchieved * 100).toFixed(1),
      chronotropic_incompetence: Boolean(chronotropicIncomp),
      duke_ischemia_score: +ischemiaScore.toFixed(1),
      rate_pressure_product: +ratePressureProduct.toFixed(1),
      metabolic_cardio_risk: metabolicCardioRisk
    }
  });
}
