export default function handler(req, res) {
  res.setHeader('Content-Type', 'application/json');
  res.setHeader('Access-Control-Allow-Origin', '*');

  const patients = [
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

  res.status(200).json({
    status: "success",
    cohort_source: "Fedesoriano 5-Hospital Dataset",
    count: patients.length,
    patients
  });
}
