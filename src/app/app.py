from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr
import os
import sys

# Ensure we can import from src/serving when running "uvicorn src.app.app:app"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from serving.inference import predict  # our single source of truth for inference

app = FastAPI()

@app.get("/")
def root():
    return {"status": "ok"}

# ------------------------------------------------------------------
# Request schema (matches the lab dataset features)
# ------------------------------------------------------------------
class LabData(BaseModel):
    labname: str
    gender: str
    age: str
    unittype: str
    recent_diagnosis: str | None = None
    result_time: int
    validation_time: int
    admissionweight: float | None = None
    lab_workload_last_hour: int

@app.post("/predict")
def api_predict(data: LabData):
    try:
        out = predict(data.dict())
        return out
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------------------------------------------------------
# Gradio UI wrapper (uses the same predict() function)
# ------------------------------------------------------------------
LABNAME_CHOICES = [
    "potassium", "alkaline phos.", "Hgb", "calcium", "bicarbonate", "Hct",
    "creatinine", "-eos", "-monos", "-basos", "MCH", "MCHC", "-lymphs",
    "magnesium", "total bilirubin", "TSH", "anion gap", "albumin",
    "WBC x 1000", "ALT (SGPT)", "RDW", "AST (SGOT)", "RBC", "sodium",
    "MCV", "chloride", "total protein", "-polys", "glucose",
    "platelets x 1000", "BUN", "troponin - I", "lactate", "PT", "PT - INR",
    "PTT", "BNP", "bedside glucose", "Fe/TIBC Ratio", "TIBC", "Fe",
    "Ferritin", "reticulocyte count", "folate", "Vitamin B12",
    "Vancomycin - trough", "Digoxin", "free T4", "HCO3", "paCO2", "pH",
    "FiO2", "Base Deficit", "paO2", "LPM O2", "Base Excess",
    "total cholesterol", "LDL", "HDL", "triglycerides", "CRP",
    "urinary osmolality", "urinary specific gravity", "urinary sodium",
    "urinary creatinine", "uric acid", "serum osmolality", "ethanol",
    "salicylate", "Acetaminophen", "phosphate", "ionized calcium",
    "direct bilirubin", "amylase", "lipase", "CPK", "CPK-MB",
    "fibrinogen", "-bands"
]

GENDER_CHOICES = ["Male", "Female"]

AGE_CHOICES = [
    "> 89", "78", "49", "53", "57", "34", "58", "59", "72", "73",
    "87", "47", "39", "69", "71", "60", "29", "17", "56", "55",
    "54", "52", "51", "50", "48", "46", "45", "44", "43", "42",
    "41", "40", "38", "37", "36", "35", "33", "32", "31", "30",
    "28", "27", "26", "25", "24", "23", "22", "21", "20", "19",
    "18", "16", "15", "14", "13", "12", "11", "10", "9", "8",
    "7", "6", "5", "4", "3", "2", "1", "0"
]

UNITTYPE_CHOICES = ["Med-Surg ICU"]

DIAGNOSIS_CHOICES = [
    "None",
    "pulmonary|respiratory failure|acute respiratory distress",
    "endocrine|glucose metabolism|diabetes mellitus",
    "gastrointestinal|post-GI surgery|s/p surgery for intestinal obstruction",
    "cardiovascular|ventricular disorders|hypertension",
    "cardiovascular|chest pain / ASHD|coronary artery disease",
    "renal|disorder of kidney|chronic kidney disease|Stage 3 (GFR 30-59)",
    "pulmonary|disorders of the airways|COPD",
    "pulmonary|disorders of vasculature|pulmonary embolism",
    "toxicology|drug overdose|tricyclic overdose",
    "pulmonary|respiratory failure|acute respiratory failure",
]

def gradio_interface(
    labname, gender, age, unittype, recent_diagnosis,
    result_time, validation_time, admissionweight, lab_workload_last_hour
):
    payload = {
        "labname": labname,
        "gender": gender,
        "age": age,
        "unittype": unittype,
        "recent_diagnosis": recent_diagnosis if recent_diagnosis != "None" else None,
        "result_time": int(result_time),
        "validation_time": int(validation_time),
        "admissionweight": float(admissionweight) if admissionweight is not None else None,
        "lab_workload_last_hour": int(lab_workload_last_hour),
    }
    out = predict(payload)
    if out.get("status") == "success":
        return f"Predicted Turnaround Time: {out['predicted_turnaround_time_mins']} minutes"
    else:
        return f"Error: {out.get('message', 'Unknown error')}"

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(LABNAME_CHOICES, label="Lab Test Name", value="potassium"),
        gr.Dropdown(GENDER_CHOICES, label="Gender", value="Female"),
        gr.Dropdown(AGE_CHOICES, label="Age Group", value="> 89"),
        gr.Dropdown(UNITTYPE_CHOICES, label="Unit Type", value="Med-Surg ICU"),
        gr.Dropdown(DIAGNOSIS_CHOICES, label="Recent Diagnosis", value="None"),
        gr.Number(label="Result Time (minutes from admission)", value=-5000),
        gr.Number(label="Validation Time (minutes from admission)", value=-5027),
        gr.Number(label="Admission Weight (kg)", value=48.1),
        gr.Number(label="Lab Workload Last Hour", value=31),
    ],
    outputs="text",
    title="Lab Turnaround Time Predictor",
    description="Enter lab order details to predict the expected turnaround time in minutes.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")