# app.py  ── Retina-analysis API, OpenAI SDK ≥ 1.55.3, with medical-disclaimer prompt
from __future__ import annotations
import base64, json, os, tempfile
from pathlib import Path
from typing import Any, Dict
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from openai import OpenAI

# ──────────────────────────────────────────────── setup
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))                    # picks up OPENAI_API_KEY from env
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────── disorders lookup table
disorders_data = pd.DataFrame({
    # (same 15-row dictionary you already had – left intact)
    "condition": [
        "Glaucoma","Glaucoma","Glaucoma","Glaucoma","Glaucoma",
        "Cataract","Cataract","Cataract","Cataract","Cataract",
        "Scarring","Scarring","Scarring","Scarring","Scarring"],
    "icd_code": [
        "H40.111","H40.112","H40.113","H40.119","H40.9",
        "H25.11","H25.9","H26.0","H25.12","H26.8",
        "H17.11","H17.12","H17.9","H17.0","H17.13"],
    "cpt_code": [
        "92133","92083","66170","92134","92014",
        "66984","92014","67028","92133","99024",
        "65755","65435","92072","92004","92310"],
    "prescription": [
        "Latanoprost eye drops","Timolol eye drops","Trabeculectomy surgery",
        "Prostaglandin analog combo","Combination drops",
        "Phacoemulsification surgery","Prescription update","Intraocular lens implant",
        "Anti-inflammatory drops","Vitamin supplements",
        "Corneal transplant","PRK laser treatment","Steroid eye drops",
        "Lubricating eye drops","Bandage contact lens"],
    "severity": [
        "10247","10245","10246","10247","10245",
        "10247","10246","10245","10247","10246",
        "10246","10245","10247","10247","10245"],
    "SOD": [
        "Chronic","Acute","Chronic","Chronic","Acute",
        "Chronic","Chronic","Acute","Chronic","Chronic",
        "Chronic","Acute","Chronic","Chronic","Acute"],
    "diagnosis_status": [
        "Active","Relapse","Active","Active","Relapse",
        "Active","Relapse","Relapse","Active","Active",
        "Active","Relapse","Active","Active","Relapse"],
    "symptom1":[
        "Elevated intra-ocular pressure","Eye pain","Progressive visual-field loss",
        "Night-vision difficulty","Sudden vision blur",
        "Blurry vision","Cloudy lens","Severe vision loss",
        "Night-driving difficulty","Vision haziness",
        "Corneal opacity","Foreign-body sensation","Milky cornea",
        "Dryness","Pain"],
    "symptom2":[
        "Peripheral vision loss","Halos around lights","Headache",
        "Optic nerve damage","Eye redness",
        "Halos around lights","Frequent Rx changes","Double vision",
        "Glare","Eye strain",
        "Blurred vision","Tearing","Vision loss",
        "Discomfort","Scar progression"],
    "symptom3":[
        "Optic nerve cupping","Blurred vision","Pressure spikes",
        "Thinning visual field","Eyelid swelling",
        "Glare sensitivity","Color fading","Light sensitivity",
        "Halos","Gradual decline",
        "Photophobia","Light sensitivity","Irritation",
        "Visual distortion","Blurred vision"],
})

# ───────────────────────────────────────────── helper functions
SYSTEM_PROMPT = (
    "You are an ophthalmology-research assistant. "
    "Given a retina image, list the *most likely* category out of "
    "glaucoma, cataract, scarring, or healthy **for research purposes only**. "
    "You are **not** providing a medical diagnosis. "
    "Always remind the user to consult a qualified eye-care professional.\n\n"
    "Respond *only* with JSON exactly like:\n"
    '{"condition": "<condition>", "confidence": <number 0-1>, '
    '"disclaimer": "This is not a medical diagnosis. Consult a professional."}'
)

REFUSAL_STRINGS = ("i’m unable", "i am unable", "sorry", "cannot analyze")

def extract_json(text: str) -> Dict[str, Any]:
    start, end = text.find("{"), text.rfind("}") + 1
    if 0 <= start < end:
        try:
            data = json.loads(text[start:end])
            data["condition"] = data.get("condition", "").lower()
            return data
        except json.JSONDecodeError:
            pass
    # model refused or couldn’t parse
    return {"condition": "unknown", "confidence": 0.0,
            "disclaimer": "Model could not analyze image."}

def details_for(cond: str) -> Dict[str, Any]:
    row = disorders_data[disorders_data["condition"].str.lower() == cond].head(1)
    if row.empty:
        return {}
    r = row.iloc[0]
    return dict(
        icd_code=r.icd_code, cpt_code=r.cpt_code, prescription=r.prescription,
        severity=r.severity, status=r.diagnosis_status,
        symptoms=[r.symptom1, r.symptom2, r.symptom3],
    )

def analyse_image(b64: str, tag: str) -> Dict[str, Any]:
    b64 = b64.split(",", 1)[-1]                 # strip data-URI header if present
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(base64.b64decode(b64))
        img_path = Path(tmp.name)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=150,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": f"Research classification request for {tag}."},
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ],
                },
            ],
        )
        reply = response.choices[0].message.content
        print(reply)
        if any(s in reply.lower() for s in REFUSAL_STRINGS):
            return {"condition": "unknown", "confidence": 0.0,
                    "disclaimer": "Model refused to analyze image."}

        result = extract_json(reply)
        cond = result.get("condition", "")
        if cond and cond not in ("healthy", "unknown"):
            result |= details_for(cond)
        return result
    finally:
        img_path.unlink(missing_ok=True)

def systemic(l: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, Any]:
    dia = card = 0
    for res in (l, r):
        c = res.get("condition")
        if c == "glaucoma":
            dia += 1; card += 1
        elif c == "cataract":
            dia += 1
        elif c == "scarring":
            card += .5
    def risk(score: float) -> tuple[str, float]:
        if score >= 1.5: return "high", 0.85
        if score >= 1.0: return "moderate", 0.70
        return "low", 0.80
    d_risk, d_conf = risk(dia)
    c_risk, c_conf = risk(card)
    return {"diabetes": {"risk": d_risk, "confidence": d_conf},
            "cardiovascular": {"risk": c_risk, "confidence": c_conf}}

# ───────────────────────────────────────────── Flask route
@app.post("/api/analyze")
def analyze_retina():
    data = request.get_json(silent=True) or {}
    if not {"leftEyeImage", "rightEyeImage"} <= data.keys():
        return jsonify({"error": "Missing images"}), 400
    left  = analyse_image(data["leftEyeImage"],  "left eye")
    right = analyse_image(data["rightEyeImage"], "right eye")
    return jsonify({"leftEye": left,
                    "rightEye": right,
                    "systemicFindings": systemic(left, right)})

# ───────────────────────────────────────────── run dev server
if __name__ == "__main__":
    app.run(debug=True, port=5000)
