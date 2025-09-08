from flask import Flask, render_template, request
import joblib, json, numpy as np, os, math

# Load metadata and model
with open("metadata.json", "r", encoding="utf-8") as f:
    META = json.load(f)

FEATURES = META["features"]
MODEL_PATH = "model.joblib"
MODEL = joblib.load(MODEL_PATH)

app = Flask(__name__)

def is_float_str(s):
    try:
        float(s)
        return True
    except:
        return False

@app.route("/", methods=["GET"])
def home():
    # blank input values
    input_values = {f: "" for f in FEATURES}
    return render_template("index.html", features=FEATURES, input_values=input_values, empty_fields=[], invalid_fields=[], result=None, proba=None, meta=META)

@app.route("/", methods=["POST"])
def predict():
    input_values = {}
    empty_fields = []
    invalid_fields = []
    # collect values as strings to re-display
    for f in FEATURES:
        v = request.form.get(f, "")
        if v is None:
            v = ""
        v = v.strip()
        input_values[f] = v
        if v == "":
            empty_fields.append(f)
    # server-side check for empty
    if empty_fields:
        return render_template("index.html", features=FEATURES, input_values=input_values, empty_fields=empty_fields, invalid_fields=invalid_fields, result=None, proba=None, meta=META, message=f"Vui lòng điền đầy đủ ({len(empty_fields)} ô trống)")
    # check numeric per field
    for f, v in input_values.items():
        if not is_float_str(v):
            invalid_fields.append(f)
    if invalid_fields:
        return render_template("index.html", features=FEATURES, input_values=input_values, empty_fields=empty_fields, invalid_fields=invalid_fields, result=None, proba=None, meta=META, message=f"Các ô sau không hợp lệ: {', '.join(invalid_fields)}")
    # convert to float array
    try:
        arr = np.array([[float(input_values[f]) for f in FEATURES]], dtype=float)
    except Exception as e:
        return render_template("index.html", features=FEATURES, input_values=input_values, empty_fields=empty_fields, invalid_fields=invalid_fields, result=None, proba=None, meta=META, message=f"Lỗi khi chuyển sang số: {e}")
    # prediction
    pred = int(MODEL.predict(arr)[0])
    proba = None
    if hasattr(MODEL, "predict_proba"):
        proba = float(MODEL.predict_proba(arr)[0,1])
    label = "Ung thư ác tính" if pred == 1 else "Không ác tính / Lành tính"
    return render_template("index.html", features=FEATURES, input_values=input_values, empty_fields=empty_fields, invalid_fields=invalid_fields, result=label, proba=proba, meta=META, message=None)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
