# app.py
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import joblib, json, os

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ── Load artifacts ─────────────────────────────────────────
model        = joblib.load("best_decision_tree.pkl")
best_thr     = float(np.load("optimal_threshold.npy"))
feature_cols = json.load(open("feature_cols.json"))                 # list of model features
fill_values  = pd.read_json("feature_medians.json", typ='series')   # medians from X_train

# ── Helpers ─────────────────────────────────────────────────
TOP_MANUAL_COLS = [
    # the ones you said matter
    "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3",
    "DAYS_BIRTH", "DAYS_EMPLOYED",
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "CNT_FAM_MEMBERS",
    "CODE_GENDER_M"
]

def add_ratios(df):
    """Recompute ratio features if raw cols exist."""
    if {'AMT_CREDIT','AMT_INCOME_TOTAL','AMT_ANNUITY','CNT_FAM_MEMBERS'}.issubset(df.columns):
        df['CREDIT_INCOME_RATIO']  = df['AMT_CREDIT']  / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
        df['ANNUITY_CREDIT_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
        df['INCOME_PER_PERSON']    = df['AMT_INCOME_TOTAL'] / (df['CNT_FAM_MEMBERS'] + 1)
    if {'AMT_GOODS_PRICE','AMT_CREDIT'}.issubset(df.columns):
        df['GOODS_CREDIT_RATIO']   = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    return df

def recompute_log1p(df):
    """If a *_LOG1P column exists and its base column exists, recompute it."""
    for col in df.columns:
        if col.endswith("_LOG1P"):
            base = col[:-6]
            if base in df.columns:
                df[col] = np.log1p(df[base])
    return df

def build_row_from_form(form):
    """Parse user inputs from the HTML form into a dict."""
    row = {}
    for c in TOP_MANUAL_COLS:
        val = form.get(c, "").strip()
        if val == "":  # leave NaN -> will be filled by medians
            continue
        # handle special fields
        if c == "CODE_GENDER_M":
            row[c] = 1 if val.upper().startswith("M") or val == "1" else 0
        elif c == "AGE_YEARS":     # if you add AGE_YEARS field
            row["DAYS_BIRTH"] = -float(val) * 365
        else:
            row[c] = float(val)
    return row

def predict_df(df_user):
    """
    Start from median row -> override with user values -> recompute engineered stuff.
    Ensures column order = feature_cols.
    """
    # base = 1-row DF of medians
    base = pd.DataFrame([fill_values], columns=feature_cols).copy()
    # override with user-provided values
    base.update(df_user)

    # recompute engineered cols that depend on raw inputs
    base = add_ratios(base)
    base = recompute_log1p(base)

    # final align
    base = base.reindex(columns=feature_cols)
    proba = model.predict_proba(base)[:, 1]
    pred  = (proba >= best_thr).astype(int)
    return pred, proba

# ── Routes ──────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # ---- Manual single-row form ----
        if "manual" in request.form:
            try:
                row_dict = build_row_from_form(request.form)
                df_in = pd.DataFrame([row_dict])
                pred, proba = predict_df(df_in)

                return render_template(
                    "index.html",
                    single_result=True,
                    pred=int(pred[0]),
                    proba=float(proba[0]),
                    thr=best_thr
                )
            except Exception as e:
                return render_template("index.html", error=str(e), thr=best_thr)

        # ---- CSV upload path ----
        file = request.files.get("file")
        if file and file.filename.lower().endswith(".csv"):
            path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(path)

            df_in = pd.read_csv(path)
            pred, proba = predict_df(df_in)

            out = df_in.copy()
            out["PROB_DEFAULT"] = proba
            out["PRED_DEFAULT"] = pred
            out_path = path.replace(".csv", "_pred.csv")
            out.to_csv(out_path, index=False)

            return render_template(
                "index.html",
                csv_done=True,
                out_file=os.path.basename(out_path),
                thr=best_thr
            )

        return render_template("index.html", error="Please upload a CSV.", thr=best_thr)

    # GET
    return render_template("index.html", thr=best_thr)

@app.route("/download/<path:fname>")
def download(fname):
    return send_from_directory(app.config["UPLOAD_FOLDER"], fname, as_attachment=True)

# ── Main ────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True)
