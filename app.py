import os
import time
import json
from typing import List, Dict, Optional

import numpy as np
from fastapi import FastAPI, Depends, Header, HTTPException, status, Request
from pydantic import BaseModel
import jwt
import httpx
import tensorflow as tf
import xgboost as xgb

# =========================
# Config (env + sensible defaults)
# =========================
MODEL_DIR = os.getenv("MODEL_DIR", "models")

MI_NORM_PATH        = os.getenv("MI_NORM_PATH",        f"{MODEL_DIR}/model_mi_norm.keras")
STTC_NORM_PATH      = os.getenv("STTC_NORM_PATH",      f"{MODEL_DIR}/model_sttc_norm.keras")
MI_STTC_PATH        = os.getenv("MI_STTC_PATH",        f"{MODEL_DIR}/model_mi_sttc.keras")
MULTI_CLASS_PATH    = os.getenv("MULTI_CLASS_PATH",    f"{MODEL_DIR}/lead1_model_full_hpo_V5.keras")
META_XGB_PATH       = os.getenv("META_XGB_PATH",       f"{MODEL_DIR}/meta_classifier_xgb.json")

# Expected output label order for each base model (match training LabelEncoder for that subset)
# Defaults match your earlier notes & common alphabetical ordering.
MI_NORM_LABELS      = os.getenv("MI_NORM_LABELS",      "MI,NORM").split(",")
STTC_NORM_LABELS    = os.getenv("STTC_NORM_LABELS",    "NORM,STTC").split(",")
MI_STTC_LABELS      = os.getenv("MI_STTC_LABELS",      "MI,STTC").split(",")
MULTI_CLASS_LABELS  = os.getenv("MULTI_CLASS_LABELS",  "MI,NORM,STTC").split(",")  # adjust if your multiclass was trained in different order

# Final class order used by the meta XGB during training (LabelEncoder.classes_)
# Your eval script likely produced ["MI","NORM","STTC"] — make it explicit:
META_CLASS_LABELS   = os.getenv("META_CLASS_LABELS",   "MI,NORM,STTC").split(",")

SEQ_LEN             = int(os.getenv("SEQ_LEN", "1000"))

# JWT settings (Azure AD B2C or your issuer)
JWT_ISSUER          = os.getenv("JWT_ISSUER", "")
JWT_AUDIENCE        = os.getenv("JWT_AUDIENCE", "")
JWT_JWKS_URL        = os.getenv("JWT_JWKS_URL", "")

# Apple App Attest toggle (server verification left as TODO)
ENABLE_APP_ATTEST   = os.getenv("ENABLE_APP_ATTEST", "0") == "1"

# =========================
# FastAPI app
# =========================
app = FastAPI(title="ECG Cloud Inference (4-base + XGB meta)", version="1.0.0")

# =========================
# Models: load on CPU
# =========================
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass

def _load_keras(path: str) -> tf.keras.Model:
    # Load without requiring the custom loss at inference
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        # if your .keras references focal_loss_fixed
        from focal_loss import focal_loss_fixed
        return tf.keras.models.load_model(path, custom_objects={"focal_loss_fixed": focal_loss_fixed}, compile=False)

t0 = time.time()
mi_norm_model        = _load_keras(MI_NORM_PATH)
sttc_norm_model      = _load_keras(STTC_NORM_PATH)
mi_sttc_model        = _load_keras(MI_STTC_PATH)
multiclass_model     = _load_keras(MULTI_CLASS_PATH)

# XGB meta model
if not os.path.exists(META_XGB_PATH):
    raise RuntimeError(f"Meta XGBoost model not found at {META_XGB_PATH}")
xgb_meta = xgb.Booster()
xgb_meta.load_model(META_XGB_PATH)

load_ms = int((time.time() - t0) * 1000)
print(f"[BOOT] Loaded Keras+XGB models in {load_ms} ms")

# =========================
# I/O schemas
# =========================
class ECGRequest(BaseModel):
    ecg: List[float]                      # Lead I samples
    resample_to: Optional[int] = SEQ_LEN  # server resamples if length differs
    mean: Optional[float] = None          # optional z-score params
    std: Optional[float] = None

class ECGResponse(BaseModel):
    label: str
    probs: Dict[str, float]               # final meta probabilities
    base: Dict[str, Dict[str, float]]     # base model probabilities (for sanity/debug)

# =========================
# Helpers
# =========================
def resample_linear(x: np.ndarray, target_len: int) -> np.ndarray:
    if x.shape[0] == target_len:
        return x.astype(np.float32)
    y = np.zeros(target_len, dtype=np.float32)
    scale = (len(x) - 1) / (target_len - 1)
    for i in range(target_len):
        pos = i * scale
        j = int(pos)
        t = pos - j
        y[i] = (1.0 - t) * x[j] + (x[j+1] if j+1 < len(x) else x[j]) * t
    return y

def zscore(x: np.ndarray, mu: Optional[float], sigma: Optional[float]) -> np.ndarray:
    if mu is None:
        mu = float(x.mean())
    if sigma is None:
        sigma = float(x.std() + 1e-6)
    return (x - mu) / max(sigma, 1e-6)

def predict_probs_keras(model: tf.keras.Model, x1d: np.ndarray) -> np.ndarray:
    x = x1d.reshape(1, SEQ_LEN, 1).astype(np.float32)
    with tf.device("/CPU:0"):
        p = model(x, training=False).numpy()[0]
    return p  # 1D array

def map_probs_to_dict(p: np.ndarray, labels: List[str]) -> Dict[str, float]:
    return {labels[i]: float(p[i]) for i in range(len(labels))}

def stack_features(p_mi_norm: np.ndarray,
                   p_sttc_norm: np.ndarray,
                   p_mi_sttc: np.ndarray,
                   p_multi: np.ndarray) -> np.ndarray:
    """
    Match your eval script:
      X_stack = concat([probs_mi_norm (2),
                        probs_sttc_norm (2),
                        probs_mi_sttc (2),
                        probs_multiclass (3)], axis=1)
    -> shape (1, 9)
    """
    return np.concatenate([p_mi_norm, p_sttc_norm, p_mi_sttc, p_multi], axis=0).reshape(1, -1).astype(np.float32)

def meta_predict_xgb(feat9: np.ndarray) -> Dict[str, float]:
    dm = xgb.DMatrix(feat9)
    probs = xgb_meta.predict(dm)[0]  # [num_class]
    # Map **exactly** in the order the meta model was trained with:
    return {META_CLASS_LABELS[i]: float(probs[i]) for i in range(len(META_CLASS_LABELS))}

# =========================
# Auth (JWT + App Attest stub)
# =========================
class JWKSCache:
    def __init__(self, url: str):
        self.url = url
        self._last = 0.0
        self._cache = None

    async def get(self):
        if not self.url:
            return None
        now = time.time()
        if self._cache and (now - self._last) < 3600:
            return self._cache
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(self.url)
            r.raise_for_status()
            self._cache = r.json()
            self._last = now
            return self._cache

jwks_cache = JWKSCache(JWT_JWKS_URL)

async def require_jwt(auth: str = Header(None, alias="Authorization")):
    if not JWT_ISSUER or not JWT_AUDIENCE or not JWT_JWKS_URL:
        # Auth disabled (dev). DO NOT ship like this.
        return
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    jwks = await jwks_cache.get()
    try:
        unverified = jwt.get_unverified_header(token)
        kid = unverified.get("kid")
        key = next((k for k in jwks["keys"] if k["kid"] == kid), None)
        if not key:
            raise Exception("JWKS key not found")
        from jwt.algorithms import RSAAlgorithm
        pubkey = RSAAlgorithm.from_jwk(json.dumps(key))
        payload = jwt.decode(token, pubkey, algorithms=["RS256"], audience=JWT_AUDIENCE, issuer=JWT_ISSUER)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"JWT invalid: {e}")

async def require_app_attest(request: Request):
    if not ENABLE_APP_ATTEST:
        return
    # Expect these from your iOS client; implement full verification for production
    assertion_b64 = request.headers.get("X-App-Attest-Assertion")
    key_id_b64    = request.headers.get("X-App-Attest-KeyId")
    if not assertion_b64 or not key_id_b64:
        raise HTTPException(status_code=401, detail="Missing App Attest headers")
    # TODO: Verify with Apple's App Attest service + your nonce policy.
    return

# =========================
# Routes
# =========================
@app.get("/healthz")
async def healthz():
    return {"ok": True, "models_loaded_ms": load_ms}

@app.post("/predict", response_model=ECGResponse)
async def predict(req: ECGRequest, _: dict = Depends(require_jwt), __: None = Depends(require_app_attest)):
    x = np.asarray(req.ecg, dtype=np.float32)
    if x.ndim != 1:
        raise HTTPException(400, "ecg must be a 1D array")
    target_len = req.resample_to or SEQ_LEN
    if len(x) != target_len:
        x = resample_linear(x, target_len)
    if target_len != SEQ_LEN:
        # ensure model input length
        x = resample_linear(x, SEQ_LEN)

    x = zscore(x, req.mean, req.std)

    # Base model probs
    p_mi_norm    = predict_probs_keras(mi_norm_model,   x)  # len 2
    p_sttc_norm  = predict_probs_keras(sttc_norm_model, x)  # len 2
    p_mi_sttc    = predict_probs_keras(mi_sttc_model,   x)  # len 2
    p_multiclass = predict_probs_keras(multiclass_model,x)  # len 3

    # Stack to 9 features
    feat9 = stack_features(p_mi_norm, p_sttc_norm, p_mi_sttc, p_multiclass)

    # Meta prediction (XGBoost)
    final_probs = meta_predict_xgb(feat9)
    final_label = max(final_probs.items(), key=lambda kv: kv[1])[0]

    # Return both final and base probs (named with correct label orders)
    base = {
        "mi_norm":      map_probs_to_dict(p_mi_norm,    MI_NORM_LABELS),
        "sttc_norm":    map_probs_to_dict(p_sttc_norm,  STTC_NORM_LABELS),
        "mi_sttc":      map_probs_to_dict(p_mi_sttc,    MI_STTC_LABELS),
        "multiclass":   map_probs_to_dict(p_multiclass, MULTI_CLASS_LABELS),
    }
    return ECGResponse(label=final_label, probs=final_probs, base=base)
