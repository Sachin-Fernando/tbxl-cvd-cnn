# build/export_single_to_coreml.py
import argparse, os, shutil, tempfile
import numpy as np
import tensorflow as tf
import coremltools as ct

# ---------- Force CPU kernels so LSTM doesn't become CuDNN ----------
try:
    tf.config.set_visible_devices([], "GPU")
    print("✅ Disabled GPU visibility for conversion (forcing CPU kernels).")
except Exception as e:
    print("Note: could not change visible devices:", e)

ENSEMBLE_DIR = os.path.join("models", "ensemble")
MODEL_FILES = {
    "mi_norm":   os.path.join(ENSEMBLE_DIR, "model_mi_norm.keras"),
    "sttc_norm": os.path.join(ENSEMBLE_DIR, "model_sttc_norm.keras"),
    "mi_sttc":   os.path.join(ENSEMBLE_DIR, "model_mi_sttc.keras"),
}
# Match LabelEncoder order you used during training
LABEL_ORDERS = {
    "mi_norm":   ["MI", "NORM"],
    "sttc_norm": ["NORM", "STTC"],
    "mi_sttc":   ["MI", "STTC"],
}

SEQ_LEN = 1000  # your Input(shape=(1000,1))

def load_keras_model(path: str) -> tf.keras.Model:
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception:
        # if focal_loss was baked in the file, try with it present
        from focal_loss import focal_loss_fixed
        return tf.keras.models.load_model(path, custom_objects={"focal_loss_fixed": focal_loss_fixed}, compile=False)

class Wrapper(tf.Module):
    """Wrap Keras model in a single-signature SavedModel (one tensor in → one tensor out)."""
    def __init__(self, model: tf.keras.Model):
        super().__init__()
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, SEQ_LEN, 1), dtype=tf.float32, name="ecg")])
    def serve(self, ecg):
        with tf.device("/CPU:0"):
            y = self.model(ecg, training=False)  # (1, 2)
        return y  # single tensor (not dict)

def _convert(savedmodel_dir: str, labels: list, prefer_mlprogram: bool):
    """Try mlprogram first; if it fails, fall back to neuralnetwork."""
    # 1) Try ML Program
    if prefer_mlprogram:
        try:
            return ct.convert(
                savedmodel_dir,
                source="tensorflow",
                convert_to="mlprogram",
                inputs=[ct.TensorType(name="ecg", shape=(1, SEQ_LEN, 1), dtype=np.float32)],
                classifier_config=ct.ClassifierConfig(class_labels=labels),
                minimum_deployment_target=ct.target.iOS16,
            ), "mlprogram"
        except Exception as e:
            print(f"   ML Program convert failed: {e}\n   ↪︎ Falling back to NeuralNetwork backend…")

    # 2) Fallback: NeuralNetwork
    mlmodel = ct.convert(
        savedmodel_dir,
        source="tensorflow",
        convert_to="neuralnetwork",
        inputs=[ct.TensorType(name="ecg", shape=(1, SEQ_LEN, 1), dtype=np.float32)],
        classifier_config=ct.ClassifierConfig(class_labels=labels),
        minimum_deployment_target=ct.target.iOS16,
    )
    return mlmodel, "neuralnetwork"

def export_one(tag: str, out_dir: str):
    path = MODEL_FILES[tag]
    labels = LABEL_ORDERS[tag]
    os.makedirs(out_dir, exist_ok=True)

    print(f"🔹 Loading Keras model: {path}")
    model = load_keras_model(path)
    units = int(model.outputs[0].shape[-1])
    if units != 2:
        raise ValueError(f"{tag}: expected 2 output units, got {units}")

    # ---- Create SavedModel with exactly ONE signature ----
    tmp = tempfile.mkdtemp(prefix=f"sm_{tag}_")
    try:
        wrapper = Wrapper(model)
        _ = wrapper.serve.get_concrete_function(tf.TensorSpec((1, SEQ_LEN, 1), tf.float32, name="ecg"))
        tf.saved_model.save(wrapper, tmp, signatures={"serving_default": wrapper.serve})

        # ---- Convert (mlprogram -> nn fallback) ----
        print("🔹 Converting SavedModel → Core ML")
        mlmodel, backend = _convert(tmp, labels, prefer_mlprogram=True)

        # Metadata
        mlmodel.author = "Sachin Fernando"
        mlmodel.short_description = f"Binary ECG classifier: {tag} (labels: {labels})"
        mlmodel.user_defined_metadata.update({
            "input_name": "ecg",
            "input_shape": str((1, SEQ_LEN, 1)),
            "labels_index_order": ",".join(labels),
            "source_model": os.path.basename(path),
            "backend": backend,
            "note": "Expect Lead I normalized & resampled to (1000,1).",
        })

        # Save with correct extension per backend
        ext = ".mlpackage" if backend == "mlprogram" else ".mlmodel"
        out_path = os.path.join(out_dir, f"ECG_{tag}{ext}")
        mlmodel.save(out_path)
        print(f"✅ Saved: {out_path}  ({backend})")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["mi_norm","sttc_norm","mi_sttc","all"], required=True)
    ap.add_argument("--out_dir", default="coreml_exports")
    args = ap.parse_args()

    tags = ["mi_norm","sttc_norm","mi_sttc"] if args.which == "all" else [args.which]
    for t in tags:
        try:
            export_one(t, args.out_dir)
        except Exception as e:
            print(f"⚠️ Failed exporting {t}: {e}")

if __name__ == "__main__":
    main()
