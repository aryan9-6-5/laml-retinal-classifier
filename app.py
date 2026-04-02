import streamlit as st
import numpy as np
import cv2
import json
import os
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
import io

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="LAML Retinal Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Styling ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
    background-color: #05080f;
    color: #d4dbe8;
}
.stApp { background-color: #05080f; }

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #3d4f6b !important;
    border: none !important;
    background: transparent !important;
    padding: 0.6rem 1.2rem !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #4fc3f7 !important;
    border-bottom: 2px solid #4fc3f7 !important;
}
div[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid #111c30 !important;
    gap: 0 !important;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem 0;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: -0.04em;
    background: linear-gradient(120deg, #4fc3f7 0%, #818cf8 55%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
    line-height: 1.1;
}
.hero .sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #3d4f6b;
    letter-spacing: 0.14em;
    text-transform: uppercase;
}

/* ── Cards ── */
.card {
    background: #080e1a;
    border: 1px solid #111c30;
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 0.7rem;
}
.card-glow {
    background: linear-gradient(135deg, #080e1a, #0b1425);
    border: 1px solid #162340;
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 0 30px rgba(79,195,247,0.04);
}

/* ── Stats row ── */
.stat-row { display:flex; gap:0.8rem; margin-bottom:1.5rem; }
.stat-box {
    flex: 1;
    background: #080e1a;
    border: 1px solid #111c30;
    border-radius: 12px;
    padding: 0.9rem 0.7rem;
    text-align: center;
}
.stat-box .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #4fc3f7;
    letter-spacing: -0.02em;
}
.stat-box .lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #3d4f6b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.15rem;
}

/* ── Diagnosis pills ── */
.diag-box {
    background: #060c18;
    border: 1px solid #162340;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-top: 1.2rem;
}
.diag-box .label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4fc3f7;
    margin-bottom: 0.7rem;
}
.dpill {
    display: inline-block;
    background: rgba(79,195,247,0.1);
    border: 1px solid rgba(79,195,247,0.28);
    color: #4fc3f7;
    border-radius: 20px;
    padding: 0.28rem 0.85rem;
    font-size: 0.84rem;
    margin: 0.2rem;
    font-weight: 600;
}
.dpill-normal {
    background: rgba(74,222,128,0.1);
    border-color: rgba(74,222,128,0.28);
    color: #4ade80;
}

/* ── Section label ── */
.slabel {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #3d4f6b;
    margin-bottom: 0.9rem;
}

/* ── Status bars ── */
.status-ok {
    background: #050e09;
    border: 1px solid #14532d;
    border-radius: 10px;
    padding: 0.7rem 1.3rem;
    margin-bottom: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #4ade80;
}
.status-warn {
    background: #120b18;
    border: 1px solid #581c87;
    border-radius: 10px;
    padding: 0.7rem 1.3rem;
    margin-bottom: 1.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.76rem;
    color: #c084fc;
}

/* ── Dataframe overrides ── */
div[data-testid="stDataFrame"] {
    background: #080e1a !important;
    border: 1px solid #111c30 !important;
    border-radius: 10px !important;
}

/* ── Expander ── */
details summary {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #3d4f6b !important;
    letter-spacing: 0.08em !important;
}

/* ── Footer ── */
.footer {
    margin-top: 3.5rem;
    padding-top: 1.2rem;
    border-top: 1px solid #0e1828;
    text-align: center;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #1e2d45;
    letter-spacing: 0.1em;
}
</style>
""", unsafe_allow_html=True)


# ── Constants ─────────────────────────────────────────────────────
DISEASE_FULL = {
    'N': 'Normal', 'D': 'Diabetic Retinopathy', 'G': 'Glaucoma',
    'C': 'Cataract', 'A': 'Age-related MD', 'H': 'Hypertension',
    'M': 'Myopia', 'O': 'Other Findings'
}
DISEASE_NAMES = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

# Canonical output directory — all notebook artifacts land here
import os
OUTPUTS_DIR = 'LAML_outputs'

def out(fname):
    """Return path inside outputs/ dir, also checks current dir as fallback."""
    p1 = os.path.join(OUTPUTS_DIR, fname)
    if os.path.exists(p1):
        return p1
    if os.path.exists(fname):
        return fname
    return None


# ── Model + config loading ────────────────────────────────────────
@st.cache_resource
def load_model_and_config():
    """
    Load LAML_final.keras, config.json, and thresholds.json from outputs/.
    Returns (model, config_dict, thresholds_dict, error_str_or_None).
    """
    try:
        import tensorflow as tf

        model_path = out('LAML_final.keras')
        if not model_path:
            return None, {}, {}, "LAML_final.keras not found"

        model = tf.keras.models.load_model(model_path, compile=False)

        # Config
        cfg = {}
        cfg_path = out('config.json')
        if cfg_path:
            with open(cfg_path) as f:
                cfg = json.load(f)

        # Thresholds — notebook saves these to thresholds.json (separate from config)
        thr_path = out('thresholds.json')
        if thr_path:
            with open(thr_path) as f:
                raw_thr = json.load(f)
            # thresholds.json may use int keys (class indices) or string keys
            # Normalise to {class_letter: float}
            thr = {}
            for k, v in raw_thr.items():
                # if key is an index string like "0","1"...
                try:
                    idx = int(k)
                    thr[DISEASE_NAMES[idx]] = float(v)
                except (ValueError, IndexError):
                    thr[str(k)] = float(v)
        elif 'optimal_thresholds' in cfg:
            thr = cfg['optimal_thresholds']
        else:
            thr = {c: 0.5 for c in DISEASE_NAMES}

        return model, cfg, thr, None

    except Exception as e:
        return None, {}, {}, str(e)


# ── Image helpers ─────────────────────────────────────────────────
def preprocess(image_bytes, size=512):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    yuv  = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(15, 15))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LANCZOS4)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return img


def gradcam(model, img_arr, class_idx):
    try:
        import tensorflow as tf
        last_conv = next(
            (l.name for l in reversed(model.layers)
             if hasattr(l, 'output') and len(l.output.shape) == 4), None
        )
        if not last_conv:
            return None

        # Model outputs a dict with 'disease' and 'lesion' keys
        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output['disease']]
        )
        inp = img_arr[np.newaxis].astype(np.float32)
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(inp)
            score = preds[:, class_idx]
        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.nn.relu(heatmap)
        heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except:
        return None


def overlay_heatmap(img_np, heatmap, alpha=0.42):
    img_u8  = (img_np * 255).astype(np.uint8)
    h       = cv2.resize(heatmap, (img_u8.shape[1], img_u8.shape[0]))
    colored = np.uint8(255 * mpl_cm.jet(h)[:, :, :3])
    blended = np.clip(alpha * colored + (1 - alpha) * img_u8, 0, 255).astype(np.uint8)
    return blended


def load_png(fname):
    """Return PIL image or None."""
    p = out(fname)
    if p:
        return Image.open(p)
    return None


# ── Load model once ───────────────────────────────────────────────
model, config, thr, load_err = load_model_and_config()

# ── Hero ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>LAML Retinal Classifier</h1>
    <div class="sub">Lesion-Aware Multi-Task Learning &nbsp;·&nbsp; ConvNeXtTiny &nbsp;·&nbsp; ODIR-5K</div>
</div>
""", unsafe_allow_html=True)

# Model status
if model is None:
    st.markdown(f"""
    <div class="status-warn">
        ⚠ Model not loaded — {load_err or 'place outputs/ folder next to app.py and restart.'}
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="status-ok">
        ✓ LAML_final.keras loaded &nbsp;·&nbsp; thresholds.json active &nbsp;·&nbsp; 512×512 px input &nbsp;·&nbsp; 8-class sigmoid heads
    </div>
    """, unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────
tab_infer, tab_perf, tab_data, tab_about = st.tabs([
    "  Inference  ",
    "  Performance  ",
    "  Dataset  ",
    "  About  ",
])


# ═══════════════════════════════════════════════════════════════════
# TAB 1 — INFERENCE
# ═══════════════════════════════════════════════════════════════════
with tab_infer:
    col_up, col_info = st.columns([1.2, 1])

    with col_up:
        st.markdown('<div class="slabel">Fundus Image Upload</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Upload fundus image",
            type=['jpg', 'jpeg', 'png'],
            label_visibility="collapsed"
        )

    with col_info:
        st.markdown('<div class="slabel">Model Summary</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="stat-row">
            <div class="stat-box"><div class="val">0.8490</div><div class="lbl">Macro AUC</div></div>
            <div class="stat-box"><div class="val">0.8752</div><div class="lbl">Rare AUC</div></div>
            <div class="stat-box"><div class="val">512px</div><div class="lbl">Input</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="card" style="font-family:'IBM Plex Mono',monospace;font-size:0.73rem;
                                  color:#3d4f6b;line-height:2;">
            Backbone&nbsp;&nbsp;&nbsp;→ ConvNeXtTiny (ImageNet)<br>
            Attention&nbsp;&nbsp;→ Squeeze-and-Excitation<br>
            Loss&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ Focal BCE + λ·Lesion BCE<br>
            λ&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ 0.3<br>
            Training&nbsp;&nbsp;&nbsp;→ ODIR + EyePACS + ORIGA + AREDS2 + RFMiD<br>
            Thresholds&nbsp;→ Per-class F1-optimal
        </div>
        """, unsafe_allow_html=True)

    # ── Run prediction ────────────────────────────────────────────
    if uploaded and model:
        img_bytes = uploaded.read()
        img_np    = preprocess(img_bytes)

        if img_np is None:
            st.error("Could not decode image — try a different file.")
        else:
            import tensorflow as tf

            inp    = img_np[np.newaxis].astype(np.float32)
            out_d  = model.predict(inp, verbose=0)

            # Handle both dict output and plain tensor output
            if isinstance(out_d, dict):
                probs = out_d['disease'][0]
            elif isinstance(out_d, (list, tuple)):
                probs = out_d[0][0]
            else:
                probs = out_d[0]

            thr_a  = np.array([thr.get(c, 0.5) for c in DISEASE_NAMES])
            binary = (probs >= thr_a).astype(int)
            found  = [DISEASE_FULL[c] for c, b in zip(DISEASE_NAMES, binary) if b]

            st.markdown("---")
            col_img, col_res = st.columns([1, 1.4])

            with col_img:
                st.markdown('<div class="slabel">Input Image</div>', unsafe_allow_html=True)
                st.image(img_np, use_container_width=True, clamp=True)

            with col_res:
                st.markdown('<div class="slabel">Disease Probabilities</div>',
                            unsafe_allow_html=True)
                for i in np.argsort(probs)[::-1]:
                    cls  = DISEASE_NAMES[i]
                    name = DISEASE_FULL[cls]
                    prob = probs[i]
                    pred = binary[i]
                    bar_pct = int(prob * 100)
                    bar_col = "#ffffff" if pred else '#0e1828'
                    tick    = '✓' if pred else ''
                    active_col = '#d4dbe8' if pred else '#2a3a55'
                    prob_col   = "#ffffff" if pred else '#2a3a55'
                    fw = '600' if pred else '400'
                    st.markdown(f"""
                    <div style="margin-bottom:0.5rem;">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:0.22rem;">
                            <span style="font-size:0.84rem;font-weight:{fw};color:{active_col};">
                                {name}
                            </span>
                            <span style="font-family:'IBM Plex Mono',monospace;
                                         font-size:0.76rem;color:{prob_col};">
                                {tick}&nbsp;{prob:.3f}
                            </span>
                        </div>
                        <div style="background:#0c1624;border-radius:3px;height:4px;">
                            <div style="width:{bar_pct}%;background:{bar_col};
                                        border-radius:3px;height:4px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Diagnosis box
            if found and found != ['Normal']:
                pills = ''.join(f'<span class="dpill">{d}</span>' for d in found)
                st.markdown(f"""
                <div class="diag-box">
                    <div class="label">Predicted Diagnosis</div>
                    {pills}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="diag-box">
                    <div class="label">Predicted Diagnosis</div>
                    <span class="dpill dpill-normal">✓ Normal</span>
                </div>
                """, unsafe_allow_html=True)

            # Grad-CAM
            st.markdown("---")
            st.markdown('<div class="slabel">Grad-CAM Attention Maps</div>',
                        unsafe_allow_html=True)

            pos_classes = [(i, probs[i]) for i in range(len(DISEASE_NAMES)) if binary[i]]
            pos_classes.sort(key=lambda x: x[1], reverse=True)
            show_classes = pos_classes[:2] if pos_classes else [(int(np.argmax(probs)), probs.max())]

            gcam_cols = st.columns(len(show_classes) + 1)
            with gcam_cols[0]:
                st.markdown('<p style="font-family:\'IBM Plex Mono\',monospace;'
                            'font-size:0.68rem;color:#3d4f6b;text-align:center;'
                            'text-transform:uppercase;letter-spacing:0.1em;">Original</p>',
                            unsafe_allow_html=True)
                st.image(img_np, use_container_width=True, clamp=True)

            for ci, (cls_i, prob_v) in enumerate(show_classes):
                hm = gradcam(model, img_np, cls_i)
                with gcam_cols[ci + 1]:
                    cls_name = DISEASE_FULL[DISEASE_NAMES[cls_i]]
                    st.markdown(
                        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
                        f'color:#4fc3f7;text-align:center;text-transform:uppercase;'
                        f'letter-spacing:0.08em;">{cls_name[:18]}<br>{prob_v:.3f}</p>',
                        unsafe_allow_html=True
                    )
                    if hm is not None:
                        st.image(overlay_heatmap(img_np, hm), use_container_width=True, clamp=True)
                    else:
                        st.caption("Grad-CAM unavailable")

            # Raw table
            with st.expander("Raw scores & thresholds"):
                import pandas as pd
                df = pd.DataFrame({
                    'Class':     DISEASE_NAMES,
                    'Disease':   [DISEASE_FULL[c] for c in DISEASE_NAMES],
                    'Prob':      [round(float(p), 4) for p in probs],
                    'Threshold': [round(float(thr.get(c, 0.5)), 4) for c in DISEASE_NAMES],
                    'Margin':    [round(float(probs[i] - thr.get(DISEASE_NAMES[i], 0.5)), 4)
                                  for i in range(len(DISEASE_NAMES))],
                    'Predicted': ['✓' if b else '' for b in binary],
                })
                st.dataframe(df, use_container_width=True, hide_index=True)

    elif uploaded and model is None:
        st.warning("Model not loaded — see status bar above.")

    elif not uploaded:
        st.markdown("""
        <div style="text-align:center;padding:5rem 0 3rem 0;color:#111c30;">
            <div style="font-size:2.8rem;margin-bottom:1.2rem;opacity:0.6;">👁</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.76rem;
                        letter-spacing:0.14em;text-transform:uppercase;color:#1e2d45;">
                Upload a fundus photograph to begin
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
with tab_perf:
    st.markdown('<div class="slabel">Training & Evaluation Results</div>',
                unsafe_allow_html=True)

    # ── AUC scores ────────────────────────────────────────────────
    auc_path = out('auc_scores_balanced.csv')
    if auc_path:
        import pandas as pd
        auc_df = pd.read_csv(auc_path)
        st.markdown("**Per-Class AUC (Balanced Test Set)**")
        # Style the dataframe
        st.dataframe(
            auc_df.style.format(precision=4)
                        .background_gradient(cmap='Blues', subset=auc_df.select_dtypes('number').columns),
            use_container_width=True, hide_index=True
        )
    else:
        st.caption("auc_scores_balanced.csv not found in outputs/")

    # ── Ablation results ──────────────────────────────────────────
    abl_path = out('ablation_results.csv')
    if abl_path:
        import pandas as pd
        st.markdown("**Ablation Study**")
        abl_df = pd.read_csv(abl_path)
        st.dataframe(
            abl_df.style.format(precision=4),
            use_container_width=True, hide_index=True
        )
    else:
        st.caption("ablation_results.csv not found in outputs/")

    # ── Training curves ───────────────────────────────────────────
    img_train = load_png('training_history.png')
    if img_train:
        st.markdown("**Training History (Phase 1 + Phase 2)**")
        st.image(img_train, use_container_width=True)
    else:
        # Try to render from CSVs directly
        log_p1 = out('log_p1.csv')
        log_p2 = out('log_p2.csv')
        if log_p1 or log_p2:
            import pandas as pd
            dfs = []
            if log_p1:
                d = pd.read_csv(log_p1); d['phase'] = 'Phase 1'; dfs.append(d)
            if log_p2:
                d = pd.read_csv(log_p2); d['phase'] = 'Phase 2'; dfs.append(d)
            logs = pd.concat(dfs, ignore_index=True)

            fig, axes = plt.subplots(1, 2, figsize=(12, 4),
                                     facecolor='#05080f')
            for ax in axes:
                ax.set_facecolor('#080e1a')
                for spine in ax.spines.values():
                    spine.set_color('#111c30')
                ax.tick_params(colors='#3d4f6b', labelsize=8)

            # Loss
            if 'loss' in logs.columns:
                axes[0].plot(logs['loss'].values, color='#4fc3f7', linewidth=1.5, label='Train loss')
            if 'val_loss' in logs.columns:
                axes[0].plot(logs['val_loss'].values, color='#c084fc', linewidth=1.5, label='Val loss')
            axes[0].set_title('Loss', color='#d4dbe8', fontsize=10)
            axes[0].legend(fontsize=8, labelcolor='#d4dbe8', facecolor='#080e1a')

            # AUC col detection
            auc_col = next((c for c in logs.columns if 'auc' in c.lower() and 'val' not in c.lower()), None)
            val_auc_col = next((c for c in logs.columns if 'auc' in c.lower() and 'val' in c.lower()), None)
            if auc_col:
                axes[1].plot(logs[auc_col].values, color='#4fc3f7', linewidth=1.5, label='Train AUC')
            if val_auc_col:
                axes[1].plot(logs[val_auc_col].values, color='#c084fc', linewidth=1.5, label='Val AUC')
            axes[1].set_title('AUC', color='#d4dbe8', fontsize=10)
            axes[1].legend(fontsize=8, labelcolor='#d4dbe8', facecolor='#080e1a')

            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#05080f', dpi=120)
            buf.seek(0)
            st.image(buf.read(), use_container_width=True)
            plt.close(fig)

    # ── ROC curves ────────────────────────────────────────────────
    img_roc = load_png('roc_curves_balanced.png')
    if img_roc:
        st.markdown("**ROC Curves (Balanced Test Set)**")
        st.image(img_roc, use_container_width=True)

    # ── AUC bar chart ─────────────────────────────────────────────
    img_bar = load_png('auc_bar_chart_balanced.png')
    if img_bar:
        st.markdown("**AUC Bar Chart**")
        st.image(img_bar, use_container_width=True)

    # ── Confusion matrices ────────────────────────────────────────
    img_cm = load_png('confusion_matrices_balanced.png')
    if img_cm:
        st.markdown("**Confusion Matrices (Balanced Test Set)**")
        st.image(img_cm, use_container_width=True)

    # ── Thresholds ────────────────────────────────────────────────
    if thr:
        import pandas as pd
        st.markdown("**Per-Class Decision Thresholds (F1-optimal)**")
        thr_df = pd.DataFrame([
            {'Class': c, 'Disease': DISEASE_FULL.get(c, c), 'Threshold': round(float(v), 4)}
            for c, v in thr.items()
        ])
        st.dataframe(thr_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DATASET
# ═══════════════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="slabel">Dataset Manifest & Splits</div>', unsafe_allow_html=True)

    manifest_path = out('dataset_manifest.csv')
    if manifest_path:
        import pandas as pd
        manifest = pd.read_csv(manifest_path)
        total = len(manifest)

        # Summary stats
        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-box">
                <div class="val">{total:,}</div>
                <div class="lbl">Total Samples</div>
            </div>
            <div class="stat-box">
                <div class="val">{manifest['source_tag'].nunique() if 'source_tag' in manifest.columns else '—'}</div>
                <div class="lbl">Sources</div>
            </div>
            <div class="stat-box">
                <div class="val">8</div>
                <div class="lbl">Disease Classes</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Source breakdown
        if 'source_tag' in manifest.columns:
            src_counts = manifest['source_tag'].value_counts().reset_index()
            src_counts.columns = ['Source', 'Count']
            src_counts['Pct'] = (src_counts['Count'] / total * 100).round(1)
            st.markdown("**Source Breakdown**")
            st.dataframe(src_counts, use_container_width=True, hide_index=True)

        # Class distribution — detect disease columns (single-char uppercase)
        disease_cols = [c for c in manifest.columns if c in list('NDGCAHMO')]
        if disease_cols:
            st.markdown("**Class Distribution**")
            class_dist = pd.DataFrame({
                'Class':   disease_cols,
                'Disease': [DISEASE_FULL.get(c, c) for c in disease_cols],
                'Count':   [int(manifest[c].sum()) for c in disease_cols],
                'Prevalence %': [(manifest[c].sum() / total * 100).round(2) for c in disease_cols],
            }).sort_values('Count', ascending=False)
            st.dataframe(class_dist, use_container_width=True, hide_index=True)

        # Split sizes
        train_i = out('train_indices.npy')
        val_i   = out('val_indices.npy')
        test_i  = out('test_indices.npy')
        if train_i and val_i and test_i:
            n_tr = len(np.load(train_i))
            n_v  = len(np.load(val_i))
            n_te = len(np.load(test_i))
            st.markdown("**Patient-Level Splits (GroupShuffleSplit)**")
            split_df = pd.DataFrame({
                'Split':   ['Train', 'Validation', 'Test'],
                'Samples': [n_tr, n_v, n_te],
                'Pct':     [f"{n_tr/total*100:.1f}%", f"{n_v/total*100:.1f}%", f"{n_te/total*100:.1f}%"],
            })
            st.dataframe(split_df, use_container_width=True, hide_index=True)

        with st.expander("Browse manifest (first 200 rows)"):
            st.dataframe(manifest.head(200), use_container_width=True)
    else:
        st.info("dataset_manifest.csv not found in outputs/. Run the notebook first.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ═══════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown('<div class="slabel">Architecture & Methods</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card-glow" style="line-height:1.9;font-size:0.9rem;color:#9aabc4;">
        <b style="color:#d4dbe8;">LAML</b> (Lesion-Aware Multi-task Learning) is a multi-label retinal
        disease classification system trained on six datasets spanning over 80,000 fundus photographs.
        <br><br>
        <b style="color:#d4dbe8;">Architecture</b><br>
        ConvNeXtTiny backbone (ImageNet pre-trained) with a Squeeze-and-Excitation channel attention
        block. Shared dense layers branch into two sigmoid heads: an 8-class disease head and a
        lesion-concept head. The lesion head provides auxiliary supervision, guiding the model to
        attend to clinically meaningful image regions.
        <br><br>
        <b style="color:#d4dbe8;">Training</b><br>
        Phase 1 (15 epochs): backbone frozen, heads trained at LR=1e-3.<br>
        Phase 2 (fine-tune): top 40% of backbone unfrozen, LR≈9e-7.
        <br><br>
        <b style="color:#d4dbe8;">Loss</b><br>
        Focal BCE on the disease head (γ=2, class-weighted) + λ·BCE on the lesion head, λ=0.3.
        Sample weights derived from per-class positive frequency to counter severe class imbalance.
        <br><br>
        <b style="color:#d4dbe8;">Datasets</b><br>
        ODIR-5K · EyePACS (DR) · ORIGA (Glaucoma) · AREDS2 (AMD) · RFMiD · Hypertension fundus
        <br><br>
        <b style="color:#d4dbe8;">Evaluation</b><br>
        Balanced test set (equal class sampling), per-class F1-optimal thresholds, macro AUC,
        confusion matrices, and ROC curves.  External validation on held-out RFMiD split.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="slabel" style="margin-top:1.5rem;">Key Outputs</div>',
                unsafe_allow_html=True)
    artifacts = [
        ('LAML_final.keras',              'Trained Keras model (full SavedModel format)'),
        ('ckpt_p1.weights.h5',            'Phase 1 best checkpoint'),
        ('ckpt_p2.weights.h5',            'Phase 2 best checkpoint'),
        ('thresholds.json',               'Per-class F1-optimal decision thresholds'),
        ('auc_scores_balanced.csv',       'Per-class AUC on balanced test set'),
        ('confusion_matrices_balanced.png','Per-class confusion matrix grid'),
        ('roc_curves_balanced.png',       'Per-class ROC curve plot'),
        ('dataset_manifest.csv',          'Full merged dataset with source tags'),
        ('ablation_results.csv',          'Ablation study results'),
    ]
    import pandas as pd
    adf = pd.DataFrame(artifacts, columns=['File', 'Description'])
    adf['Present'] = adf['File'].apply(lambda f: '✓' if out(f) else '✗')
    st.dataframe(adf, use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    LAML · Lesion-Aware Multi-Task Learning · ConvNeXtTiny · ODIR-5K · 2024
</div>
""", unsafe_allow_html=True)