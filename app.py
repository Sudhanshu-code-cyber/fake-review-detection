import streamlit as st
from config import APP_NAME
from ml.predictor import predict_review


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed"
)


# =========================================================
# CSS
# =========================================================

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                 Roboto, "Helvetica Neue", Arial, sans-serif;
}
.stApp { background-color: #f7f7f5; }
.block-container { max-width: 780px; padding-top: 2.5rem; padding-bottom: 3rem; }
#MainMenu, footer, header { visibility: hidden; }

/* Header */
.header { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
.header .mark {
    width: 44px; height: 44px; border-radius: 10px;
    background: #111827; color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px;
}
.header h1 { font-size: 1.6rem; font-weight: 700; margin: 0; color: #111827; }
.header .sub { color: #6b7280; font-size: 0.95rem; margin: 0; line-height: 1.4; }
.rule { height: 1px; background: #e5e7eb; margin: 22px 0 26px; border: none; }

/* Textarea */
.stTextArea label {
    font-weight: 600 !important; color: #374151 !important; font-size: 0.9rem !important;
}
.stTextArea textarea {
    background: #fff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 10px !important;
    font-size: 1rem !important;
    padding: 14px !important;
    color: #111827 !important;
}

/* Button */
.stButton > button {
    background: #111827; color: #fff; font-weight: 600;
    font-size: 0.98rem; border: none; border-radius: 10px;
    padding: 0.65rem 1.25rem; width: 100%;
}
.stButton > button:hover { background: #1f2937; }

/* Result card */
.result {
    background: #fff; border: 1px solid #e5e7eb; border-radius: 12px;
    padding: 24px 26px; margin-top: 20px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.result .row { display: flex; align-items: center; justify-content: space-between; }
.result .label {
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #6b7280; margin-bottom: 8px;
}
.result .verdict { font-size: 1.35rem; font-weight: 700; margin: 0; }
.result.real { border-left: 4px solid #059669; }
.result.fake { border-left: 4px solid #dc2626; }
.result.real .verdict { color: #047857; }
.result.fake .verdict { color: #b91c1c; }
.result .confidence {
    font-size: 1.5rem; font-weight: 700; color: #111827;
    margin: 0; text-align: right;
}
.result .confidence small {
    display: block; font-size: 0.72rem; font-weight: 600; color: #9ca3af;
}

/* Error */
.empty-error {
    background: #fff7ed; border: 1px solid #fed7aa; border-left: 4px solid #f97316;
    border-radius: 10px; padding: 14px 16px; margin-top: 15px;
    color: #9a3412; font-weight: 600;
}

/* Info strip */
.info-strip { display: flex; gap: 14px; margin-top: 30px; }
.info-strip .item {
    flex: 1; background: #fff; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 14px 16px;
}
.info-strip .item .k {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 4px;
}
.info-strip .item .v { font-size: 0.95rem; font-weight: 600; color: #111827; }

/* Footer */
.foot {
    margin-top: 40px; padding-top: 18px; border-top: 1px solid #e5e7eb;
    text-align: center; color: #9ca3af; font-size: 0.82rem;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================

st.session_state.setdefault("prediction", None)
st.session_state.setdefault("confidence", None)
st.session_state.setdefault("error", False)


# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="header">
    <div class="mark">🔍</div>
    <div>
        <h1>ReviewGuard</h1>
        <p class="sub">Detect fake reviews with a machine learning model trained on review data.</p>
    </div>
</div>
<hr class="rule">
""", unsafe_allow_html=True)




review = st.text_area(
    "Review text",
    placeholder="Paste a review here — e.g. Best purchase I've made this year, arrived in two days.",
    height=170,
    key="review_text"
)

if st.button("🔎 Analyze Review", use_container_width=True):
    if not review.strip():
        st.session_state.error = True
        st.session_state.prediction = None
        st.session_state.confidence = None
    else:
        st.session_state.error = False
        with st.spinner("Analyzing review..."):
            prediction, confidence = predict_review(review)
        st.session_state.prediction = prediction
        st.session_state.confidence = confidence


# =========================================================
# ERROR
# =========================================================

if st.session_state.error:
    st.markdown("""
    <div class="empty-error">
        ⚠️ Please enter a review before clicking Analyze Review.
    </div>
    """, unsafe_allow_html=True)




if st.session_state.prediction is not None:
    prediction = st.session_state.prediction
    confidence = st.session_state.confidence

    is_fake = prediction.lower() == "fake"
    icon, verdict, css = ("⚠️", "Fake review", "fake") if is_fake else ("✅", "Real review", "real")

    st.markdown(f"""
    <div class="result {css}">
        <div class="row">
            <div>
                <div class="label">Verdict</div>
                <p class="verdict">{icon} {verdict}</p>
            </div>
            <div>
                <div class="label" style="text-align:right;">Confidence</div>
                <p class="confidence">{confidence:.1f}%<small>model score</small></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)




st.markdown("""
<div class="info-strip">
    <div class="item"><div class="k">Model</div><div class="v">🧠 Naive Bayes</div></div>
    <div class="item"><div class="k">Task</div><div class="v">📝 Text Classification</div></div>
    <div class="item"><div class="k">Speed</div><div class="v">⚡ Instant</div></div>
</div>

<div class="foot">MCA · Social Media Analytics · Fake Review Detection</div>
""", unsafe_allow_html=True)