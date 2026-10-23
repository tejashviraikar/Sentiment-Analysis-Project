"""Streamlit demo:  python -m streamlit run src/app.py"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import clean_text

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Review Sentiment Analyzer", page_icon="🎬", layout="centered")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root { --pos:#1F9D74; --neg:#D64550; --brand:#4457D8; }
html, body, [class*="css"], .stApp { font-family:'DM Sans', system-ui, sans-serif; }
.block-container { max-width: 760px; padding-top: 2.2rem; }
header[data-testid="stHeader"] { background: transparent; }

.hero { padding: 1.6rem 1.6rem 1.4rem; border-radius: 18px; color:#fff; margin-bottom: 1.4rem;
        background: linear-gradient(135deg, #25306F 0%, #4457D8 60%, #6D7CF0 100%); }
.hero h1 { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:2.2rem;
           line-height:1.15; margin:0 0 .35rem; color:#fff; padding:0; }
.hero p { margin:0; opacity:.88; font-size:1rem; }

div[data-testid="stTextArea"] textarea { border-radius: 14px; font-size: 1.02rem; line-height: 1.5; }
div.stButton > button { border-radius: 999px; font-weight: 600; padding: .45rem 1.1rem; }
div.stButton > button[kind="primary"] { background: var(--brand); border-color: var(--brand); color:#fff; }

.ticket { display:flex; border-radius:16px; overflow:hidden; margin: 1.2rem 0 .9rem;
          border:1.5px solid var(--accent); background: color-mix(in srgb, var(--accent) 10%, transparent); }
.ticket.pos { --accent: var(--pos); }
.ticket.neg { --accent: var(--neg); }
.ticket-main { flex:1; display:flex; align-items:center; gap:1rem; padding:1.1rem 1.3rem; }
.ticket-emoji { font-size:2.6rem; line-height:1; }
.verdict { font-family:'Fraunces', Georgia, serif; font-size:1.9rem; font-weight:700;
           color: var(--accent); line-height:1.1; }
.sub { opacity:.8; font-size:.95rem; margin-top:.2rem; }
.ticket-stub { width:130px; display:flex; flex-direction:column; align-items:center; justify-content:center;
               border-left:2px dashed var(--accent); padding:.8rem; text-align:center; }
.pct { font-family:'Fraunces', Georgia, serif; font-size:2rem; font-weight:700; color: var(--accent); }
.pct-label { font-size:.8rem; opacity:.75; }

.meter { position:relative; height:12px; border-radius:999px; margin-top:.4rem;
         background: linear-gradient(90deg, var(--neg), #E8C547 50%, var(--pos)); }
.marker { position:absolute; top:-5px; width:6px; height:22px; border-radius:3px; background:#fff;
          border:2px solid #25306F; transform: translateX(-50%); }
.scale { display:flex; justify-content:space-between; font-size:.82rem; opacity:.7; margin-top:.35rem; }
.note { margin-top:.8rem; font-size:.92rem; opacity:.85; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

st.markdown(
    '<div class="hero"><h1>Is this review a thumbs up or a thumbs down?</h1>'
    "<p>Paste a movie review and the model tells you how it reads.</p></div>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "best_model.joblib")


try:
    model = load_model()
except FileNotFoundError:
    st.error("No trained model found. Run `python src/train.py` first, then reload this page.")
    st.stop()

EXAMPLES = {
    "Loved it": "An absolute masterpiece. The acting was brilliant and the ending was perfect.",
    "Hated it": "Terrible movie. The plot made no sense and I regret wasting two hours on this.",
    "Not bad": "This movie was not bad at all. I actually enjoyed it a lot.",
    "Mixed": "The visuals are stunning and the cast is great, but the script is lazy and the ending ruins everything.",
}


def use_example(text):
    st.session_state["review"] = text


tab_try, tab_results = st.tabs(["Try a review", "How the models compare"])

with tab_try:
    st.write("Start with an example, or write your own:")
    cols = st.columns(len(EXAMPLES))
    for col, (label, text) in zip(cols, EXAMPLES.items()):
        col.button(label, on_click=use_example, args=(text,), key=f"ex_{label}")

    text = st.text_area("Your review", key="review", height=140,
                        placeholder="e.g. The story kept me hooked until the very last scene...",
                        label_visibility="collapsed")
    analyze = st.button("Analyze review", type="primary")

    if analyze:
        if not text.strip():
            st.info("Write a review or pick an example above, then press Analyze.")
        else:
            x = [clean_text(text)]
            if hasattr(model, "predict_proba"):
                p_pos = float(model.predict_proba(x)[0][1])
            else:  # LinearSVC has no probabilities: squash the margin
                p_pos = float(1 / (1 + np.exp(-model.decision_function(x)[0])))
            positive = p_pos >= 0.5
            conf = max(p_pos, 1 - p_pos)
            word = "Positive" if positive else "Negative"
            close = conf < 0.65
            note = ("This one is a close call. The review may be mixed or use wording the model finds ambiguous."
                    if close else "The wording of this review leans clearly one way.")
            st.markdown(
                f"""
                <div class="ticket {'pos' if positive else 'neg'}">
                  <div class="ticket-main">
                    <div class="ticket-emoji">{'😊' if positive else '😞'}</div>
                    <div><div class="verdict">{word}</div>
                    <div class="sub">The model is {conf:.0%} sure this review is {word.lower()}.</div></div>
                  </div>
                  <div class="ticket-stub"><div class="pct">{conf:.0%}</div>
                  <div class="pct-label">confidence</div></div>
                </div>
                <div class="meter"><div class="marker" style="left:{p_pos*100:.1f}%"></div></div>
                <div class="scale"><span>Negative</span><span>Positive</span></div>
                <div class="note">{note}</div>
                """,
                unsafe_allow_html=True,
            )

    with st.expander("How does it work?"):
        st.write(
            "Your review is cleaned (HTML removed, stop words dropped, negations like “not” kept), "
            "turned into TF-IDF word weights, and scored by the best model from our comparison. "
            "It counts words and word pairs, so sarcasm can fool it."
        )

with tab_results:
    comp = RESULTS / "model_comparison.csv"
    if not comp.exists():
        st.info("Run `python src/train.py` to generate the comparison results.")
    else:
        df = pd.read_csv(comp)
        best = df.iloc[0]
        c1, c2, c3 = st.columns(3)
        c1.metric("Best model", best["Model"])
        c2.metric("Accuracy", f"{best['Accuracy']:.1%}")
        c3.metric("F1 score", f"{best['F1']:.3f}")
        st.dataframe(df, hide_index=True)
        for img, cap in [("model_comparison.png", "F1 score by model"),
                         ("confusion_matrices.png", "Confusion matrices: where each model goes wrong")]:
            if (RESULTS / img).exists():
                st.image(str(RESULTS / img), caption=cap)
        words = RESULTS / "top_words.csv"
        if words.exists():
            st.subheader("Words that push the score most")
            st.dataframe(pd.read_csv(words), hide_index=True)
