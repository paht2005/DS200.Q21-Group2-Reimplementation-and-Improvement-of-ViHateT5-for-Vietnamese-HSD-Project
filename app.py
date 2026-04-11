"""
Streamlit demo app for ViHateT5 — Vietnamese Hate Speech Detection.

Run with:
    streamlit run app.py
"""

import streamlit as st
import torch
import unicodedata
import pandas as pd
import numpy as np
import time

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ViHateT5 Demo — Vietnamese Hate Speech Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
T5_MODELS = {
    "ViHateT5 Fine-tune Balanced (Best)": "NCPhat2005/vit5_finetune_balanced",
    "ViHateT5 Fine-tune Hate-only": "NCPhat2005/vit5_finetune_hate_only",
    "ViHateT5 Fine-tune Multi": "NCPhat2005/vit5_finetune_multi",
    "ViHateT5 Reimplementation": "NCPhat2005/vihatet5_reimpl",
}

TASK_PREFIXES = {
    "ViHSD — Hate Speech Detection": "hate-speech-detection",
    "ViCTSD — Toxic Speech Detection": "toxic-speech-detection",
    "ViHOS — Hate Spans Detection": "hate-spans-detection",
}

LABEL_COLORS = {
    "CLEAN": "#28a745",
    "NONE": "#28a745",
    "OFFENSIVE": "#ffc107",
    "TOXIC": "#dc3545",
    "HATE": "#dc3545",
}

# Paper Table 3 benchmark results (Macro F1)
BENCHMARK_DATA = {
    "Model": [
        "BERT (multilingual, cased)",
        "BERT (multilingual, uncased)",
        "DistilBERT (multilingual)",
        "XLM-RoBERTa",
        "PhoBERT",
        "PhoBERT_v2",
        "viBERT",
        "ViSoBERT",
        "ViHateT5 (Paper)",
    ],
    "ViHSD (MF1)": [0.6444, 0.6292, 0.6334, 0.6508, 0.6476, 0.6660, 0.6285, 0.6771, 0.6867],
    "ViCTSD (MF1)": [0.6710, 0.6796, 0.6850, 0.7153, 0.7131, 0.7139, 0.6765, 0.7145, 0.7163],
    "ViHOS (MF1)": [0.7637, 0.7393, 0.7615, 0.8133, 0.7281, 0.7351, 0.7291, 0.8604, 0.8637],
    "Average MF1": [0.6930, 0.6827, 0.6933, 0.7265, 0.6963, 0.7050, 0.6780, 0.7507, 0.7556],
}

# T5 model comparison (Paper Table 4)
T5_COMPARISON_DATA = {
    "Model": ["mT5-base", "ViT5-base", "ViHateT5-base (Paper)"],
    "ViHSD (MF1)": [0.6676, 0.6695, 0.6867],
    "ViCTSD (MF1)": [0.6993, 0.6482, 0.7163],
    "ViHOS (MF1)": [0.8660, 0.8690, 0.8637],
    "Average MF1": [0.7289, 0.7443, 0.7556],
}


# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_t5_model(model_name: str):
    """Load T5 model and tokenizer from HuggingFace, cached across reruns."""
    from transformers import T5ForConditionalGeneration, PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_t5_inference(text: str, task_prefix: str, model, tokenizer, device) -> str:
    """Run a single T5 inference and return the decoded output."""
    input_text = f"{task_prefix}: {text}"
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=256, num_beams=1, do_sample=False)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def extract_hate_spans(original_text: str, tagged_text: str):
    """Extract hate spans from T5 output containing [HATE]...[HATE] tags.

    Returns a list of (start, end) character ranges in the *original* text.
    """
    tag = "[hate]"
    tagged_lower = unicodedata.normalize("NFC", tagged_text.lower())
    original_lower = unicodedata.normalize("NFC", original_text.lower())

    substrings = []
    pos = tagged_lower.find(tag)
    while pos != -1:
        end = tagged_lower.find(tag, pos + len(tag))
        if end == -1:
            break
        substrings.append(tagged_lower[pos + len(tag) : end])
        pos = tagged_lower.find(tag, end + len(tag))

    if not substrings:
        return []

    spans = []
    for sub in substrings:
        idx = original_lower.find(sub)
        while idx != -1:
            spans.append((idx, idx + len(sub)))
            idx = original_lower.find(sub, idx + 1)
    return sorted(set(spans))


def highlight_hate_spans_html(text: str, spans) -> str:
    """Return HTML with hate spans highlighted in red."""
    if not spans:
        return text
    # Merge overlapping spans
    merged = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    parts = []
    prev = 0
    for s, e in merged:
        parts.append(text[prev:s])
        parts.append(
            f'<span style="background-color:#ff4b4b;color:white;padding:2px 4px;border-radius:4px;font-weight:600;">{text[s:e]}</span>'
        )
        prev = e
    parts.append(text[prev:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image("https://i.imgur.com/WmMnSRt.png", width=220)
    st.title("ViHateT5 Demo")
    st.caption("DS200.Q21 — Big Data Analysis — Group 02")

    selected_model_label = st.selectbox("🤖 Select model", list(T5_MODELS.keys()))
    model_id = T5_MODELS[selected_model_label]
    st.code(model_id, language=None)

    st.divider()
    st.markdown(
        "**Tasks supported:**\n"
        "1. ViHSD — Hate Speech Detection\n"
        "2. ViCTSD — Toxic Speech Detection\n"
        "3. ViHOS — Hate Spans Detection"
    )
    st.divider()
    st.markdown(
        "**Paper:** [ViHateT5 (ACL 2024)](https://aclanthology.org/2024.findings-acl.355.pdf)"
    )
    device_info = "GPU 🟢" if torch.cuda.is_available() else "CPU 🔵"
    st.markdown(f"**Device:** {device_info}")

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
with st.spinner(f"Loading model **{selected_model_label}** …"):
    model, tokenizer, device = load_t5_model(model_id)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_inference, tab_batch, tab_compare, tab_analysis, tab_about = st.tabs(
    ["🔍 Inference", "📂 Batch Inference", "📊 Model Comparison", "🔬 Analysis", "ℹ️ About"]
)

# ======================== TAB 1: Single Inference ========================
with tab_inference:
    st.header("Single Text Inference")
    st.markdown("Enter Vietnamese text and select a task to classify or detect hate spans.")

    col_input, col_result = st.columns([1.2, 1])

    with col_input:
        selected_task = st.radio(
            "Task",
            list(TASK_PREFIXES.keys()),
            horizontal=True,
        )
        user_text = st.text_area(
            "Input text",
            height=120,
            placeholder="Nhập văn bản tiếng Việt tại đây…",
        )

        # Quick sample buttons
        st.markdown("**Quick samples:**")
        sample_cols = st.columns(3)
        with sample_cols[0]:
            if st.button("🟢 Clean", use_container_width=True):
                user_text = "Cảm ơn bạn đã chia sẻ thông tin hữu ích này!"
                st.session_state["_sample_text"] = user_text
        with sample_cols[1]:
            if st.button("🟡 Offensive", use_container_width=True):
                user_text = "Bạn nói chuyện như thế này thật không hay chút nào."
                st.session_state["_sample_text"] = user_text
        with sample_cols[2]:
            if st.button("🔴 Hate", use_container_width=True):
                user_text = "Đồ ngu như mày thì đừng có nói nữa!"
                st.session_state["_sample_text"] = user_text

        if "_sample_text" in st.session_state:
            user_text = st.session_state.pop("_sample_text")

        run_btn = st.button("▶ Run Inference", type="primary", use_container_width=True)

    with col_result:
        if run_btn and user_text.strip():
            task_prefix = TASK_PREFIXES[selected_task]
            start = time.time()
            output = run_t5_inference(user_text, task_prefix, model, tokenizer, device)
            elapsed = time.time() - start

            st.markdown(f"⏱️ Inference time: **{elapsed:.2f}s**")

            if "hate-spans-detection" in task_prefix:
                # Hate spans detection output
                st.subheader("Predicted Output")
                st.code(output)
                spans = extract_hate_spans(user_text, output)
                if spans:
                    st.subheader("Highlighted Hate Spans")
                    html = highlight_hate_spans_html(user_text, spans)
                    st.markdown(html, unsafe_allow_html=True)
                    st.caption(f"Detected {len(spans)} hate span(s)")
                else:
                    st.success("No hate spans detected in this text.")
            else:
                # Classification output
                label = output.upper()
                color = LABEL_COLORS.get(label, "#6c757d")
                st.markdown(
                    f"""
                    <div style="text-align:center;padding:30px;border-radius:12px;
                                background-color:{color}22;border:2px solid {color};">
                        <h2 style="color:{color};margin:0;">
                            {label}
                        </h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Run all 3 tasks for comprehensive view
                st.divider()
                st.markdown("**All-task results for this text:**")
                for task_name, prefix in TASK_PREFIXES.items():
                    if "hate-spans" in prefix:
                        out = run_t5_inference(user_text, prefix, model, tokenizer, device)
                        spans = extract_hate_spans(user_text, out)
                        span_info = f"{len(spans)} span(s)" if spans else "None"
                        st.markdown(f"- **{task_name}:** {span_info}")
                    else:
                        out = run_t5_inference(user_text, prefix, model, tokenizer, device)
                        lbl = out.upper()
                        c = LABEL_COLORS.get(lbl, "#6c757d")
                        st.markdown(
                            f"- **{task_name}:** <span style='color:{c};font-weight:700'>{lbl}</span>",
                            unsafe_allow_html=True,
                        )
        elif run_btn:
            st.warning("Please enter some text first.")

# ======================== TAB 2: Batch Inference ========================
with tab_batch:
    st.header("Batch Inference")
    st.markdown("Upload a CSV file with a text column to run batch predictions.")

    batch_task = st.selectbox("Task", list(TASK_PREFIXES.keys()), key="batch_task")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(10), use_container_width=True)

        text_col = st.selectbox("Select text column", df.columns.tolist())
        batch_run = st.button("▶ Run Batch Inference", type="primary")

        if batch_run:
            prefix = TASK_PREFIXES[batch_task]
            results = []
            progress = st.progress(0, text="Processing…")
            total = len(df)

            for i, text in enumerate(df[text_col].astype(str)):
                out = run_t5_inference(text, prefix, model, tokenizer, device)
                if "hate-spans" in prefix:
                    spans = extract_hate_spans(text, out)
                    results.append({
                        "text": text,
                        "raw_output": out,
                        "num_spans": len(spans),
                        "span_indices": str(spans),
                    })
                else:
                    results.append({
                        "text": text,
                        "prediction": out.upper(),
                    })
                progress.progress((i + 1) / total, text=f"Processing {i+1}/{total}…")

            result_df = pd.DataFrame(results)
            st.success(f"Processed {total} samples!")
            st.dataframe(result_df, use_container_width=True)

            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download results CSV",
                csv,
                "predictions.csv",
                "text/csv",
                use_container_width=True,
            )

# ======================== TAB 3: Model Comparison ========================
with tab_compare:
    st.header("Model Comparison — Benchmark Results")
    st.markdown(
        "Results reproduced from the ViHateT5 paper (Table 3 & 4). "
        "All scores are **Macro F1** on respective test sets."
    )

    # BERT vs ViHateT5
    st.subheader("All Models — Macro F1 Comparison")
    df_bench = pd.DataFrame(BENCHMARK_DATA)

    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(df_bench))
    width = 0.22
    tasks = ["ViHSD (MF1)", "ViCTSD (MF1)", "ViHOS (MF1)"]
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    for i, (task, color) in enumerate(zip(tasks, colors)):
        bars = ax.bar(x + i * width, df_bench[task], width, label=task, color=color)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(df_bench["Model"], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0.55, 0.95)
    ax.legend(loc="upper left")
    ax.set_title("Model Comparison — Macro F1 on Vietnamese HSD Benchmarks")
    fig.tight_layout()
    st.pyplot(fig)

    # Save chart
    fig.savefig("results/images/model_comparison_macro_f1.png", dpi=150, bbox_inches="tight")

    st.divider()

    # T5 Model comparison
    st.subheader("T5-based Models Comparison")
    df_t5 = pd.DataFrame(T5_COMPARISON_DATA)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    x2 = np.arange(len(df_t5))
    width2 = 0.2
    for i, (task, color) in enumerate(zip(tasks, colors)):
        bars2 = ax2.bar(x2 + i * width2, df_t5[task], width2, label=task, color=color)
        for bar in bars2:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.003, f"{h:.3f}",
                     ha="center", va="bottom", fontsize=8)

    ax2.set_xticks(x2 + width2)
    ax2.set_xticklabels(df_t5["Model"], fontsize=9)
    ax2.set_ylabel("Macro F1-score")
    ax2.set_ylim(0.6, 0.92)
    ax2.legend()
    ax2.set_title("T5-based Models — Macro F1 Comparison")
    fig2.tight_layout()
    st.pyplot(fig2)
    fig2.savefig("results/images/t5_comparison_macro_f1.png", dpi=150, bbox_inches="tight")

    st.divider()

    # Average MF1 ranking
    st.subheader("Average Macro F1 Ranking")
    df_rank = df_bench[["Model", "Average MF1"]].sort_values("Average MF1", ascending=True)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    bar_colors = ["#e15759" if "ViHateT5" in m else "#4e79a7" for m in df_rank["Model"]]
    ax3.barh(df_rank["Model"], df_rank["Average MF1"], color=bar_colors)
    for i, v in enumerate(df_rank["Average MF1"]):
        ax3.text(v + 0.003, i, f"{v:.4f}", va="center", fontsize=9)
    ax3.set_xlabel("Average Macro F1-score")
    ax3.set_xlim(0.6, 0.8)
    ax3.set_title("Average MF1 Ranking Across All Tasks")
    fig3.tight_layout()
    st.pyplot(fig3)
    fig3.savefig("results/images/average_mf1_ranking.png", dpi=150, bbox_inches="tight")

    st.divider()

    # Data tables
    st.subheader("Raw Data — All Models")
    st.dataframe(
        df_bench.style.highlight_max(subset=["ViHSD (MF1)", "ViCTSD (MF1)", "ViHOS (MF1)", "Average MF1"],
                                      color="#c6efce"),
        use_container_width=True,
    )

# ======================== TAB 4: Analysis ========================
with tab_analysis:
    st.header("Strengths & Weaknesses Analysis")

    st.subheader("1. Strengths of ViHateT5")
    st.markdown("""
    | # | Strength | Evidence |
    |---|----------|----------|
    | 1 | **Unified multi-task model** | Single model handles 3 tasks (ViHSD, ViCTSD, ViHOS) via task prefixes, eliminating the need for separate fine-tuned models |
    | 2 | **Domain-specific pre-training** | Pre-trained on VOZ-HSD (10M+ social media comments) — captures informal Vietnamese, slang, teencode better than Wikipedia-trained models |
    | 3 | **State-of-the-art performance** | Achieves highest Average MF1 (0.7556) across all 3 benchmark datasets |
    | 4 | **Syllable-level hate span detection** | T5 text-to-text architecture naturally handles span extraction via `[HATE]` tags — no need for IOB tagging |
    | 5 | **Text-to-text flexibility** | Can be extended to new tasks (Q&A, summarization) by simply changing the prefix |
    """)

    st.subheader("2. Weaknesses of ViHateT5")
    st.markdown("""
    | # | Weakness | Impact |
    |---|----------|--------|
    | 1 | **Auto-labeled pre-training data** | VOZ-HSD labels generated by ViSoBERT classifier may propagate errors — noisy labels limit the ceiling of pre-training quality |
    | 2 | **Large model size (223M params)** | ~2x larger than ViSoBERT (98M) but only marginal MF1 improvement (+0.49%). Not efficient for deployment |
    | 3 | **Slow inference (seq2seq generation)** | T5 autoregressive decoding is slower than BERT single-pass classification. Problematic for real-time applications |
    | 4 | **Single GPU training limitation** | Pre-trained on single NVIDIA A6000 — limited data scale (1.7GB) compared to XLM-R (2.5TB) |
    | 5 | **No cross-lingual capability** | Monolingual model — cannot transfer to other low-resource languages unlike mT5/XLM-R |
    | 6 | **ViHOS performance gap small** | On ViHOS, ViSoBERT (86.04%) nearly matches ViHateT5 (86.37%) with much fewer parameters |
    """)

    st.subheader("3. Comparison with Related Methods")
    st.markdown("""
    | Method | Approach | Pros | Cons |
    |--------|----------|------|------|
    | **BERT-based** (PhoBERT, ViSoBERT) | Encoder + classification head | Fast inference, small model | Separate model per task |
    | **mT5 / ViT5** | Text-to-text (general) | Multi-task, multilingual | Not domain-specific for HSD |
    | **ViHateT5** (this paper) | Text-to-text (domain-specific) | Best MF1, unified model | Slow inference, large model |
    | **FT5 / mFT5** (English) | Text-to-text for HSD | Proven architecture | English-only, no Vietnamese |
    | **HateBERT / fBERT** | Domain-specific BERT | Effective for English HSD | No Vietnamese support |
    """)

    st.subheader("4. Our Proposed Improvements")
    st.markdown("""
    In this reimplementation, we explore and propose the following improvements:

    **4.1 Data-Centric Improvement: Auto-Labeling Pipeline**
    - Built a ViSoBERT-based classifier to auto-label 10M+ samples from VOZ-HSD
    - Achieved 97.5% agreement with manual annotations
    - Enables large-scale domain-specific pre-training without manual annotation cost

    **4.2 Pre-training Data Strategy Experiments**
    - Compared **balanced** (50% hate / 50% clean) vs **hate-only** (100% hate) pre-training data
    - Found that balanced pre-training leads to better generalization across tasks
    - Original paper used only full-data (5.54% hate ratio) — our balanced approach provides different insights

    **4.3 Potential Future Improvements**
    | Improvement | Expected Benefit |
    |-------------|-----------------|
    | **Ensemble T5 + ViSoBERT** | Combine seq2seq and encoder predictions via weighted voting for higher accuracy |
    | **Focal Loss** | Address class imbalance (HATE class is underrepresented) during fine-tuning |
    | **Knowledge Distillation** | Compress ViHateT5 (223M) into smaller model while retaining performance |
    | **Data Augmentation** | Back-translation, synonym replacement for minority classes |
    | **Curriculum Learning** | Train on easy examples first, gradually introduce harder samples |
    """)

    # Pre-training data ratio impact chart
    st.subheader("5. Pre-training Data Ratio Impact")
    st.markdown("How different data ratios affect model performance (from paper Table 5):")

    pretrain_data = {
        "Config": [
            "100% hate (584K) — 10ep", "100% hate (584K) — 20ep",
            "50% hate (1.17M) — 10ep", "50% hate (1.17M) — 20ep",
            "5.54% hate (10.7M) — 10ep", "5.54% hate (10.7M) — 20ep",
        ],
        "ViHSD": [0.6548, 0.6577, 0.6600, 0.6620, 0.6286, 0.6800],
        "ViCTSD": [0.6134, 0.6258, 0.6022, 0.6642, 0.7358, 0.7027],
        "ViHOS": [0.8542, 0.8601, 0.8577, 0.8588, 0.8591, 0.8644],
    }
    df_pretrain = pd.DataFrame(pretrain_data)

    fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, task in enumerate(["ViHSD", "ViCTSD", "ViHOS"]):
        ax = axes[i]
        bars = ax.bar(range(len(df_pretrain)), df_pretrain[task],
                      color=["#a6cee3", "#1f78b4", "#b2df8a", "#33a02c", "#fb9a99", "#e31a1c"])
        ax.set_xticks(range(len(df_pretrain)))
        ax.set_xticklabels(df_pretrain["Config"], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Macro F1")
        ax.set_title(task)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.4f}",
                    ha="center", va="bottom", fontsize=7)
    fig4.suptitle("Pre-training Data Ratio Impact on Downstream Tasks", fontsize=12)
    fig4.tight_layout()
    st.pyplot(fig4)
    fig4.savefig("results/images/pretrain_data_ratio_impact.png", dpi=150, bbox_inches="tight")

# ======================== TAB 5: About ========================
with tab_about:
    st.header("About This Project")

    st.markdown("""
    ### DS200.Q21 — Big Data Analysis — Course Project (Group 02)
    **University of Information Technology (UIT — VNU-HCM)**

    ---

    #### Paper
    **ViHateT5: Enhancing Hate Speech Detection in Vietnamese With a Unified Text-to-Text Transformer Model**
    - *Author:* Luan Thanh Nguyen
    - *Published:* Findings of ACL 2024, pp. 5948–5961
    - *Link:* [ACL Anthology](https://aclanthology.org/2024.findings-acl.355.pdf)

    ---

    #### Team
    | No. | Student ID | Full Name | Role |
    |-----|-----------|-----------|------|
    | 1 | 23521143 | Phat Cong Nguyen | Leader |
    | 2 | 23520032 | An Thanh Hoang Truong | Member |
    | 3 | 23520213 | Cuong Viet Vu | Member |

    ---

    #### Repository
    - **GitHub:** [DS200.Q21_Project](https://github.com/paht2005/DS200.Q21_Project)
    - **HuggingFace Collection:** [NCPhat2005 — DS200.Q21](https://huggingface.co/collections/NCPhat2005/ds200q21-big-data-analysis-group-2)

    ---

    #### How to Run This Demo
    ```bash
    # Install dependencies
    pip install -r requirements.txt

    # Run the Streamlit app
    streamlit run app.py
    ```

    ---

    #### Methodology Overview

    This project reimplements and extends the ViHateT5 paper with three main pipelines:

    1. **Auto-Labeling Pipeline:** Use ViSoBERT to label 10M+ samples from VOZ forum data
    2. **Continual Pre-training:** T5 Span Corruption on domain-specific VOZ-HSD data
    3. **Multi-task Fine-tuning:** Unified T5 model for ViHSD, ViCTSD, and ViHOS tasks

    The demo supports all three downstream tasks:
    - **ViHSD:** 3-class hate speech detection (CLEAN / OFFENSIVE / HATE)
    - **ViCTSD:** Binary toxic speech detection (NONE / TOXIC)
    - **ViHOS:** Character-level hate span detection with `[HATE]` markers
    """)
