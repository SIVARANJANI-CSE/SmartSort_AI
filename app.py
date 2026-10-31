from __future__ import annotations

import html
import io
from pathlib import Path

import joblib
import streamlit as st
from PIL import Image

from image_classifier import (
    ImageDataset,
    TrainingResult,
    dataset_from_directory,
    dataset_from_zip,
    load_training_result,
    predict_image,
    predict_images,
    save_training_result,
    train_classifier,
)

APP_DIRECTORY = Path(__file__).parent
MODEL_PATH = APP_DIRECTORY / "models" / "image_classifier.joblib"
PAGE_OPTIONS = ("Classify waste", "Home", "About", "Tips", "Dataset info")
FEATURE_OVERVIEW = (
    (":material/insights: Smart analysis", "Classify waste in seconds with a fast HOG-based model."),
    (":material/folder_open: Dataset control", "Train from bundled data or your own ZIP archive."),
    (":material/collections_bookmark: Batch scan", "Upload multiple images and review all results in one pass."),
    (":material/download: Export", "Download your trained model for reuse or deployment."),
)
WASTE_DETAILS = {
    "O": {
        "name": "Organic Waste",
        "color": "#259E55",
        "soft_color": "#EAF7EE",
        "recommendation": "Place this in the organic waste bin. It can be composted and returned to the soil.",
        "bin_label": "ORGANIC WASTE",
    },
    "R": {
        "name": "Recyclable Waste",
        "color": "#0876D1",
        "soft_color": "#EAF3FF",
        "recommendation": "Place this in the recycling bin. Empty and clean it before sorting.",
        "bin_label": "RECYCLABLE",
    },
}

st.set_page_config(
    page_title="SmartSort AI | Waste classification",
    page_icon=":material/eco:",
    layout="wide",
)


@st.cache_data(max_entries=1)
def load_included_dataset(directory: str) -> ImageDataset:
    return dataset_from_directory(Path(directory))


def bin_illustration(color: str, label: str) -> str:
    safe_label = html.escape(label)
    return f"""
    <svg viewBox="0 0 160 190" role="img" aria-label="{safe_label} bin">
      <rect x="30" y="26" width="100" height="16" rx="7" fill="{color}"/>
      <path d="M38 46h84l-8 119a10 10 0 0 1-10 9H56a10 10 0 0 1-10-9z" fill="{color}"/>
      <path d="M56 18h48l8 9H48z" fill="{color}" opacity=".76"/>
      <rect x="48" y="64" width="64" height="58" rx="9" fill="white" opacity=".97"/>
      <path d="M80 72l-18 28h12l-3 16 26-30H84l5-14z" fill="{color}"/>
      <circle cx="51" cy="176" r="8" fill="#344054"/>
      <circle cx="109" cy="176" r="8" fill="#344054"/>
    </svg>
    """


def render_result_card(label: str, relative_score: float) -> None:
    details = WASTE_DETAILS.get(label)
    if details is None:
        details = {
            "name": f"Class {label}",
            "color": "#526174",
            "soft_color": "#F0F3F7",
            "recommendation": "Check your local waste-sorting guidance for this category.",
            "bin_label": "SORT BY LOCAL GUIDANCE",
        }

    category = html.escape(details["name"])
    color = details["color"]
    safe_label = html.escape(details["bin_label"])
    recommendation = html.escape(details["recommendation"])
    score = max(0, min(100, relative_score * 100))
    st.html(
        f"""
        <style>
          .sort-result,
          .sort-result * {{
            box-sizing: border-box !important;
          }}
          .sort-result {{
            display: grid !important;
            /* Proportional columns keep the recommendation wide and the bin compact. */
            grid-template-columns: minmax(0, 1.5fr) minmax(145px, 0.95fr) !important;
            gap: 22px !important;
            align-items: stretch !important;
            width: 100% !important;
            min-width: 0 !important;
            padding: 26px !important;
            min-height: 390px !important;
            border: 1px solid #dce5ee !important;
            border-radius: 22px !important;
            background: #ffffff !important;
            color: #172b4d !important;
            font-family: inherit !important;
            box-shadow: 0 12px 32px rgba(11, 40, 42, 0.08) !important;
          }}
          .sort-result > div:first-child {{
            min-width: 0 !important;
          }}
          .sort-result .eyebrow {{
            color: #526174 !important;
            font-size: .82rem !important;
            font-weight: 800 !important;
            letter-spacing: .06em !important;
          }}
          .sort-result h3 {{
            margin: 12px 0 18px !important;
            color: {color} !important;
            font-size: 1.65rem !important;
            line-height: 1.2 !important;
            font-weight: 800 !important;
          }}
          .sort-result .score {{
            margin: 0 !important;
            color: #8a9bb0 !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
          }}
          .sort-result .score strong {{
            color: {color} !important;
            font-weight: 800 !important;
          }}
          .sort-result .track {{
            height: 12px !important;
            margin: 14px 0 22px !important;
            overflow: hidden !important;
            border-radius: 99px !important;
            background: #e1e7eb !important;
          }}
          .sort-result .fill {{
            width: {score:.1f}% !important;
            height: 100% !important;
            border-radius: inherit !important;
            background: {color} !important;
          }}
          .sort-result .tip {{
            padding: 18px !important;
            border: 1px solid #edf1f5 !important;
            border-radius: 18px !important;
            background: #f1f7fd !important;
            color: #263648 !important;
            font-size: 1rem !important;
            line-height: 1.6 !important;
            font-weight: 500 !important;
          }}
          .sort-result .tip strong {{
            display: block !important;
            margin-bottom: 8px !important;
            color: {color} !important;
            font-size: 1.15rem !important;
            font-weight: 800 !important;
          }}
          .sort-result .bin {{
            display: flex !important;
            width: 100% !important;
            flex-direction: column !important;
            align-items: center !important;
            justify-content: center !important;
            gap: 12px !important;
            min-width: 0 !important;
            padding: 18px 10px !important;
            border-radius: 20px !important;
            background: #edf6ff !important;
          }}
          .sort-result svg {{
            display: block !important;
            width: min(100%, 145px) !important;
            height: 220px !important;
            max-width: 100% !important;
          }}
          .sort-result .bin span {{
            color: {color} !important;
            font-size: 1rem !important;
            font-weight: 800 !important;
            text-align: center !important;
          }}
          @media (max-width: 700px) {{
            .sort-result {{
              grid-template-columns: minmax(0, 1.35fr) minmax(120px, 0.85fr) !important;
              gap: 14px !important;
              padding: 18px !important;
              min-height: 330px !important;
            }}
            .sort-result h3 {{ font-size: 1.3rem !important; }}
            .sort-result .tip {{ padding: 14px !important; font-size: .92rem !important; }}
            .sort-result svg {{ width: min(100%, 112px) !important; height: 165px !important; }}
            .sort-result .bin span {{ font-size: .8rem !important; }}
          }}
          @media (max-width: 390px) {{
            .sort-result {{
              grid-template-columns: minmax(0, 1fr) 105px !important;
              gap: 10px !important;
              padding: 12px !important;
            }}
            .sort-result h3 {{ font-size: 1.12rem !important; }}
            .sort-result .tip {{ padding: 10px !important; font-size: .82rem !important; }}
            .sort-result svg {{ height: 135px !important; }}
          }}
        </style>
        <section class="sort-result">
          <div>
            <div class="eyebrow">ANALYSIS RESULT</div>
            <h3>{category}</h3>
            <p class="score">Model score: <strong>{score:.1f}%</strong></p>
            <div class="track"><div class="fill"></div></div>
            <div class="tip"><strong>Recommendation</strong>{recommendation}</div>
          </div>
          <div class="bin">{bin_illustration(color, safe_label)}<span>{safe_label}</span></div>
        </section>
        """
    )


def render_classifier(model: object | None) -> None:
    if model is None:
        st.info("A trained model is not available yet. Open Dataset info to train one.")
        return

    metric_cols = st.columns(3)
    metric_cols[0].metric("Classification", "2 labels", "O / R")
    metric_cols[1].metric("Model state", "Ready", "Local inference")
    metric_cols[2].metric("Best practice", "Check result", "Validate with local guidance")

    upload_column, result_column = st.columns((0.9, 1.5), gap="medium")
    with upload_column:
        with st.container(border=True):
            st.subheader("1. Upload waste image")
            st.caption("Use a clear image of a single item to get a sorting suggestion.")
            uploaded_image = st.file_uploader(
                "Upload an image (JPG, PNG, WEBP)",
                type=("jpg", "jpeg", "png", "bmp", "webp"),
                key="prediction_image",
            )
            if uploaded_image is not None:
                image_bytes = uploaded_image.getvalue()
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        preview = image.copy()
                except (OSError, ValueError) as exc:
                    st.error(f"Could not open this image: {exc}", icon=":material/error:")
                else:
                    st.image(
                        preview,
                        caption=uploaded_image.name,
                        width="stretch",
                    )

            st.markdown("### Batch scan")
            batch_files = st.file_uploader(
                "Upload multiple images to classify in one batch",
                type=("jpg", "jpeg", "png", "bmp", "webp"),
                accept_multiple_files=True,
                key="batch_predictions",
            )
            if batch_files:
                try:
                    predictions = predict_images(model, [item.getvalue() for item in batch_files])
                except (OSError, ValueError) as exc:
                    st.error(f"Batch classification failed: {exc}", icon=":material/error:")
                else:
                    batch_rows = [
                        {
                            "Image": file.name,
                            "Prediction": label,
                            "Confidence": f"{score:.1%}",
                        }
                        for file, (label, score) in zip(batch_files, predictions)
                    ]
                    st.dataframe(batch_rows, hide_index=True, use_container_width=True)

    with result_column:
        with st.container(border=True):
            st.subheader("2. Analysis result")
            if uploaded_image is None:
                st.html(
                    """
                    <div style="min-height:260px;display:grid;place-items:center;
                        border:1px dashed rgba(255,255,255,0.28);border-radius:14px;
                        background:rgba(255,255,255,0.08);color:#f3f7f6;text-align:center;padding:20px">
                      <div>
                        <div style="font-size:2.3rem;color:#7ce7b5">♻</div>
                        <strong style="color:#f3f7f6;font-weight:800;display:block;margin-bottom:8px; font-size:1.1rem;">Your result will appear here</strong>
                        <p style="margin:8px 0 0;color:#edf2f4;font-weight:600;line-height:1.4;">Upload a waste image to get a sorting suggestion.</p>
                      </div>
                    </div>
                    """
                )
            elif "preview" in locals():
                try:
                    predicted_label, relative_score = predict_image(model, image_bytes)
                except (OSError, ValueError) as exc:
                    st.error(f"Could not classify this image: {exc}", icon=":material/error:")
                else:
                    render_result_card(predicted_label, relative_score)
                    st.caption(
                        "The model score is a relative score, not a calibrated probability."
                    )


if "training_result" not in st.session_state:
    st.session_state["training_result"] = None
    if MODEL_PATH.is_file():
        try:
            st.session_state["training_result"] = load_training_result(MODEL_PATH)
        except (EOFError, OSError, ValueError) as exc:
            st.session_state["model_load_error"] = str(exc)

training_result: TrainingResult | None = st.session_state["training_result"]
with st.sidebar:
    st.markdown("## :material/eco: SmartSort AI")
    st.caption("Professional waste classification studio")
    st.divider()
    page = st.radio(
        "Navigation",
        options=PAGE_OPTIONS,
        index=0,
        key="navigation",
        label_visibility="collapsed",
    )
    st.divider()
    if training_result is not None:
        st.success("AI model ready", icon=":material/check_circle:")
        st.caption(f"Latest accuracy: {training_result.accuracy:.1%}")
    else:
        st.info("Train a model to begin", icon=":material/info:")
    st.caption("Keep our environment clean and green!")

st.html(
    """
    <style>
      .stApp {
        background: linear-gradient(135deg, #0b1211 0%, #111d1a 36%, #0d1718 100%);
        color: #edf7f3;
      }
      .smart-hero {
        margin: 0 0 18px;
        padding: 24px 28px;
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 22px;
        background: rgba(18, 28, 26, 0.9);
        box-shadow: 0 14px 30px rgba(0,0,0,0.22);
        color: #edf7f3;
      }
      .smart-hero .brand {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 760;
        letter-spacing: -.04em;
      }
      .smart-hero .tagline {
        margin: 6px 0 0;
        color: #c9d9d3;
        font-size: .98rem;
      }
      .smart-ready {
        display: inline-block;
        margin-top: 16px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(18, 201, 135, 0.18);
        color: #9ef3cb;
        font-size: .78rem;
        font-weight: 700;
        border: 1px solid rgba(158, 243, 203, 0.2);
      }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(11, 17, 16, 0.96), rgba(15, 22, 21, 0.96));
        border-right: 1px solid rgba(255,255,255,0.06);
      }
      .stContainer > div {
        border-radius: 18px;
      }
      .block-container {
        padding-top: 2rem;
      }
      .stRadio > div, .stSelectbox, .stDataFrame, .stButton > button,
      .stFileUploader, .stTextInput, .stNumberInput, .stSlider {
        border-radius: 14px;
      }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(15, 22, 21, 0.8);
        border: 1px solid rgba(255,255,255,0.06);
      }
      .stApp {
        color: #f3f7f6;
      }
      .stMarkdown, .stCaption, .stDataFrame, .stMetric, .stSuccess, .stInfo, .stError,
      .stButton, .stRadio label, .stCheckbox label, .stSelectbox label,
      .stTextInput label, .stNumberInput label, .stSlider label,
      h1, h2, h3, h4, h5, h6, p, li {
        color: #f3f7f6 !important;
      }
      div[data-testid="stFileUploader"] {
        background: rgba(15, 22, 21, 0.9) !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        border-radius: 16px !important;
      }
      div[data-testid="stFileUploader"] * {
        color: #f3f7f6 !important;
        font-weight: 700 !important;
      }
      div[data-testid="stFileUploader"] label,
      div[data-testid="stFileUploader"] button,
      div[data-testid="stFileUploader"] .st-emotion-cache-1hfw9n3,
      div[data-testid="stFileUploader"] .st-emotion-cache-18rm35g {
        color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 0 0 0 rgba(0,0,0,0) !important;
      }
      .stAppHeader {
        display: none;
      }
      header[data-testid="stHeader"] {
        display: none;
      }
      .stRadio [role="radio"] {
        background: rgba(255,255,255,0.18);
      }
      .stRadio [role="radio"][aria-checked="true"] {
        background: rgba(24, 224, 152, 0.8);
      }
    </style>
    """
)

if page in ("Classify waste", "Home"):
    st.html(
        f"""
        <header class="smart-hero">
          <p class="brand">🌿 SmartSort AI</p>
          <p class="tagline">AI-powered waste classification for a cleaner tomorrow.</p>
          <span class="smart-ready">
            {"Model ready · " + f"{training_result.accuracy:.1%}" + " test accuracy" if training_result else "Train a model to get started"}
          </span>
        </header>
        """
    )
    if "model_load_error" in st.session_state:
        st.error(
            "The saved model could not be loaded. Train a new model from Dataset info. "
            f"Details: {st.session_state['model_load_error']}",
            icon=":material/error:",
        )
    feature_cols = st.columns(4)
    for column, (title, description) in zip(feature_cols, FEATURE_OVERVIEW):
        with column:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.caption(description)
    render_classifier(training_result.model if training_result else None)

elif page == "About":
    st.title("About SmartSort AI")
    st.write(
        "SmartSort AI helps sort an item into the Organic Waste or Recyclable Waste "
        "category using the locally trained image classifier."
    )
    st.info(
        "This is a computer-vision suggestion, not an official waste-disposal decision. "
        "Local recycling rules vary; check your local guidance."
    )

elif page == "Tips":
    st.title("Waste sorting tips")
    tip_columns = st.columns(2)
    with tip_columns[0]:
        with st.container(border=True):
            st.markdown("### :material/compost: Organic waste")
            st.write(
                "Fruit and vegetable scraps are usually compostable. Keep plastic, "
                "glass, and metal out of the compost bin."
            )
    with tip_columns[1]:
        with st.container(border=True):
            st.markdown("### :material/recycling: Recyclable waste")
            st.write(
                "Empty containers and follow local rules for sorting plastic, paper, "
                "glass, and metal."
            )

elif page == "Dataset info":
    st.title("Dataset and model")
    dataset_source = st.segmented_control(
        "Dataset source",
        options=("Included dataset", "Upload ZIP"),
        default="Included dataset",
        key="dataset_source",
    )
    uploaded_dataset = None
    if dataset_source == "Upload ZIP":
        uploaded_dataset = st.file_uploader(
            "Upload a ZIP containing TRAIN/<label>/<image> folders",
            type=("zip",),
            key="dataset_zip",
        )

    dataset: ImageDataset | None = None
    try:
        if dataset_source == "Included dataset":
            local_dataset = APP_DIRECTORY / "DATASET"
            if local_dataset.is_dir():
                dataset = load_included_dataset(str(local_dataset))
            else:
                st.info(
                    "The training dataset is not bundled with this deployment. "
                    "Choose Upload ZIP to train with your own labeled dataset."
                )
        elif uploaded_dataset is not None:
            dataset = dataset_from_zip(uploaded_dataset.getvalue())
        else:
            st.info("Upload a labeled ZIP to inspect its classes and train a model.")
    except (OSError, ValueError) as exc:
        st.error(str(exc), icon=":material/error:")

    if dataset is not None:
        train_counts = {
            label: sum(record.label == label for record in dataset.train_records)
            for label in dataset.classes
        }
        test_counts = {
            label: sum(record.label == label for record in dataset.test_records)
            for label in dataset.classes
        }
        metric_columns = st.columns(3)
        metric_columns[0].metric("Labels", len(dataset.classes))
        metric_columns[1].metric("Training images", f"{len(dataset.train_records):,}")
        metric_columns[2].metric("Test images", f"{len(dataset.test_records):,}")
        st.dataframe(
            [
                {
                    "Label": label,
                    "Category": WASTE_DETAILS.get(label, {}).get("name", f"Class {label}"),
                    "Training images": train_counts[label],
                    "Test images": test_counts[label],
                }
                for label in dataset.classes
            ],
            hide_index=True,
        )
        with st.container(border=True):
            st.subheader("Train or update the model")
            st.caption("A supplied TEST split is used to measure model accuracy.")
            with st.form("training_form"):
                per_class_limit = st.slider(
                    "Maximum images per label in each split",
                    min_value=50,
                    max_value=5_000,
                    value=5_000,
                    step=50,
                )
                train_clicked = st.form_submit_button(
                    "Train model",
                    type="primary",
                    icon=":material/model_training:",
                )
        if train_clicked:
            progress_bar = st.progress(0, text="Starting training")

            def report_progress(phase: str, current: int, total: int) -> None:
                progress_bar.progress(
                    current / max(total, 1),
                    text=f"{phase}: {current:,} / {total:,}",
                )

            try:
                result = train_classifier(
                    dataset,
                    per_class_limit=per_class_limit,
                    progress=report_progress,
                )
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(f"Training failed: {exc}", icon=":material/error:")
            else:
                st.session_state["training_result"] = result
                training_result = result
                st.session_state.pop("model_load_error", None)
                progress_bar.progress(1.0, text="Training complete")
                try:
                    save_training_result(result, MODEL_PATH)
                except OSError as exc:
                    st.error(f"Model trained, but could not be saved: {exc}", icon=":material/error:")
                else:
                    st.success("Model trained and saved on this machine.")
                st.metric("Test accuracy", f"{result.accuracy:.1%}")
                st.caption(
                    f"{result.training_count:,} training images · "
                    f"{result.validation_count:,} validation images · "
                    f"{result.validation_source}"
                )

    if training_result is not None:
        st.subheader("Current model")
        st.metric("Validation accuracy", f"{training_result.accuracy:.1%}")
        st.caption(
            "This score is not a guarantee of real-world performance. "
            "The prediction score shown during classification is not calibrated."
        )
        model_buffer = io.BytesIO()
        joblib.dump(training_result.model, model_buffer)
        st.download_button(
            "Download trained model",
            data=model_buffer.getvalue(),
            file_name="image_classifier.joblib",
            mime="application/octet-stream",
            icon=":material/download:",
        )
