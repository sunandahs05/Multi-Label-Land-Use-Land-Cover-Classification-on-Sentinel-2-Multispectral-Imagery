from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "temporal_data"

st.set_page_config(
    page_title="Temporal LULC Explorer",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Polished, restrained green research-demo theme.
st.markdown("""
<style>
:root { --forest: #176b4d; --leaf: #2d8a62; --mint: #eaf6ef; --ink: #18352b; }
.stApp { background: linear-gradient(180deg, #fbfdfb 0%, #ffffff 28%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #edf6f0 0%, #f7faf8 100%); border-right: 1px solid #d9e9df; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #176b4d; }
h1, h2, h3 { color: #173e30; letter-spacing: -0.025em; }
[data-testid="stMetric"] { background: #f2f8f4; border: 1px solid #d9eadf; padding: 0.85rem 1rem; border-radius: 14px; }
[data-testid="stMetricLabel"] { color: #4a6d5c; }
[data-testid="stMetricValue"] { color: #176b4d; }
.stButton > button { border-radius: 10px; border-color: #b7d8c4; }
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; }
div[data-testid="stAlert"] { border-radius: 12px; }
.small-note { color: #597367; font-size: 0.9rem; }
.hero { padding: 1.25rem 1.4rem; border-radius: 18px; background: linear-gradient(110deg, #e7f5ec, #f6fbf7); border: 1px solid #d4e9dc; margin-bottom: 1rem; }
.hero-kicker { text-transform: uppercase; letter-spacing: .12em; color: #287453; font-size: .75rem; font-weight: 700; }
.hero-title { font-size: 1.8rem; font-weight: 750; color: #163d2e; margin: .25rem 0; }
.hero-sub { color: #426353; }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="hero"><div class="hero-kicker">BigEarthNet-v2 · Research dashboard</div><div class="hero-title">Temporal LULC Explorer</div><div class="hero-sub">From multispectral observations to temporal features and multi-label predictions.</div></div>', unsafe_allow_html=True)

# A restrained visual style that works with Streamlit's light/dark themes.
st.caption("Sentinel-2 multispectral imagery · Multi-label land-use/land-cover classification")

PAGES = [
    "Project overview",
    "Model comparison",
    "Training analysis",
    "Dataset & methodology",
    "Temporal imagery & features",
    "Available files",
]
page = st.sidebar.radio("Navigate", PAGES)
st.sidebar.divider()
st.sidebar.caption("Results are loaded from the local JSON/CSV files in `temporal_data/`. No training is started by this dashboard.")


def read_json(relative_path: str) -> dict[str, Any] | None:
    path = ROOT / relative_path
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            value = json.load(f)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_csv(relative_path: str) -> pd.DataFrame | None:
    path = ROOT / relative_path
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError):
        return None


def fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def json_model_rows(relative_path: str, group: str) -> list[dict[str, Any]]:
    data = read_json(relative_path)
    if not data:
        return []
    rows: list[dict[str, Any]] = []
    # Controlled comparison JSON uses model name -> metrics.
    if relative_path.endswith("controlled_comparison/results.json"):
        for key, metrics in data.items():
            if isinstance(metrics, dict):
                rows.append({
                    "Experiment group": group,
                    "Model": key.replace("_", " ").title(),
                    **metrics,
                })
    else:
        rows.append({
            "Experiment group": group,
            "Model": str(data.get("experiment", group)),
            **data,
        })
    return rows


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalised = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        if key in normalised:
            return normalised[key]
    # Allow common prefixes/suffixes.
    for candidate in candidates:
        key = candidate.lower().replace(" ", "_")
        for norm, original in normalised.items():
            if key in norm:
                return original
    return None


def metric_chart(df: pd.DataFrame, metrics: list[str], title: str):
    available = [m for m in metrics if m in df.columns]
    if not available:
        st.info("No recognised metric columns were found in this result file. The raw table is shown below.")
        st.dataframe(df, width="stretch", hide_index=True)
        return
    label_col = "Model" if "Model" in df.columns else df.columns[0]
    melted = df.melt(id_vars=[label_col], value_vars=available, var_name="Metric", value_name="Score")
    melted["Score"] = pd.to_numeric(melted["Score"], errors="coerce")
    melted = melted.dropna(subset=["Score"])
    fig = px.bar(
        melted, x="Metric", y="Score", color=label_col, barmode="group",
        title=title, text_auto=".3f",
    )
    # Keep one distinct, consistent colour per model rather than colouring every bar alike.
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", legend_title_text="",
        yaxis_range=[0, 1], margin=dict(l=10, r=10, t=60, b=10),
        colorway=["#16805d", "#3987c9", "#e39b32", "#9467bd", "#d65f5f"],
    )
    st.plotly_chart(fig, width="stretch")


def discover_image_files() -> list[Path]:
    """Find optional real image examples placed in conventional demo folders."""
    folders = [ROOT / "data" / "demo_images", ROOT / "temporal_data" / "demo_images", ROOT / "demo_images"]
    extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
    found: list[Path] = []
    for folder in folders:
        if folder.exists():
            found.extend(x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in extensions)
    return sorted(set(found))

def discover_tensor_files() -> list[Path]:
    folders = [ROOT / "temporal_data" / "sample_tensors", ROOT / "data" / "demo_tensors"]
    found: list[Path] = []
    for folder in folders:
        if folder.exists():
            found.extend(x for x in folder.glob("*.npy") if x.is_file())
    return sorted(found)

def to_hwc(tensor: np.ndarray) -> np.ndarray:
    """Convert one six-band image to HWC if it is in CHW format."""
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D image, received shape {tensor.shape}")
    if tensor.shape[0] in (3, 6) and tensor.shape[-1] not in (3, 6):
        tensor = np.transpose(tensor, (1, 2, 0))
    return tensor

def make_rgb(image: np.ndarray) -> np.ndarray:
    """Render Sentinel-2 true-colour RGB from channel order B02,B03,B04,B08,B11,B12."""
    image = to_hwc(np.asarray(image, dtype=np.float32))
    if image.shape[-1] >= 6:
        rgb = image[..., [2, 1, 0]]  # Red=B04, Green=B03, Blue=B02
    elif image.shape[-1] == 3:
        rgb = image
    else:
        raise ValueError(f"Need 3 RGB channels or 6 Sentinel-2 bands; got {image.shape}")
    finite = rgb[np.isfinite(rgb)]
    if finite.size == 0:
        return np.zeros((*rgb.shape[:2], 3), dtype=np.uint8)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    rgb = np.nan_to_num((rgb - lo) / (hi - lo), nan=0.0, posinf=1.0, neginf=0.0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


@st.cache_data(show_spinner="Loading the matching feature sequence…")
def find_temporal_feature_row(spatial_id: str) -> tuple[str, int, np.ndarray, np.ndarray] | None:
    """Resolve a sample location to the correct split-specific compressed feature cache."""
    manifest_path = DATA_DIR / "temporal_sequences.csv"
    cache_dir = DATA_DIR / "full_feature_cache"
    if not manifest_path.exists():
        return None
    try:
        manifest = pd.read_csv(manifest_path)
        if not {"spatial_id", "split"}.issubset(manifest.columns):
            return None
        matches = manifest.loc[manifest["spatial_id"].astype(str) == str(spatial_id)]
        if matches.empty:
            return None
        split = str(matches.iloc[0]["split"])
        split_rows = manifest.loc[manifest["split"].astype(str) == split].reset_index(drop=True)
        row_matches = split_rows.index[split_rows["spatial_id"].astype(str) == str(spatial_id)].tolist()
        if not row_matches:
            return None
        row = int(row_matches[0])
        cache_path = cache_dir / f"temporal_features_{split}_seed2026.npz"
        if not cache_path.exists():
            return None
        with np.load(cache_path, allow_pickle=False) as cache:
            x_key = "X_temporal" if "X_temporal" in cache.files else None
            y_key = "Y" if "Y" in cache.files else None
            if x_key is None or y_key is None:
                return None
            if row >= cache[x_key].shape[0]:
                return None
            sequence = np.asarray(cache[x_key][row], dtype=np.float32)
            target = np.asarray(cache[y_key][row], dtype=np.float32)
        if sequence.shape != (4, 2048):
            return None
        return split, row, sequence, target
    except (OSError, ValueError, KeyError, pd.errors.ParserError):
        return None


def render_feature_heatmap(sequence: np.ndarray, title: str = "Feature activations across four dates"):
    """Render the real four-by-2048 feature sequence with a diverging colour scale."""
    fig = go.Figure(data=go.Heatmap(
        z=sequence,
        x=np.arange(1, sequence.shape[1] + 1),
        y=["Date 1", "Date 2", "Date 3", "Date 4"],
        colorscale="RdYlBu_r",
        zmid=float(np.nanmedian(sequence)),
        colorbar=dict(title="Feature value"),
        hovertemplate="%{y}<br>Feature %{x}<br>Value: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=title, template="plotly_white", height=330,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Feature index (1–2048)", yaxis_title="Observation date",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig

# Read the project's saved experiment results. Groups remain separate by design.
baseline_test = read_csv("temporal_data/full_data_test_comparison.csv")
baseline_validation = read_csv("temporal_data/full_data_validation_comparison.csv")
pilot = read_csv("temporal_data/improved_pilot_comparison.csv")
gru_metrics = read_json("temporal_data/gru_experiment/test_metrics.json")
attention_metrics = read_json("temporal_data/attention_experiment/test_metrics.json")
controlled = read_json("temporal_data/controlled_comparison/results.json")

if page == "Project overview":
    st.markdown(
        "This project studies whether multiple observations of the same geographic location "
        "help multi-label land-use/land-cover (LULC) classification compared with a single observation."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Indexed observations", "480,038")
    c2.metric("LULC labels", "19")
    c3.metric("Spectral bands", "6")
    c4.metric("Dates per sequence", "4")
    st.subheader("Pipeline at a glance")
    steps = [
        ("1 · Dataset", "BigEarthNet-v2 metadata and Sentinel-2 observations"),
        ("2 · Temporal cohort", "Four observations per eligible location; fixed train/validation/test splits"),
        ("3 · Feature extraction", "Pretrained ResNet50 produces a 2,048-dimensional feature vector per date"),
        ("4 · Temporal aggregation", "Compare uniform averaging, GRU and learned temporal attention"),
        ("5 · Evaluation", "Macro-F1, Micro-F1, mean average precision (mAP), and loss"),
    ]
    for heading, detail in steps:
        with st.container(border=True):
            st.markdown(f"**{heading}**")
            st.write(detail)
    st.subheader("Current experiment result files")
    status_cols = st.columns(3)
    status_cols[0].metric("GRU metrics", "Available" if gru_metrics else "Not found")
    status_cols[1].metric("Attention metrics", "Available" if attention_metrics else "Not found")
    status_cols[2].metric("Controlled comparison", "Available" if controlled else "Not found")
    st.info("This dashboard visualises saved experiment results. It does not launch training or claim that every experiment used identical settings.")

elif page == "Model comparison":
    st.header("Model comparison")
    st.caption("Compare scores only within experiments that share a compatible training setup.")
    st.subheader("Controlled comparison: uniform averaging vs temporal attention")
    rows = json_model_rows("temporal_data/controlled_comparison/results.json", "Controlled comparison")
    if rows:
        controlled_df = pd.DataFrame(rows)
        display_cols = [c for c in ["Model", "best_epoch", "best_validation_loss", "test_loss", "macro_f1", "micro_f1", "mAP", "threshold"] if c in controlled_df.columns]
        st.dataframe(controlled_df[display_cols].rename(columns={
            "best_epoch": "Best epoch",
            "best_validation_loss": "Best validation loss",
            "test_loss": "Test loss",
            "macro_f1": "Macro-F1",
            "micro_f1": "Micro-F1",
            "mAP": "mAP",
            "threshold": "Threshold",
        }), width="stretch", hide_index=True)
        chart_df = controlled_df.rename(columns={
            "macro_f1": "Macro-F1", "micro_f1": "Micro-F1", "mAP": "mAP"
        })
        metric_chart(chart_df, ["Macro-F1", "Micro-F1", "mAP"], "Controlled test metrics")
        st.markdown(
            "**Interpretation:** in this controlled run, temporal attention has higher Macro-F1 and Micro-F1, "
            "while uniform averaging has higher mAP. There is no single winner across all three metrics."
        )
    else:
        st.warning("Controlled comparison results were not found.")

    st.divider()
    st.subheader("Earlier GRU and attention experiments")
    rows = []
    if gru_metrics:
        rows.extend(json_model_rows("temporal_data/gru_experiment/test_metrics.json", "Earlier GRU experiment"))
    if attention_metrics:
        rows.extend(json_model_rows("temporal_data/attention_experiment/test_metrics.json", "Earlier attention experiment"))
    if rows:
        df = pd.DataFrame(rows)
        wanted = ["Experiment group", "Model", "best_epoch", "best_validation_loss", "test_loss", "macro_f1", "micro_f1", "mAP", "threshold"]
        present = [c for c in wanted if c in df.columns]
        st.dataframe(df[present].rename(columns={
            "best_epoch": "Best epoch", "best_validation_loss": "Best validation loss",
            "test_loss": "Test loss", "macro_f1": "Macro-F1",
            "micro_f1": "Micro-F1", "mAP": "mAP", "threshold": "Threshold",
        }), width="stretch", hide_index=True)
        st.caption("These earlier experiments are displayed separately from the controlled comparison because the training setups are not guaranteed to be identical.")

    st.divider()
    st.subheader("Full-data single-date vs four-date baseline")
    if baseline_test is not None:
        st.write("Saved file: `temporal_data/full_data_test_comparison.csv`")
        st.dataframe(baseline_test, width="stretch", hide_index=True)
        st.caption("The source table is shown verbatim first. Metric plotting will use its actual column names, avoiding assumptions about the CSV schema.")
    else:
        st.warning("The full-data test comparison CSV was not found.")
    with st.expander("Other saved comparison files"):
        if baseline_validation is not None:
            st.markdown("**Validation comparison**")
            st.dataframe(baseline_validation, width="stretch", hide_index=True)
        if pilot is not None:
            st.markdown("**Pilot comparison**")
            st.dataframe(pilot, width="stretch", hide_index=True)

elif page == "Training analysis":
    st.header("Training analysis")
    histories = [
        ("Controlled uniform average", "temporal_data/controlled_comparison/uniform_average_history.csv"),
        ("Controlled temporal attention", "temporal_data/controlled_comparison/temporal_attention_history.csv"),
        ("Earlier GRU", "temporal_data/gru_experiment/training_history.csv"),
        ("Earlier temporal attention", "temporal_data/attention_experiment/training_history.csv"),
    ]
    found = [(label, path, read_csv(path)) for label, path in histories]
    found = [(label, path, df) for label, path, df in found if df is not None and not df.empty]
    if not found:
        st.warning("No training-history CSV files were found.")
    else:
        labels = [label for label, _, _ in found]
        selected = st.selectbox("Choose a training history", labels)
        label, path, df = next(item for item in found if item[0] == selected)
        st.caption(f"Source: `{path}`")
        epoch_col = find_column(df, ["epoch", "epochs", "step"])
        loss_candidates = [c for c in df.columns if "loss" in str(c).lower()]
        if epoch_col and loss_candidates:
            plot_df = df.copy()
            plot_df[epoch_col] = pd.to_numeric(plot_df[epoch_col], errors="coerce")
            for col in loss_candidates:
                plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")
            plot_df = plot_df.dropna(subset=[epoch_col])
            fig = go.Figure()
            for col in loss_candidates:
                fig.add_trace(go.Scatter(
                    x=plot_df[epoch_col], y=plot_df[col],
                    mode="lines", name=str(col).replace("_", " ").title(),
                ))
            fig.update_layout(
                template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=f"{label}: loss by epoch", xaxis_title="Epoch",
                yaxis_title="Loss", legend_title_text="", margin=dict(l=10, r=10, t=60, b=10),
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Could not automatically identify epoch and loss columns; showing the saved history table.")
        st.dataframe(df, width="stretch", hide_index=True)
    st.caption("Training histories are plotted from saved CSVs; no model training runs when opening this page.")

elif page == "Dataset & methodology":
    st.header("Dataset & methodology")
    st.subheader("Dataset summary")
    dataset_rows = [
        ("Dataset", "BigEarthNet-v2"),
        ("Indexed observations", "480,038"),
        ("Parquet shards", "64"),
        ("Land-cover labels", "19"),
        ("Bands per observation", "6"),
        ("Observations per temporal sequence", "4"),
        ("Eligible locations", "30,011"),
        ("Eligible observations", "120,044"),
    ]
    st.table(pd.DataFrame(dataset_rows, columns=["Property", "Value"]))
    st.subheader("Six Sentinel-2 bands")
    st.table(pd.DataFrame([
        ("B02", "Blue"), ("B03", "Green"), ("B04", "Red"),
        ("B08", "Near-infrared (NIR)"), ("B11", "Short-wave infrared 1 (SWIR1)"),
        ("B12", "Short-wave infrared 2 (SWIR2)"),
    ], columns=["Band", "Description"]))
    st.subheader("Temporal feature representation")
    st.code("(batch, 4 dates, 6 bands, 120, 120) → ResNet50 → (batch, 4 dates, 2048 features)")
    st.markdown(
        "- **Uniform averaging:** averages the four date-level feature vectors.\n"
        "- **GRU:** processes the ordered sequence of date-level feature vectors.\n"
        "- **Temporal attention:** learns weights over the date-level feature vectors.\n\n"
        "The output is multi-label: each location can have multiple land-cover labels. "
        "The saved evaluation uses Macro-F1, Micro-F1 and mAP, with a threshold of 0.5 in the listed experiments."
    )
    st.warning("The project predicts labels associated with a patch; it should not be described as direct land-cover change detection unless a separate change-detection target is implemented.")

elif page == "Temporal imagery & features":
    st.header("Temporal imagery & feature explorer")
    st.markdown("Inspect a location across four dates, then see how each observation becomes a feature vector. This page is a visual explainer; it does not run feature extraction.")
    tensor_files = discover_tensor_files()
    image_files = discover_image_files()
    tab_images, tab_features, tab_setup = st.tabs(["🛰️ Four-date image sequence", "🧬 Extracted features", "📂 Add your own examples"])

    with tab_images:
        if tensor_files:
            # Prefer the bundled multi-location sample tensor when present.
            preferred = [p for p in tensor_files if p.name == "sample_X.npy"]
            default_index = tensor_files.index(preferred[0]) if preferred else 0
            selected_tensor = st.selectbox(
                "Choose a saved temporal tensor", tensor_files, index=default_index,
                format_func=lambda x: str(x.relative_to(ROOT)),
            )
            try:
                arr = np.load(selected_tensor, mmap_mode="r", allow_pickle=False)
                meta_candidates = [selected_tensor.with_name("sample_metadata.json"), selected_tensor.parent / "sample_metadata.json"]
                metadata = None
                for meta_path in meta_candidates:
                    if meta_path.exists():
                        try:
                            with meta_path.open("r", encoding="utf-8-sig") as handle:
                                metadata = json.load(handle)
                            break
                        except (OSError, json.JSONDecodeError):
                            pass

                if arr.ndim == 5 and arr.shape[1] == 4 and arr.shape[2] == 6:
                    n_locations = int(arr.shape[0])
                    meta_list = metadata if isinstance(metadata, list) else []
                    def location_label(i: int) -> str:
                        if i < len(meta_list) and isinstance(meta_list[i], dict):
                            item = meta_list[i]
                            return f"{i + 1:02d} · {item.get('spatial_id', f'Location {i + 1}')} · {item.get('split', 'unknown split')}"
                        return f"{i + 1:02d} · Location {i + 1}"
                    selected_location = st.selectbox(
                        "Select a location", range(n_locations), format_func=location_label,
                        key=f"location_{selected_tensor.name}",
                    )
                    sequence = np.asarray(arr[selected_location])
                    item = meta_list[selected_location] if selected_location < len(meta_list) and isinstance(meta_list[selected_location], dict) else {}
                    dates = item.get("dates", [])
                    labels_by_date = item.get("label_names_by_date", [])
                    st.markdown(f"**Location:** `{item.get('spatial_id', f'Location {selected_location + 1}')}`  ·  **Split:** `{item.get('split', 'not recorded')}`")
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        date_text = dates[i] if i < len(dates) else f"Observation {i + 1}"
                        with col:
                            st.markdown(f"**Date {i + 1}**")
                            st.image(make_rgb(sequence[i]), caption=date_text, width="stretch")
                            if i < len(labels_by_date) and labels_by_date[i]:
                                st.caption("Labels: " + ", ".join(str(x).replace("_", " ") for x in labels_by_date[i]))
                    if item.get("image_ids"):
                        with st.expander("View source image identifiers"):
                            for date_text, image_id in zip(dates, item["image_ids"]):
                                st.text(f"{date_text}: {image_id}")
                    st.caption(f"Source: `{selected_tensor.relative_to(ROOT)}` · full array `{arr.shape}` · selected sequence `{sequence.shape}`. RGB composite uses B04/B03/B02; the model receives all six bands.")
                    if n_locations > 1:
                        st.success(f"Loaded {n_locations} real sample locations with four observations each. Select another location above for a second example.")
                elif arr.ndim == 4 and arr.shape[0] == 4:
                    cols = st.columns(4)
                    for i, col in enumerate(cols):
                        with col:
                            st.markdown(f"**Observation {i + 1}**")
                            st.image(make_rgb(np.asarray(arr[i])), caption=f"Observation {i + 1} · RGB composite", width="stretch")
                    st.caption(f"Source: `{selected_tensor.relative_to(ROOT)}` · tensor shape `{arr.shape}` · RGB uses B04/B03/B02.")
                else:
                    st.warning(f"Tensor shape {arr.shape} is not a recognised four-date tensor or batch of four-date tensors.")
            except Exception as exc:
                st.error(f"Could not render this tensor: {exc}")
        elif image_files:
            st.info("No four-date tensor found, but image files are available. Select a location group or inspect images below.")
            groups: dict[str, list[Path]] = {}
            for path in image_files:
                groups.setdefault(path.parent.name, []).append(path)
            group_name = st.selectbox("Image folder", list(groups))
            selected = groups[group_name][:8]
            cols = st.columns(min(4, max(1, len(selected))))
            for i, path in enumerate(selected):
                try:
                    with cols[i % len(cols)]:
                        st.image(Image.open(path), caption=path.name, width="stretch")
                except Exception:
                    pass
        else:
            st.info("No local image/tensor demo has been found yet. The repository's model results still work without it. Add one real `(4, 6, 120, 120)` NumPy tensor to `temporal_data/sample_tensors/` to show the same location at four observations.")

    with tab_features:
        st.markdown("#### From image to learned features")
        st.write("Each date is encoded as 2,048 numbers by the pretrained ResNet50. The four vectors are then supplied to a temporal classifier.")
        st.caption("Choose a location in the image tab first. The dashboard will look up its split and matching row in the saved feature cache automatically.")

        sample_path = DATA_DIR / "sample_tensors" / "sample_X.npy"
        metadata_path = DATA_DIR / "sample_tensors" / "sample_metadata.json"
        sample_metadata = []
        if metadata_path.exists():
            try:
                with metadata_path.open("r", encoding="utf-8-sig") as handle:
                    loaded_metadata = json.load(handle)
                if isinstance(loaded_metadata, list):
                    sample_metadata = loaded_metadata
            except (OSError, json.JSONDecodeError):
                pass

        # Use the selected sample location from the image tab's Streamlit state.
        chosen_item = None
        if sample_metadata:
            current_index = st.session_state.get("location_sample_X.npy", 0)
            if isinstance(current_index, int) and 0 <= current_index < len(sample_metadata):
                chosen_item = sample_metadata[current_index]

        if isinstance(chosen_item, dict) and chosen_item.get("spatial_id"):
            spatial_id = str(chosen_item["spatial_id"])
            mapped = find_temporal_feature_row(spatial_id)
            if mapped is not None:
                split, row, feature_sequence, target = mapped
                k1, k2, k3 = st.columns(3)
                k1.metric("Location", spatial_id)
                k2.metric("Dataset split", split.title())
                k3.metric("Feature sequence", "4 × 2,048")
                st.plotly_chart(render_feature_heatmap(feature_sequence), width="stretch")

                # A per-date distribution plot helps explain how activations vary over time.
                distribution = pd.DataFrame({
                    "Activation": feature_sequence.reshape(-1),
                    "Observation": np.repeat(["Date 1", "Date 2", "Date 3", "Date 4"], feature_sequence.shape[1]),
                })
                fig_hist = px.histogram(
                    distribution, x="Activation", color="Observation", barmode="overlay",
                    nbins=55, opacity=0.68,
                    color_discrete_map={"Date 1": "#16805d", "Date 2": "#3987c9", "Date 3": "#e39b32", "Date 4": "#9467bd"},
                    title="Feature-value distributions by date",
                    labels={"Activation": "Feature value", "count": "Number of features"},
                )
                fig_hist.update_layout(
                    template="plotly_white", height=340, bargap=0.08,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend_title_text="Observation", margin=dict(l=10, r=10, t=55, b=10),
                )
                st.plotly_chart(fig_hist, width="stretch")
                st.caption("The heatmap and histogram show learned numerical activations, not land-cover pixels. Feature indices do not have individual human-readable meanings.")

                with st.expander("Technical details and integrity checks"):
                    st.write(f"Cache: `temporal_data/full_feature_cache/temporal_features_{split}_seed2026.npz`")
                    st.write(f"Split-relative row: `{row}` · feature shape: `{feature_sequence.shape}` · target vector length: `{target.shape[0]}`")
                    st.write("The sample location is matched by `spatial_id` against `temporal_sequences.csv`, then indexed in that split's manifest order.")
                    vocab = read_csv("temporal_data/label_vocabulary.csv")
                    if vocab is not None:
                        st.write("Ground-truth target vector (1 = present label):")
                        target_df = pd.DataFrame({"Label index": np.arange(len(target)), "Present": target.astype(int)})
                        st.dataframe(target_df[target_df["Present"] == 1], width="stretch", hide_index=True)
            else:
                st.warning("Could not map this sample to its split-specific feature cache. Check that `temporal_sequences.csv` and `temporal_data/full_feature_cache/` are available locally.")
        else:
            st.info("Select a location in the **Four-date image sequence** tab to load its matching feature sequence here.")

        st.markdown("#### Pipeline")
        f1, f2, f3 = st.columns(3)
        with f1:
            st.markdown("**1 · Multispectral image**")
            st.code("6 bands × 120 × 120")
        with f2:
            st.markdown("**2 · ResNet50 encoder**")
            st.code("2,048 features / date")
        with f3:
            st.markdown("**3 · Temporal classifier**")
            st.code("4 dates × 2,048")

    with tab_setup:
        st.markdown("### Suggested demo folder layout")
        st.code("data/\n  demo_images/       # optional RGB previews, grouped by location/folder\n  demo_features/     # one small .npy feature sample\ntemporal_data/\n  sample_tensors/    # one four-date tensor: (4, 6, 120, 120)", language="text")
        st.markdown("### Important for a credible presentation")
        st.markdown("- Use four observations from the **same patch/location**, ordered by acquisition date.\n- Label the displayed dates only if you can map them to the real acquisition dates. Otherwise use Observation 1–4.\n- Use a true-colour composite from B04/B03/B02 for an intuitive preview; the model itself uses all six bands.\n- Show a feature heatmap as a representation of learned values, not as a reconstructed image or an explanation of what every feature means.\n- Keep large raw imagery and full feature caches out of Git; add only small, redistributable demo assets.")

elif page == "Available files":
    st.header("Available results and project files")
    st.write("The dashboard checks the following result files in the local repository:")
    expected = [
        "temporal_data/full_data_test_comparison.csv",
        "temporal_data/full_data_validation_comparison.csv",
        "temporal_data/improved_pilot_comparison.csv",
        "temporal_data/gru_experiment/test_metrics.json",
        "temporal_data/gru_experiment/training_history.csv",
        "temporal_data/attention_experiment/test_metrics.json",
        "temporal_data/attention_experiment/training_history.csv",
        "temporal_data/controlled_comparison/results.json",
        "temporal_data/controlled_comparison/uniform_average_history.csv",
        "temporal_data/controlled_comparison/temporal_attention_history.csv",
        "temporal_data/class_distribution.csv",
        "temporal_data/label_vocabulary.csv",
    ]
    for rel in expected:
        p = ROOT / rel
        st.write(("✅" if p.exists() else "⚠️") + f" `{rel}`" + (f" · {p.stat().st_size:,} bytes" if p.exists() else " · not found"))
    class_dist = read_csv("temporal_data/class_distribution.csv")
    if class_dist is not None:
        with st.expander("Class distribution chart and table", expanded=True):
            st.dataframe(class_dist, width="stretch", hide_index=True)
            label_col = find_column(class_dist, ["label", "class", "label_name", "category"])
            count_col = find_column(class_dist, ["count", "positive_count", "samples", "frequency"])
            if label_col and count_col:
                chart_data = class_dist.copy()
                chart_data[count_col] = pd.to_numeric(chart_data[count_col], errors="coerce")
                chart_data = chart_data.dropna(subset=[count_col]).sort_values(count_col, ascending=True)
                fig_dist = px.bar(
                    chart_data, x=count_col, y=label_col, orientation="h",
                    color=count_col, color_continuous_scale="Viridis",
                    title="Label frequency (class imbalance)",
                    labels={count_col: "Number of positive examples", label_col: "Land-cover label"},
                )
                fig_dist.update_layout(template="plotly_white", height=max(380, 24 * len(chart_data)),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig_dist, width="stretch")
    vocab = read_csv("temporal_data/label_vocabulary.csv")
    if vocab is not None:
        with st.expander("Label vocabulary"):
            st.dataframe(vocab, width="stretch", hide_index=True)

st.sidebar.divider()
st.sidebar.caption("Research demo · Local saved results · No training triggered")
