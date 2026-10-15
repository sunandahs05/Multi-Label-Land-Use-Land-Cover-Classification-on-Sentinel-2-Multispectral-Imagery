# Temporal LULC Explorer — dashboard setup

## 1. Add the dashboard
Copy `app.py` into the repository root (beside `README.md` and `requirements.txt`). Keep the existing `scripts/` and `temporal_data/` structure intact.

Append these two lines to the existing `requirements.txt` (do not replace the current dependencies):

```text
streamlit
plotly
```

Then run from the repository root in PowerShell:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address Streamlit prints, usually `http://localhost:8501`.

## 2. Temporal imagery demo
The **Temporal imagery & features** page can render a NumPy array containing one four-observation sequence. Put one or more small sample tensors in `temporal_data/sample_tensors/`:

- shape `(4, 6, 120, 120)` (channels-first), or `(4, 120, 120, 6)` (channels-last)
- band order: B02, B03, B04, B08, B11, B12
- observation order: chronological, for the same patch/location

The page renders a true-colour composite from B04/B03/B02 for each observation. These RGB previews are for human interpretation; the model uses all six bands. If actual acquisition dates are not mapped, the UI labels them Observation 1–4 rather than inventing dates.

## 3. Feature visualisation
The app searches for NumPy files whose names include `feature` under `temporal_data/` and `data/demo_features/`. Supported display shapes:

- `(N, 4, 2048)` — choose one sequence and view its four date-wise feature vectors as a heatmap.
- `(N, 2048)` — choose one vector and view feature values by dimension.

If the cache uses a different format, export only one example to `data/demo_features/example_temporal_features.npy`. Do not copy the full feature cache into Git. A feature heatmap shows latent numerical representations; it is not a reconstructed image and individual feature dimensions do not necessarily have human-readable meanings.

## 4. Optional standalone image files
Image files (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`, `.webp`) placed in `data/demo_images/`, `temporal_data/demo_images/`, or root `demo_images/` can also be previewed. Group related images in a folder named for a patch or example. For a presentation, ensure all four images really correspond to the same location and are ordered correctly.

## 5. What the dashboard does and does not do
- Reads saved JSON/CSV metrics and training histories.
- Shows controlled and earlier experiments separately to avoid implying that all settings were identical.
- Visualises optional local image tensors and saved feature arrays.
- Does **not** train models, load checkpoints for inference, or claim direct land-cover change detection.

Large imagery, datasets, checkpoints, and complete feature caches should remain local and ignored by Git. Include only small demo examples that are allowed to be redistributed.
