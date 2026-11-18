"""
visualize.py
================================================================================
Visualización e Inferencia con el Mejor Modelo Entrenado.

Genera los siguientes artefactos en un directorio de salida:
  1. Curva Precision-Recall y PR-AUC (pr_curve.png)
  2. Matriz de Confusión a nivel de Crop (confusion_matrix_crop.png)
  3. Matriz de Confusión a nivel de Objeto (confusion_matrix_object.png)
  4. Mejores predicciones positivas (best_positive/)
  5. Peores predicciones — falsos positivos y falsos negativos (worst_predictions/)
  6. Carpeta de imágenes clasificadas por categoría de la matriz de confusión
     (confusion_images/TP/, confusion_images/FP/, confusion_images/TN/, confusion_images/FN/)

Uso desde terminal:
  python training/visualize.py --region_id ChugChug
  python training/visualize.py --region_id Lluta --checkpoint_path training/checkpoints/best_convnext_tiny_Lluta.pth
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as T

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for script usage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("Visualize")

# Reutilizar la raíz del proyecto para importar módulos compartidos
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_project_path(path: Optional[str | Path]) -> Optional[Path]:
    """
    Resuelve una ruta relativa preferentemente respecto al CWD o a PROJECT_ROOT.
    Permite ejecutar el script sin importar si se llama desde la raíz del proyecto
    o desde la carpeta 'training/'.
    """
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    # Si existe relativo al directorio de ejecución actual
    if p.exists():
        return p.resolve()
    # Si existe relativo a la raíz del proyecto
    cand_root = PROJECT_ROOT / p
    if cand_root.exists():
        return cand_root.resolve()
    # Si se pasó relativo a training/
    cand_train = PROJECT_ROOT / "training" / p
    if cand_train.exists():
        return cand_train.resolve()
    # Fallback por defecto: anclar a PROJECT_ROOT
    return (PROJECT_ROOT / p).resolve()


# ==============================================================================
# CARGA DE METADATOS (replicado de train_eval_single_region.py)
# ==============================================================================
def load_and_filter_metadata(
    metadata_path: Optional[str | Path] = None,
    images_dir: Optional[str | Path] = None,
    target_region: str = "ChugChug",
) -> pd.DataFrame:
    """
    Carga el DataFrame de metadatos y filtra estrictamente por region_id == target_region.
    Estandariza columnas a: region_id, geoglyph_id, crop_id, label, image_path.
    """
    images_dir = _resolve_project_path(images_dir) if images_dir is not None else PROJECT_ROOT / "ML" / target_region
    metadata_path = _resolve_project_path(metadata_path) if metadata_path is not None else None

    df: Optional[pd.DataFrame] = None

    candidate_csv_paths = []
    if metadata_path is not None:
        candidate_csv_paths.append(metadata_path)
    candidate_csv_paths.extend([
        images_dir / "image_metadata.csv",
        PROJECT_ROOT / "ML" / target_region / "image_metadata.csv",
        PROJECT_ROOT / "ML" / "sites_metadata.csv",
    ])

    for csv_candidate in candidate_csv_paths:
        if csv_candidate.is_file():
            logger.info("Cargando metadatos desde: %s", csv_candidate)
            df = pd.read_csv(csv_candidate)
            break

    if df is None:
        logger.warning(
            "No se encontró un archivo CSV. Reconstruyendo metadatos desde %s", images_dir
        )
        if not images_dir.is_dir():
            raise FileNotFoundError(f"El directorio de imágenes no existe: {images_dir}")

        records = []
        for img_file in images_dir.glob("*.png"):
            stem = img_file.stem
            parts = stem.split("_")
            if len(parts) >= 3:
                site, geo_id, crop_id = parts[0], parts[1], parts[2]
            elif len(parts) == 2:
                site, geo_id, crop_id = parts[0], parts[1], "0"
            else:
                site, geo_id, crop_id = target_region, stem, "0"

            try:
                numeric_id = int(geo_id)
                label = 1 if numeric_id <= 124 else 0
            except ValueError:
                label = 1

            records.append({
                "region_id": site,
                "geoglyph_id": str(geo_id),
                "crop_id": str(crop_id),
                "label": int(label),
                "image_path": str(img_file),
            })
        df = pd.DataFrame(records)

    # Estandarización de columnas
    rename_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if col_upper in ("SITE_NAME", "REGION", "SITE", "REGION_ID"):
            rename_map[col] = "region_id"
        elif col_upper in ("GEO_ID", "GEOGLYPH_ID", "POLYGON_ID", "OBJECT_ID"):
            rename_map[col] = "geoglyph_id"
        elif col_upper in ("CROP_ID", "CROP_N", "CROP"):
            rename_map[col] = "crop_id"
        elif col_upper in ("DATA_LABEL", "LABEL", "TARGET", "CLASS"):
            rename_map[col] = "label"
        elif col_upper in ("IMAGE_PATH", "PATH", "FILEPATH", "IMG_PATH"):
            rename_map[col] = "image_path"
    df = df.rename(columns=rename_map)

    if "region_id" in df.columns:
        df = df[df["region_id"].astype(str).str.lower() == target_region.lower()].copy()
    else:
        df["region_id"] = target_region

    if len(df) == 0:
        raise ValueError(f"No se encontraron registros para region_id == '{target_region}'.")

    if "label" in df.columns:
        unique_labels = set(df["label"].unique())
        if 2 in unique_labels or "2" in unique_labels:
            df["label"] = df["label"].apply(lambda x: 1 if int(x) == 1 else 0)
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise KeyError("No se encontró la columna de etiquetas ('label' / 'DATA_LABEL').")

    df["geoglyph_id"] = df["geoglyph_id"].astype(str)
    if "crop_id" not in df.columns:
        df["crop_id"] = df.groupby("geoglyph_id").cumcount().astype(str)
    else:
        df["crop_id"] = df["crop_id"].astype(str)

    if "image_path" not in df.columns or not Path(str(df["image_path"].iloc[0])).exists():
        def _resolve_img_path(row: pd.Series) -> str:
            candidates = []
            id_val = str(row.get("ID", "")).strip()
            if id_val and id_val != "nan":
                candidates.append(images_dir / f"{id_val}.png")

            reg = str(row.get("region_id", ""))
            geo = str(row.get("geoglyph_id", ""))
            crop = str(row.get("crop_id", ""))

            candidates.append(images_dir / f"{reg}_{geo}_{crop}.png")
            if crop.isdigit():
                candidates.append(images_dir / f"{reg}_{geo}_{int(crop)}.png")
            if geo.isdigit():
                candidates.append(images_dir / f"{reg}_{int(geo)}_{crop}.png")
                if crop.isdigit():
                    candidates.append(images_dir / f"{reg}_{int(geo)}_{int(crop)}.png")

            for c in candidates:
                if c and Path(c).is_file():
                    return str(c)

            # Búsqueda por glob si candidatos no coincidieron
            matches = list(images_dir.glob(f"*_{geo}_{crop}.png"))
            if matches:
                return str(matches[0])
            if crop.isdigit():
                matches_int = list(images_dir.glob(f"*_{geo}_{int(crop)}.png"))
                if matches_int:
                    return str(matches_int[0])
            return str(candidates[0] if candidates else "")

        df["image_path"] = df.apply(_resolve_img_path, axis=1)

    existing_mask = df["image_path"].apply(lambda p: Path(p).is_file())
    missing_count = (~existing_mask).sum()
    if missing_count > 0:
        logger.warning("%d imágenes no encontradas en disco, descartadas.", missing_count)
        df = df[existing_mask].copy()

    df = df.reset_index(drop=True)
    logger.info(
        "Dataset: %d recortes (%d pos, %d neg) en %d entidades.",
        len(df), (df["label"] == 1).sum(), (df["label"] == 0).sum(),
        df["geoglyph_id"].nunique(),
    )
    return df


# ==============================================================================
# ENTITY-AWARE SPLIT (reproduce exactamente el split de entrenamiento)
# ==============================================================================
def entity_aware_train_val_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce el split entity-aware idéntico al de entrenamiento."""
    groups = df["geoglyph_id"].values
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_groups = set(train_df["geoglyph_id"].unique())
    val_groups = set(val_df["geoglyph_id"].unique())
    assert len(train_groups & val_groups) == 0, "¡FUGA DETECTADA!"

    logger.info("Split reproducido: Train=%d  Val=%d", len(train_df), len(val_df))
    return train_df, val_df


# ==============================================================================
# CARGA DEL MODELO DESDE CHECKPOINT
# ==============================================================================
def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Carga la arquitectura y pesos del mejor checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = checkpoint.get("args", {})
    model_name = saved_args.get("model_name", "convnext_tiny").lower()

    if "convnext" in model_name:
        model = tv_models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
    elif "resnet50" in model_name:
        model = tv_models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError(f"Modelo no soportado: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    best_auc = checkpoint.get("best_val_pr_auc", -1.0)
    epoch = checkpoint.get("epoch", 0)
    logger.info("Modelo cargado: epoch=%d, PR-AUC=%.4f, arch=%s", epoch, best_auc, model_name)
    return model, checkpoint


# ==============================================================================
# INFERENCIA SOBRE EL DATASET
# ==============================================================================
@torch.no_grad()
def run_inference(
    model: nn.Module,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Ejecuta inferencia sobre todas las imágenes del DataFrame y retorna probabilidades."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    all_probs = []
    n = len(df)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_tensors = []

        for idx in range(start, end):
            img_path = df.iloc[idx]["image_path"]
            try:
                with Image.open(img_path) as img:
                    img_rgb = img.convert("RGB")
            except Exception:
                img_rgb = Image.new("RGB", (224, 224), color=(0, 0, 0))
            batch_tensors.append(transform(img_rgb))

        batch = torch.stack(batch_tensors).to(device, non_blocking=True)
        logits = model(batch)
        probs = torch.sigmoid(logits).squeeze(-1).cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs.item()])
        all_probs.extend(probs.tolist())

    return np.array(all_probs)


# ==============================================================================
# VISUALIZACIONES
# ==============================================================================

def plot_pr_curve(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    output_path: Path,
    region_id: str,
) -> float:
    """Genera la curva Precision-Recall y retorna el PR-AUC."""
    pr_auc = average_precision_score(y_true, y_probs)
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall_vals, precision_vals, color="#2563EB", linewidth=2,
            label=f"PR-AUC = {pr_auc:.4f}")
    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color="#2563EB")
    ax.set_xlabel("Recall (Producer's Accuracy)", fontsize=12)
    ax.set_ylabel("Precision (User's Accuracy)", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {region_id}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=12)
    ax.set_xlim([0.0, 1.05])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR Curve guardada en: %s (AUC=%.4f)", output_path, pr_auc)
    return float(pr_auc)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
    labels: List[str] = None,
) -> np.ndarray:
    """Genera y guarda la visualización de la matriz de confusión."""
    if labels is None:
        labels = ["Negativo (Fondo)", "Positivo (Geoglifo)"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cantidad", fontsize=11)

    # Anotaciones con porcentajes
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100.0 * count / total if total > 0 else 0.0
            color = "white" if count > cm.max() / 2 else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=14, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Neg", "Pred Pos"], fontsize=11)
    ax.set_yticklabels(["Real Neg", "Real Pos"], fontsize=11)
    ax.set_xlabel("Predicción", fontsize=12)
    ax.set_ylabel("Verdad", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Anotaciones de métricas debajo de la gráfica
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fig.text(0.5, -0.02,
             f"Precision={precision:.3f}  |  Recall={recall:.3f}  |  F1={f1:.3f}",
             ha="center", fontsize=11, fontstyle="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix guardada en: %s", output_path)
    return cm


def save_best_and_worst_predictions(
    df: pd.DataFrame,
    y_probs: np.ndarray,
    output_dir: Path,
    n_images: int = 25,
) -> None:
    """
    Guarda grillas de las mejores y peores predicciones.
    - best_positive/: Positivos reales con mayor confianza (verdaderos positivos más seguros).
    - worst_predictions/: Falsos positivos y falsos negativos más confiantes (errores más graves).
    """
    df_viz = df.copy()
    df_viz["prob"] = y_probs

    threshold = 0.5

    # --- Mejores predicciones positivas (TP con alta confianza) ---
    tp_mask = (df_viz["label"] == 1) & (df_viz["prob"] >= threshold)
    best_positives = df_viz[tp_mask].nlargest(n_images, "prob")

    best_dir = output_dir / "best_positive"
    best_dir.mkdir(parents=True, exist_ok=True)

    if len(best_positives) > 0:
        _save_image_grid(
            best_positives, best_dir / "best_positive_grid.png",
            title="Mejores Predicciones Positivas (TP con alta confianza)",
            n_cols=5,
        )
        for i, (_, row) in enumerate(best_positives.iterrows()):
            src = Path(row["image_path"])
            if src.is_file():
                shutil.copy2(src, best_dir / f"rank{i+1:02d}_p{row['prob']:.3f}_{src.name}")

    # --- Peores predicciones: Falsos Positivos (negativos clasificados como positivos) ---
    fp_mask = (df_viz["label"] == 0) & (df_viz["prob"] >= threshold)
    worst_fp = df_viz[fp_mask].nlargest(n_images, "prob")

    worst_dir = output_dir / "worst_predictions"
    worst_dir.mkdir(parents=True, exist_ok=True)

    fp_dir = worst_dir / "false_positives"
    fp_dir.mkdir(parents=True, exist_ok=True)

    if len(worst_fp) > 0:
        _save_image_grid(
            worst_fp, fp_dir / "false_positives_grid.png",
            title="Peores Falsos Positivos (Fondo clasificado como Geoglifo)",
            n_cols=5, border_color="red",
        )
        for i, (_, row) in enumerate(worst_fp.iterrows()):
            src = Path(row["image_path"])
            if src.is_file():
                shutil.copy2(src, fp_dir / f"rank{i+1:02d}_p{row['prob']:.3f}_{src.name}")

    # --- Peores predicciones: Falsos Negativos (positivos clasificados como negativos) ---
    fn_mask = (df_viz["label"] == 1) & (df_viz["prob"] < threshold)
    worst_fn = df_viz[fn_mask].nsmallest(n_images, "prob")

    fn_dir = worst_dir / "false_negatives"
    fn_dir.mkdir(parents=True, exist_ok=True)

    if len(worst_fn) > 0:
        _save_image_grid(
            worst_fn, fn_dir / "false_negatives_grid.png",
            title="Peores Falsos Negativos (Geoglifo clasificado como Fondo)",
            n_cols=5, border_color="orange",
        )
        for i, (_, row) in enumerate(worst_fn.iterrows()):
            src = Path(row["image_path"])
            if src.is_file():
                shutil.copy2(src, fn_dir / f"rank{i+1:02d}_p{row['prob']:.3f}_{src.name}")

    logger.info("Mejores/peores predicciones guardadas en: %s", output_dir)


def save_confusion_image_folders(
    df: pd.DataFrame,
    y_probs: np.ndarray,
    output_dir: Path,
    threshold: float = 0.5,
) -> Dict[str, int]:
    """
    Copia cada imagen del set de evaluación a la carpeta correspondiente
    de la matriz de confusión: TP/, FP/, TN/, FN/.
    Los archivos se renombran con la probabilidad para facilitar la inspección.
    """
    confusion_dir = output_dir / "confusion_images"
    categories = {"TP": [], "FP": [], "TN": [], "FN": []}

    y_pred = (y_probs >= threshold).astype(int)
    y_true = df["label"].values

    for idx in range(len(df)):
        true_label = int(y_true[idx])
        pred_label = int(y_pred[idx])

        if true_label == 1 and pred_label == 1:
            cat = "TP"
        elif true_label == 0 and pred_label == 1:
            cat = "FP"
        elif true_label == 0 and pred_label == 0:
            cat = "TN"
        else:
            cat = "FN"
        categories[cat].append(idx)

    counts = {}
    for cat, indices in categories.items():
        cat_dir = confusion_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        counts[cat] = len(indices)

        for idx in indices:
            row = df.iloc[idx]
            prob = y_probs[idx]
            src = Path(row["image_path"])
            if src.is_file():
                dst_name = f"p{prob:.4f}_{src.name}"
                shutil.copy2(src, cat_dir / dst_name)

    logger.info(
        "Imágenes de confusión organizadas: TP=%d, FP=%d, TN=%d, FN=%d",
        counts["TP"], counts["FP"], counts["TN"], counts["FN"],
    )
    return counts


def _save_image_grid(
    df_subset: pd.DataFrame,
    output_path: Path,
    title: str,
    n_cols: int = 5,
    border_color: Optional[str] = None,
) -> None:
    """Genera una grilla de imágenes con sus probabilidades anotadas."""
    n = len(df_subset)
    if n == 0:
        return
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    for i, (_, row) in enumerate(df_subset.iterrows()):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        try:
            img = Image.open(row["image_path"]).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "ERROR", ha="center", va="center", transform=ax.transAxes)

        prob = row["prob"]
        label = "GEO" if row["label"] == 1 else "NEG"
        ax.set_title(f"{label} | p={prob:.3f}", fontsize=9, fontweight="bold")

        if border_color:
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
                spine.set_visible(True)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==============================================================================
# FUNCIÓN PRINCIPAL
# ==============================================================================
def run_visualization(args: argparse.Namespace) -> None:
    """Orquesta la carga del modelo, inferencia y generación de todas las visualizaciones."""
    # Reproducibilidad
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("Dispositivo: %s", device)

    # 1. Resolver checkpoint path
    checkpoint_path = _resolve_project_path(args.checkpoint_path)
    if checkpoint_path is None or not checkpoint_path.is_file():
        # Buscar en ubicaciones comunes
        alt_paths = [
            PROJECT_ROOT / "training" / "checkpoints" / Path(args.checkpoint_path).name,
            PROJECT_ROOT / "training" / "checkpoints" / f"best_convnext_tiny_{args.region_id}.pth",
            PROJECT_ROOT / "training" / "checkpoints" / f"best_resnet50_{args.region_id}.pth",
            PROJECT_ROOT / "checkpoints" / f"best_convnext_tiny_{args.region_id}.pth",
        ]
        found = False
        for alt in alt_paths:
            if alt.is_file():
                checkpoint_path = alt
                found = True
                break
        if not found:
            raise FileNotFoundError(
                f"No se encontró el checkpoint: {args.checkpoint_path}\n"
                f"Rutas buscadas: {[str(p) for p in alt_paths]}"
            )
    logger.info("Checkpoint: %s", checkpoint_path)

    # 2. Cargar modelo
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)

    # 3. Cargar datos de la región
    df = load_and_filter_metadata(
        metadata_path=args.metadata_csv,
        images_dir=args.images_dir,
        target_region=args.region_id,
    )

    # 4. División de evaluación o set completo
    if args.eval_full:
        eval_df = df
        eval_label = "full_dataset"
        logger.info("Evaluando sobre el dataset completo (%d muestras).", len(eval_df))
    else:
        saved_args = checkpoint.get("args", {})
        train_ratio = saved_args.get("train_ratio", args.train_ratio)
        seed = saved_args.get("seed", args.seed)
        train_df, val_df = entity_aware_train_val_split(df, train_size=train_ratio, random_state=seed)
        eval_df = val_df
        eval_label = "validation_set"
        logger.info("Evaluando sobre el set de validación (%d muestras).", len(eval_df))

    # 5. Inferencia
    logger.info("Ejecutando inferencia sobre %d imágenes...", len(eval_df))
    y_probs = run_inference(model, eval_df, device, batch_size=args.batch_size)
    y_true = eval_df["label"].values
    y_pred = (y_probs >= args.threshold).astype(int)

    # 6. Crear directorio de salida
    output_base = _resolve_project_path(args.output_dir)
    output_dir = output_base / args.region_id
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Directorio de salida: %s", output_dir)

    # 7. Curva PR
    pr_auc = plot_pr_curve(y_true, y_probs, output_dir / "pr_curve.png", args.region_id)

    # 8. Matriz de confusión a nivel de crop
    cm_crop = plot_confusion_matrix(
        y_true, y_pred, output_dir / "confusion_matrix_crop.png",
        title=f"Matriz de Confusión — Crop Level — {args.region_id}",
    )

    # 9. Matriz de confusión a nivel de objeto
    obj_df = eval_df.copy()
    obj_df["prob"] = y_probs
    obj_results = []
    for geo_id, group in obj_df.groupby("geoglyph_id"):
        true_label = int(group["label"].max())
        agg_prob = float(np.max(group["prob"].values))
        obj_results.append({
            "geoglyph_id": geo_id,
            "true_label": true_label,
            "agg_prob": agg_prob,
            "pred_label": int(agg_prob >= args.object_threshold),
        })
    df_obj = pd.DataFrame(obj_results)

    if len(df_obj) > 0:
        plot_confusion_matrix(
            df_obj["true_label"].values,
            df_obj["pred_label"].values,
            output_dir / "confusion_matrix_object.png",
            title=f"Matriz de Confusión — Object Level (p≥{args.object_threshold}) — {args.region_id}",
        )

    # 10. Best / Worst predictions
    save_best_and_worst_predictions(
        eval_df, y_probs, output_dir, n_images=args.n_images,
    )

    # 11. Carpeta de imágenes por categoría de confusión
    counts = save_confusion_image_folders(
        eval_df, y_probs, output_dir, threshold=args.threshold,
    )

    # 12. Guardar resumen JSON
    tn, fp, fn, tp = cm_crop.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    summary = {
        "region_id": args.region_id,
        "eval_set": eval_label,
        "checkpoint": str(checkpoint_path),
        "best_epoch": checkpoint.get("epoch", "N/A"),
        "num_samples": len(eval_df),
        "threshold": args.threshold,
        "object_threshold": args.object_threshold,
        "crop_metrics": {
            "pr_auc": pr_auc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion_matrix": {"TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)},
        },
        "object_metrics": {
            "num_objects": len(df_obj),
            "num_positive_objects": int((df_obj["true_label"] == 1).sum()),
            "num_negative_objects": int((df_obj["true_label"] == 0).sum()),
        } if len(df_obj) > 0 else {},
        "confusion_image_counts": counts,
        "output_dir": str(output_dir),
    }

    summary_path = output_dir / "visualization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    # Final report
    print("\n" + "=" * 60)
    print(f"  VISUALIZACIÓN COMPLETADA — {args.region_id}")
    print("=" * 60)
    print(f"  Checkpoint:     {checkpoint_path.name}")
    print(f"  Epoch:          {checkpoint.get('epoch', 'N/A')}")
    print(f"  Eval set:       {eval_label} ({len(eval_df)} muestras)")
    print(f"  PR-AUC:         {pr_auc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    print("-" * 60)
    print(f"  Confusion (crop):  TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print("-" * 60)
    print(f"  Archivos generados en: {output_dir}")
    print(f"    ├── pr_curve.png")
    print(f"    ├── confusion_matrix_crop.png")
    print(f"    ├── confusion_matrix_object.png")
    print(f"    ├── best_positive/")
    print(f"    ├── worst_predictions/")
    print(f"    │   ├── false_positives/")
    print(f"    │   └── false_negatives/")
    print(f"    ├── confusion_images/")
    print(f"    │   ├── TP/ ({counts.get('TP', 0)} images)")
    print(f"    │   ├── FP/ ({counts.get('FP', 0)} images)")
    print(f"    │   ├── TN/ ({counts.get('TN', 0)} images)")
    print(f"    │   └── FN/ ({counts.get('FN', 0)} images)")
    print(f"    └── visualization_summary.json")
    print("=" * 60 + "\n")


# ==============================================================================
# CLI
# ==============================================================================
def _load_config_defaults(config_path: str | Path) -> Dict[str, Any]:
    """
    Carga un archivo JSON de configuración de evaluación y retorna un dict
    con los parámetros. Las claves del JSON mapean 1:1 a los argumentos CLI.
    """
    resolved = _resolve_project_path(config_path)
    if resolved is None or not resolved.is_file():
        cand = Path(config_path)
        if cand.is_file():
            resolved = cand.resolve()
        else:
            raise FileNotFoundError(f"Config no encontrado: {config_path}")
    with open(resolved, "r", encoding="utf-8") as f:
        config = json.load(f)
    logger.info("Config cargado desde: %s", resolved)
    return config


def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualización e Inferencia con el mejor modelo entrenado."
    )
    # Config JSON (se aplica primero, CLI override después)
    parser.add_argument("--config", type=str, default=None,
                        help="Ruta a un archivo JSON de configuración de evaluación. "
                             "Los valores del JSON se usan como defaults; los argumentos "
                             "CLI explícitos los sobreescriben.")

    # Rutas
    parser.add_argument("--region_id", type=str, default=None,
                        help="ID de la región a evaluar (default: 'ChugChug')")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="Ruta al checkpoint .pth del modelo")
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Ruta al CSV de metadatos (auto-detectado si no se provee)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Directorio con las imágenes PNG de la región")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directorio base para guardar las visualizaciones")

    # Evaluación
    parser.add_argument("--eval_full", action="store_true", default=None,
                        help="Evaluar sobre el dataset completo en vez del set de validación")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Umbral de clasificación a nivel de crop (default: 0.5)")
    parser.add_argument("--object_threshold", type=float, default=None,
                        help="Umbral para detección a nivel de objeto (default: 0.65)")
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="Proporción de train para reproducir el split (default: 0.8)")
    parser.add_argument("--n_images", type=int, default=None,
                        help="Número de imágenes para best/worst grids (default: 25)")

    # Ejecución
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size para inferencia (default: 64)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla para reproducibilidad (default: 42)")
    parser.add_argument("--no_cuda", action="store_true", default=None,
                        help="Forzar CPU")

    return parser


# Defaults que se aplican cuando ni el config ni el CLI especifican un valor.
_HARDCODED_DEFAULTS = {
    "region_id": "ChugChug",
    "checkpoint_path": "training/checkpoints/best_convnext_tiny_ChugChug.pth",
    "metadata_csv": None,
    "images_dir": None,
    "output_dir": "training/visualizations",
    "eval_full": False,
    "threshold": 0.5,
    "object_threshold": 0.65,
    "train_ratio": 0.8,
    "n_images": 25,
    "batch_size": 64,
    "seed": 42,
    "no_cuda": False,
}


def _resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Resuelve la prioridad: CLI explícito > config JSON > hardcoded defaults.
    """
    # 1. Cargar config JSON si se proporcionó
    config = {}
    if args.config is not None:
        config = _load_config_defaults(args.config)

    # 2. Para cada parámetro, aplicar prioridad
    for key, hardcoded in _HARDCODED_DEFAULTS.items():
        cli_value = getattr(args, key, None)
        if cli_value is not None:
            # CLI explícito: mantener
            continue
        elif key in config:
            # Config JSON
            setattr(args, key, config[key])
        else:
            # Hardcoded default
            setattr(args, key, hardcoded)

    return args


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    args = _resolve_args(args)
    run_visualization(args)

