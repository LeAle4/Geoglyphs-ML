"""
train_eval_single_region.py
================================================================================
Entrenamiento y Evaluación Modular en PyTorch para Clasificación de Recortes
de Geoglifos (224x224 px) en la Región 'ChugChug'.

Características principales:
1. Carga y Filtrado de Datos (region_id == 'ChugChug').
2. Entity-Aware Train/Val Split (80/20) agrupado estrictamente por geoglyph_id
   para evitar fuga de datos por autocorrelación espacial o memorización.
3. SingleRegionHierarchicalCappedSampler: Sampler jerárquico que agrupa recortes
   por geoglyph_id y limita N_c <= 15 recortes por geoglifo.
4. Modelos: ConvNeXt-Tiny o ResNet-50 preentrenados en ImageNet.
   Pérdidas: FocalLoss(gamma=2.0) o BCEWithLogitsLoss para desbalance extremo.
5. Accuracy Metrics Framework para Teledetección:
   - PROHIBIDO: Cohen's Kappa y Overall Accuracy como métrica principal.
   - Nivel Crop: Precision (User's Accuracy, UA), Recall (Producer's Accuracy, PA),
     y PR-AUC (Average Precision) como métrica escalar rectora.
   - Matriz de Confusión Ajustada por Población (Olofsson et al., 2014).
   - Nivel Objeto: F1-score Agregado a nivel de Geoglifo (umbral p >= 0.65).
6. Bucle de entrenamiento y registro con guardado del mejor checkpoint por PR-AUC.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
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


# ==============================================================================
# CONFIGURACIÓN DE LOGGING
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("GeoglyphPipeline")


# ==============================================================================
# 1. CARGA, FILTRADO Y VALIDACIÓN DE METADATOS
# ==============================================================================
def load_and_filter_metadata(
    metadata_path: Optional[str | Path] = None,
    images_dir: Optional[str | Path] = None,
    target_region: str = "ChugChug",
) -> pd.DataFrame:
    """
    Carga el DataFrame de metadatos y filtra estrictamente por region_id == target_region.
    
    Estandariza los nombres de columnas a:
      - 'region_id': identificador de la región/sitio de estudio.
      - 'geoglyph_id': identificador único de la entidad física real (objeto).
      - 'crop_id': identificador ordinal del recorte dentro de la entidad.
      - 'label': etiqueta binaria (1: geoglifo positivo, 0: fondo/negativo).
      - 'image_path': ruta absoluta o relativa válida al archivo PNG (224x224).
    """
    project_root = Path(__file__).resolve().parent
    if images_dir is None:
        images_dir = project_root / "ML" / target_region
    else:
        images_dir = Path(images_dir)

    df: Optional[pd.DataFrame] = None

    # Intentar cargar desde el archivo CSV proporcionado o rutas por defecto
    candidate_csv_paths = []
    if metadata_path is not None:
        candidate_csv_paths.append(Path(metadata_path))
    candidate_csv_paths.extend([
        images_dir / "image_metadata.csv",
        project_root / "ML" / target_region / "image_metadata.csv",
        project_root / "ML" / "sites_metadata.csv",
    ])

    for csv_candidate in candidate_csv_paths:
        if csv_candidate.is_file():
            logger.info("Cargando metadatos desde: %s", csv_candidate)
            df = pd.read_csv(csv_candidate)
            break

    # Si no se encuentra un CSV preexistente, reconstruir dinámicamente desde el directorio de recortes
    if df is None:
        logger.warning(
            "No se encontró un archivo CSV en las rutas esperadas. "
            "Reconstruyendo metadatos a partir de los archivos en %s", images_dir
        )
        if not images_dir.is_dir():
            raise FileNotFoundError(f"El directorio de imágenes no existe: {images_dir}")
        
        records = []
        for img_file in images_dir.glob("*.png"):
            stem = img_file.stem
            parts = stem.split("_")
            # Formato estándar esperado: {SiteName}_{GeoID}_{CropID}
            if len(parts) >= 3:
                site = parts[0]
                geo_id = parts[1]
                crop_id = parts[2]
            elif len(parts) == 2:
                site = parts[0]
                geo_id = parts[1]
                crop_id = "0"
            else:
                site = target_region
                geo_id = stem
                crop_id = "0"
            
            # En la convención del proyecto, IDs <= 124 son geoglifos, > 124 son negativos
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

    # Estandarización de nombres de columnas
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

    # 1. Filtrar muestras estrictamente para region_id == target_region
    if "region_id" in df.columns:
        initial_count = len(df)
        df = df[df["region_id"].astype(str).str.lower() == target_region.lower()].copy()
        logger.info(
            "Filtrado por region_id == '%s': %d/%d muestras conservadas.",
            target_region, len(df), initial_count
        )
    else:
        logger.warning("Columna 'region_id' no encontrada. Asumiendo que todas las muestras son de %s.", target_region)
        df["region_id"] = target_region

    if len(df) == 0:
        raise ValueError(f"No se encontraron registros para region_id == '{target_region}'.")

    # Mapeo de etiquetas: 1 (geoglifo) y 2 (ground/fondo) -> 1 y 0
    if "label" in df.columns:
        unique_labels = set(df["label"].unique())
        if 2 in unique_labels or "2" in unique_labels:
            df["label"] = df["label"].apply(lambda x: 1 if int(x) == 1 else 0)
        else:
            df["label"] = df["label"].astype(int)
    else:
        raise KeyError("No se encontró la columna de etiquetas ('label' / 'DATA_LABEL').")

    # Asegurar geoglyph_id como string consistente
    df["geoglyph_id"] = df["geoglyph_id"].astype(str)
    if "crop_id" not in df.columns:
        df["crop_id"] = df.groupby("geoglyph_id").cumcount().astype(str)
    else:
        df["crop_id"] = df["crop_id"].astype(str)

    # Resolver rutas de imágenes si no existen
    if "image_path" not in df.columns or not Path(str(df["image_path"].iloc[0])).exists():
        def _resolve_img_path(row: pd.Series) -> str:
            candidates = [
                images_dir / f"{row['region_id']}_{row['geoglyph_id']}_{row['crop_id']}.png",
                images_dir / f"{row['region_id']}_{row['geoglyph_id']}_{int(row['crop_id'])}.png" if row['crop_id'].isdigit() else "",
                images_dir / f"{row.get('ID', '')}.png",
            ]
            for c in candidates:
                if c and Path(c).is_file():
                    return str(c)
            # Búsqueda difusa por geoglyph_id y crop_id
            matches = list(images_dir.glob(f"*_{row['geoglyph_id']}_{row['crop_id']}.png"))
            if matches:
                return str(matches[0])
            return str(candidates[0])

        df["image_path"] = df.apply(_resolve_img_path, axis=1)

    # Validar existencia de archivos
    existing_mask = df["image_path"].apply(lambda p: Path(p).is_file())
    missing_count = (~existing_mask).sum()
    if missing_count > 0:
        logger.warning("%d imágenes no se encontraron en disco y fueron descartadas.", missing_count)
        df = df[existing_mask].copy()

    df = df.reset_index(drop=True)
    n_pos = (df["label"] == 1).sum()
    n_neg = (df["label"] == 0).sum()
    n_entities = df["geoglyph_id"].nunique()

    logger.info(
        "Dataset cargado exitosamente: %d recortes totales (%d positivos, %d negativos) "
        "distribuidos en %d entidades únicas ('geoglyph_id').",
        len(df), n_pos, n_neg, n_entities
    )
    return df


# ==============================================================================
# 2. DIVISIÓN DEL DATASET POR ENTIDAD (ENTITY-AWARE SPLIT)
# ==============================================================================
def entity_aware_train_val_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Agrupa los datos por `geoglyph_id` para realizar un split 80% train / 20% val.
    
    Garantía estricta:
    Todos los recortes pertenecientes al mismo objeto físico (`geoglyph_id`) quedan
    estrictamente en un solo conjunto para evitar la fuga de datos por memorización
    de objetos y autocorrelación espacial.
    """
    # Determinamos la etiqueta a nivel de objeto para verificar balance
    obj_meta = df.groupby("geoglyph_id")["label"].max()
    groups = df["geoglyph_id"].values
    
    splitter = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, val_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    # Verificación de aislamiento estricto
    train_groups = set(train_df["geoglyph_id"].unique())
    val_groups = set(val_df["geoglyph_id"].unique())
    overlap = train_groups.intersection(val_groups)
    assert len(overlap) == 0, f"¡FUGA DETECTADA! Entidades compartidas: {overlap}"

    logger.info("=== Entity-Aware Split Completado ===")
    logger.info(
        "Train: %d recortes (%d positivos, %d negativos) de %d geoglifos únicos.",
        len(train_df), (train_df["label"] == 1).sum(), (train_df["label"] == 0).sum(), len(train_groups)
    )
    logger.info(
        "Val:   %d recortes (%d positivos, %d negativos) de %d geoglifos únicos.",
        len(val_df), (val_df["label"] == 1).sum(), (val_df["label"] == 0).sum(), len(val_groups)
    )
    return train_df, val_df


# ==============================================================================
# 3. MUESTREO JERÁRQUICO LIMITADO (SingleRegionHierarchicalCappedSampler)
# ==============================================================================
class SingleRegionHierarchicalCappedSampler(Sampler[int]):
    """
    Sampler jerárquico personalizado en PyTorch que agrupa índices por `geoglyph_id`
    y limita el número máximo de recortes a N_c <= max_crops_per_geo por geoglifo
    (vía submuestreo espacial uniforme en índice o muestreo aleatorio uniforme).
    
    Propósito en teledetección:
    Previene que geoglifos masivos con docenas de recortes solapados dominen el
    cálculo del gradiente y el aprendizaje de la red neuronal.
    """
    def __init__(
        self,
        geoglyph_ids: Sequence[str],
        max_crops_per_geo: int = 15,
        shuffle: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.max_crops_per_geo = max_crops_per_geo
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Agrupar índices por entidad geoglyph_id
        self.geo_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, geo_id in enumerate(geoglyph_ids):
            self.geo_to_indices[str(geo_id)].append(idx)

        # Precalcular longitud esperada por época
        self._length = sum(min(len(indices), self.max_crops_per_geo) for indices in self.geo_to_indices.values())
        logger.info(
            "HierarchicalCappedSampler inicializado: %d entidades, "
            "tope N_c=%d, total muestras/época=%d (de %d totales).",
            len(self.geo_to_indices), self.max_crops_per_geo, self._length, len(geoglyph_ids)
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        sampled_indices: List[int] = []

        for geo_id, indices in self.geo_to_indices.items():
            k = len(indices)
            if k <= self.max_crops_per_geo:
                sampled_indices.extend(indices)
            else:
                # Submuestreo uniforme regular sobre el recorrido espacial/ordinal de recortes
                # Para entrenamiento, agregamos variabilidad estocástica manteniendo equidistancia
                if self.shuffle:
                    # Muestreo uniforme aleatorio sin reemplazo
                    chosen = rng.choice(indices, size=self.max_crops_per_geo, replace=False).tolist()
                else:
                    # Submuestreo determinista equidistante espacial
                    step_indices = np.linspace(0, k - 1, self.max_crops_per_geo, dtype=int)
                    chosen = [indices[i] for i in step_indices]
                sampled_indices.extend(chosen)

        if self.shuffle:
            rng.shuffle(sampled_indices)

        return iter(sampled_indices)

    def __len__(self) -> int:
        return self._length


# ==============================================================================
# PYTORCH DATASET Y TRANSFORMACIONES
# ==============================================================================
class GeoglyphCropDataset(Dataset):
    """Dataset para recortes 224x224 px con transformaciones geoespaciales."""
    def __init__(self, df: pd.DataFrame, transform: Optional[T.Compose] = None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
        except Exception as e:
            logger.error("Error al abrir imagen: %s (%s)", img_path, e)
            img_rgb = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform is not None:
            tensor_img = self.transform(img_rgb)
        else:
            tensor_img = T.ToTensor()(img_rgb)

        return {
            "image": tensor_img,
            "label": torch.tensor(row["label"], dtype=torch.float32),
            "geoglyph_id": str(row["geoglyph_id"]),
            "crop_id": str(row["crop_id"]),
            "index": idx,
        }


def get_transforms() -> Tuple[T.Compose, T.Compose]:
    """
    Transformaciones invariantes a rotación ortogonal comunes en teledetección satelital / UAV.
    """
    # Estadísticas estándar de ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 360)),  # Imágenes aéreas son isotrópicas
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    val_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_transform


# ==============================================================================
# 4. MODELO Y FUNCIÓN DE PÉRDIDA
# ==============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss para Clasificación Binaria:
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Diseñada para contrarrestar el desbalance masivo de clases en paisajes áridos,
    focalizando el gradiente en ejemplos difíciles y mitigando la dominancia del fondo.
    """
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = targets * probs + (1.0 - targets) * (1.0 - probs)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss = modulating_factor * bce_loss

        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def build_model(model_name: str = "convnext_tiny", pretrained: bool = True) -> nn.Module:
    """
    Construye el clasificador binario en PyTorch adaptando la cabeza para 1 logit de salida.
    """
    model_name_clean = model_name.lower().replace("-", "_")

    if "convnext" in model_name_clean:
        weights = tv_models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = tv_models.convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, 1)
        logger.info("Modelo ConvNeXt-Tiny cargado (Pretrained=%s, in_features=%d).", pretrained, in_features)

    elif "resnet50" in model_name_clean:
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
        logger.info("Modelo ResNet-50 cargado (Pretrained=%s, in_features=%d).", pretrained, in_features)

    else:
        raise ValueError(f"Modelo no soportado: '{model_name}'. Elige 'convnext_tiny' o 'resnet50'.")

    return model


# ==============================================================================
# 5. ACCURACY METRICS FRAMEWORK (TELEDETECCIÓN & EVALUACIÓN GEOESPACIAL)
# ==============================================================================
class GeospatialAccuracyFramework:
    """
    Framework de métricas de precisión para Teledetección y Detección de Geoglifos.
    
    Estándares estrictos implementados:
    1. MÉTRICAS PROHIBIDAS:
       - Cohen's Kappa desactivado explícitamente debido a la conocida 'paradoja de Kappa'
         y su sesgo invalidante ante prevalencias asimétricas (Foody 2020, Pontius & Millones 2011).
       - Overall Accuracy prohibida como criterio de optimización o métrica principal.
       
    2. EVALUACIÓN A NIVEL DE CROP (Patch-Level):
       - User's Accuracy (Precision): TP / (TP + FP)
       - Producer's Accuracy (Recall): TP / (TP + FN)
       - Precision-Recall AUC (PR-AUC / Average Precision): Métrica escalar rectora.
       - Matriz de Confusión Ajustada por Población (Olofsson et al., 2014) utilizando
         la prevalencia real del paisaje.
         
    3. EVALUACIÓN AGREGADA A NIVEL DE OBJETO (Geoglyph-Level F1-Score):
       - Agrupa las probabilidades de los recortes pertenecientes a un mismo geoglyph_id.
       - Geoglifo detectado si max(p_crops) >= object_threshold (default 0.65).
       - F1-Score agregado sobre entidades físicas reales.
    """
    def __init__(
        self,
        object_threshold: float = 0.65,
        landscape_positive_prevalence: float = 0.05,
    ):
        self.object_threshold = object_threshold
        self.landscape_positive_prevalence = landscape_positive_prevalence

    def compute_crop_metrics(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Cálculo riguroso de métricas a nivel de recorte individual."""
        y_pred = (y_probs >= threshold).astype(int)

        # Métrica rectora: PR-AUC (Average Precision)
        pr_auc = float(average_precision_score(y_true, y_probs))

        # Precision (User's Accuracy, UA) y Recall (Producer's Accuracy, PA)
        precision_ua = float(precision_score(y_true, y_pred, zero_division=0))
        recall_pa = float(recall_score(y_true, y_pred, zero_division=0))
        f1_crop = float(2 * (precision_ua * recall_pa) / (precision_ua + recall_pa)) if (precision_ua + recall_pa) > 0 else 0.0

        # Matriz de confusión muestral tradicional
        cm_sample = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm_sample.ravel()

        # Matriz de confusión ajustada por población (Olofsson et al., 2014)
        # W1 = prevalencia real de geoglifos en el paisaje, W0 = 1 - W1
        W1 = self.landscape_positive_prevalence
        W0 = 1.0 - W1

        # Estimación de proporciones de área p_ij ajustadas por estratos de clase
        # Si la muestra fue generada por muestreo estratificado o con ratio de negativos artificial
        n_pos_samples = max(tp + fn, 1)
        n_neg_samples = max(tn + fp, 1)

        # Matriz de proporciones poblacionales estimadas
        p_11 = W1 * (tp / n_pos_samples)
        p_10 = W1 * (fn / n_pos_samples)
        p_01 = W0 * (fp / n_neg_samples)
        p_00 = W0 * (tn / n_neg_samples)

        pop_adjusted_cm = np.array([[p_00, p_01], [p_10, p_11]])
        pop_adjusted_ua = p_11 / (p_11 + p_01) if (p_11 + p_01) > 0 else 0.0
        pop_adjusted_pa = p_11 / (p_11 + p_10) if (p_11 + p_10) > 0 else 0.0

        return {
            "pr_auc": pr_auc,
            "precision_ua": precision_ua,
            "recall_pa": recall_pa,
            "f1_crop": f1_crop,
            "confusion_matrix_sample": {
                "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
            },
            "population_adjusted": {
                "W_positive": W1,
                "adjusted_matrix_proportions": pop_adjusted_cm.tolist(),
                "pop_adjusted_ua": float(pop_adjusted_ua),
                "pop_adjusted_pa": float(pop_adjusted_pa),
            },
        }

    def compute_object_level_f1(
        self,
        df_val: pd.DataFrame,
        y_probs: np.ndarray,
        aggregation: str = "max",
    ) -> Dict[str, Any]:
        """
        Reagrupa las predicciones por `geoglyph_id` y evalúa el F1 a nivel de entidad física.
        """
        df_temp = df_val.copy()
        df_temp["prob"] = y_probs

        obj_results = []
        for geo_id, group in df_temp.groupby("geoglyph_id"):
            true_label = int(group["label"].max())
            probs = group["prob"].values
            
            if aggregation == "max":
                agg_prob = float(np.max(probs))
            elif aggregation == "mean":
                agg_prob = float(np.mean(probs))
            else:
                agg_prob = float(np.max(probs))

            # Detección exitosa si la probabilidad supera el umbral (p >= 0.65)
            pred_detected = int(agg_prob >= self.object_threshold)
            obj_results.append({
                "geoglyph_id": geo_id,
                "true_label": true_label,
                "agg_prob": agg_prob,
                "pred_label": pred_detected,
                "num_crops": len(probs),
            })

        df_obj = pd.DataFrame(obj_results)
        y_true_obj = df_obj["true_label"].values
        y_pred_obj = df_obj["pred_label"].values

        obj_precision = float(precision_score(y_true_obj, y_pred_obj, zero_division=0))
        obj_recall = float(recall_score(y_true_obj, y_pred_obj, zero_division=0))
        obj_f1 = float(2 * (obj_precision * obj_recall) / (obj_precision + obj_recall)) if (obj_precision + obj_recall) > 0 else 0.0

        cm_obj = confusion_matrix(y_true_obj, y_pred_obj, labels=[0, 1])
        tn_o, fp_o, fn_o, tp_o = cm_obj.ravel()

        return {
            "object_f1": obj_f1,
            "object_precision": obj_precision,
            "object_recall": obj_recall,
            "num_objects": len(df_obj),
            "num_positive_objects": int((y_true_obj == 1).sum()),
            "num_negative_objects": int((y_true_obj == 0).sum()),
            "confusion_matrix_object": {
                "TN": int(tn_o), "FP": int(fp_o), "FN": int(fn_o), "TP": int(tp_o)
            },
        }


# ==============================================================================
# 6. BUCLE DE ENTRENAMIENTO Y VALIDACIÓN
# ==============================================================================
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Ejecuta una época de entrenamiento."""
    model.train()
    running_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

    return running_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    val_df: pd.DataFrame,
    criterion: nn.Module,
    framework: GeospatialAccuracyFramework,
    device: torch.device,
) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
    """Ejecuta la validación calculando métricas a nivel de crop y de objeto."""
    model.eval()
    running_loss = 0.0
    total_samples = 0

    all_probs = []
    all_targets = []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True).unsqueeze(1)

        logits = model(images)
        loss = criterion(logits, labels)
        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size

        all_probs.extend(probs.squeeze(1).cpu().numpy().tolist())
        all_targets.extend(labels.squeeze(1).cpu().numpy().tolist())

    val_loss = running_loss / max(total_samples, 1)
    y_probs = np.array(all_probs)
    y_true = np.array(all_targets)

    # 1. Métricas a nivel de Crop
    crop_metrics = framework.compute_crop_metrics(y_true, y_probs)

    # 2. Métricas agregadas a nivel de Objeto (Geoglyph)
    obj_metrics = framework.compute_object_level_f1(val_df, y_probs)

    return val_loss, crop_metrics, obj_metrics


# ==============================================================================
# 7. FUNCIÓN PRINCIPAL DE EJECUCIÓN
# ==============================================================================
def run_training_and_evaluation(args: argparse.Namespace) -> None:
    """Función orquestadora de carga, división, entrenamiento y evaluación."""
    # Fijar semillas para reproducibilidad científica
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info("Dispositivo de cómputo: %s", device)

    # 1. Carga y Filtrado de Metadatos
    df = load_and_filter_metadata(
        metadata_path=args.metadata_csv,
        images_dir=args.images_dir,
        target_region=args.region_id,
    )

    # 2. Entity-Aware Split (80% train / 20% val por geoglyph_id)
    train_df, val_df = entity_aware_train_val_split(
        df=df,
        train_size=args.train_ratio,
        random_state=args.seed,
    )

    # Transformaciones
    train_transform, val_transform = get_transforms()
    train_dataset = GeoglyphCropDataset(train_df, transform=train_transform)
    val_dataset = GeoglyphCropDataset(val_df, transform=val_transform)

    # 3. Muestreador Jerárquico Capped (N_c <= 15)
    train_sampler = SingleRegionHierarchicalCappedSampler(
        geoglyph_ids=train_df["geoglyph_id"].tolist(),
        max_crops_per_geo=args.max_crops_per_geo,
        shuffle=True,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # En validación no truncamos para evaluar la totalidad de los objetos
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Modelo y Pérdida
    model = build_model(model_name=args.model_name, pretrained=args.pretrained)
    model.to(device)

    if args.loss_type == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
        logger.info("Función de pérdida: FocalLoss(gamma=%.2f, alpha=%s)", args.focal_gamma, args.focal_alpha)
    else:
        criterion = nn.BCEWithLogitsLoss()
        logger.info("Función de pérdida: BCEWithLogitsLoss()")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 5. Accuracy Framework
    framework = GeospatialAccuracyFramework(
        object_threshold=args.object_threshold,
        landscape_positive_prevalence=args.landscape_prevalence,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / f"best_{args.model_name}_{args.region_id}.pth"

    best_val_pr_auc = -1.0
    history = []

    logger.info("Iniciando entrenamiento por %d épocas...", args.epochs)
    logger.info("-" * 80)

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, crop_metrics, obj_metrics = evaluate(
            model, val_loader, val_df, criterion, framework, device
        )
        scheduler.step()

        val_pr_auc = crop_metrics["pr_auc"]
        val_ua = crop_metrics["precision_ua"]
        val_pa = crop_metrics["recall_pa"]
        val_obj_f1 = obj_metrics["object_f1"]
        val_obj_prec = obj_metrics["object_precision"]
        val_obj_rec = obj_metrics["object_recall"]

        is_best = val_pr_auc > best_val_pr_auc
        if is_best:
            best_val_pr_auc = val_pr_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_pr_auc": best_val_pr_auc,
                "crop_metrics": crop_metrics,
                "object_metrics": obj_metrics,
                "args": vars(args),
            }, best_checkpoint_path)

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_pr_auc": val_pr_auc,
            "val_precision_ua": val_ua,
            "val_recall_pa": val_pa,
            "val_object_f1": val_obj_f1,
            "val_object_precision": val_obj_prec,
            "val_object_recall": val_obj_rec,
            "best": is_best,
        }
        history.append(epoch_record)

        # Registro detallado en cada época
        logger.info(
            "Época [%2d/%2d] | Train Loss: %.4f | Val Loss: %.4f | "
            "Crop PR-AUC: %.4f (UA: %.3f, PA: %.3f) | "
            "Object F1: %.4f (Prec: %.3f, Rec: %.3f)%s",
            epoch, args.epochs, train_loss, val_loss,
            val_pr_auc, val_ua, val_pa,
            val_obj_f1, val_obj_prec, val_obj_rec,
            " -> [MEJOR CHECKPOINT GUARDADO]" if is_best else ""
        )

    logger.info("=" * 80)
    logger.info("Entrenamiento finalizado. Mejor PR-AUC de Validación: %.4f", best_val_pr_auc)
    logger.info("Checkpoint óptimo preservado en: %s", best_checkpoint_path)

    # Cargar y reportar métricas completas del mejor modelo
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_crop, final_obj = evaluate(model, val_loader, val_df, criterion, framework, device)

    print("\n" + "=" * 60)
    print(f"REPORTE FINAL DE EVALUACIÓN - REGIÓN: {args.region_id}")
    print("=" * 60)
    print(f"Mejor Época: {checkpoint['epoch']}")
    print(f"Crop-Level PR-AUC (Average Precision): {final_crop['pr_auc']:.4f}")
    print(f"Crop-Level Precision (User's Accuracy, UA): {final_crop['precision_ua']:.4f}")
    print(f"Crop-Level Recall (Producer's Accuracy, PA): {final_crop['recall_pa']:.4f}")
    print(f"Crop-Level F1-Score: {final_crop['f1_crop']:.4f}")
    print("\nMatriz de Confusión Muestral a nivel de Crop:")
    print(f"  TN: {final_crop['confusion_matrix_sample']['TN']}, FP: {final_crop['confusion_matrix_sample']['FP']}")
    print(f"  FN: {final_crop['confusion_matrix_sample']['FN']}, TP: {final_crop['confusion_matrix_sample']['TP']}")
    print("\nMatriz Ajustada por Población (Olofsson et al., 2014):")
    print(f"  Prevalencia de Paisaje (W_pos): {final_crop['population_adjusted']['W_positive']:.4f}")
    print(f"  User's Accuracy Ajustada: {final_crop['population_adjusted']['pop_adjusted_ua']:.4f}")
    print(f"  Producer's Accuracy Ajustada: {final_crop['population_adjusted']['pop_adjusted_pa']:.4f}")
    print("-" * 60)
    print(f"Object-Level F1-Score (Umbral p >= {args.object_threshold}): {final_obj['object_f1']:.4f}")
    print(f"Object-Level Precision: {final_obj['object_precision']:.4f}")
    print(f"Object-Level Recall:    {final_obj['object_recall']:.4f}")
    print(f"Objetos Validados: {final_obj['num_objects']} ({final_obj['num_positive_objects']} pos, {final_obj['num_negative_objects']} neg)")
    print(f"Matriz de Confusión a nivel de Objeto: {final_obj['confusion_matrix_object']}")
    print("=" * 60 + "\n")

    # Guardar métricas en JSON
    metrics_report_path = output_dir / f"metrics_report_{args.region_id}.json"
    with open(metrics_report_path, "w", encoding="utf-8") as f:
        json.dump({
            "region_id": args.region_id,
            "best_epoch": checkpoint["epoch"],
            "best_crop_metrics": final_crop,
            "best_object_metrics": final_obj,
            "training_history": history,
        }, f, indent=4)
    logger.info("Reporte detallado de métricas guardado en: %s", metrics_report_path)


# ==============================================================================
# ENTRADA DE LÍNEA DE COMANDOS
# ==============================================================================
def get_args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenamiento y Evaluación Geoespacial de Geoglifos (PyTorch)"
    )
    # Rutas y Región
    parser.add_argument("--metadata_csv", type=str, default=None,
                        help="Ruta al CSV de metadatos (e.g. ML/ChugChug/image_metadata.csv)")
    parser.add_argument("--images_dir", type=str, default=None,
                        help="Ruta al directorio de recortes PNG de la región")
    parser.add_argument("--region_id", type=str, default="ChugChug",
                        help="ID de la región a procesar (default: 'ChugChug')")
    parser.add_argument("--output_dir", type=str, default="checkpoints",
                        help="Directorio donde guardar checkpoints y reportes")

    # Modelo y Pérdida
    parser.add_argument("--model_name", type=str, default="convnext_tiny",
                        choices=["convnext_tiny", "resnet50"],
                        help="Arquitectura de red neuronal ('convnext_tiny' o 'resnet50')")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Cargar pesos preentrenados de ImageNet")
    parser.add_argument("--loss_type", type=str, default="focal",
                        choices=["focal", "bce"],
                        help="Tipo de pérdida para desbalance de clase ('focal' o 'bce')")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Parámetro gamma para Focal Loss (default: 2.0)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                        help="Parámetro alpha para Focal Loss (default: 0.25)")

    # Estrategia de Muestreo y División
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Proporción para el split por entidad (default: 0.8)")
    parser.add_argument("--max_crops_per_geo", type=int, default=15,
                        help="Tope máximo de recortes N_c por geoglifo en Sampler (default: 15)")
    parser.add_argument("--object_threshold", type=float, default=0.65,
                        help="Umbral de probabilidad para detección a nivel de objeto (default: 0.65)")
    parser.add_argument("--landscape_prevalence", type=float, default=0.05,
                        help="Prevalencia real de geoglifos en el paisaje para matriz ajustada (default: 0.05)")

    # Hiperparámetros de Optimización
    parser.add_argument("--epochs", type=int, default=30,
                        help="Número de épocas de entrenamiento (default: 30)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Tamaño de lote (batch size) (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Tasa de aprendizaje (learning rate) inicial (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help="Weight decay para regularización L2 en AdamW (default: 1e-2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Semilla aleatoria para reproducibilidad (default: 42)")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Trabajadores en DataLoader (default: 0 para Windows/CPU)")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Desactivar CUDA forzando CPU")

    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    run_training_and_evaluation(args)
