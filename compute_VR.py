#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Modes
-----
1) STATIC mode (mode = "static"):
   - Each concept (slug) has a folder with a reference image.
   - For each model and slug:
       * Split the reference and each generated image into a regular grid
         of patches.
       * Embed patches with a DINO-style model.
       * For each generated image, compute per-patch maximum similarity to
         any reference patch and aggregate:
             - vr_max  = maximum over all generated patches
             - vr_mean = mean of per-patch maxima
       * Write one row per generated image to a CSV file.

2) DYNAMIC mode (mode = "dynamic"):
   - Each concept (slug) has a folder of multiple reference images.
   - For each model and slug:
       * Collect patches from ALL kept reference images into one pool.
       * For each generated image:
           - Split into patches.
           - Embed all patches.
           - Compute per-patch maximum similarity to ANY reference patch.
           - Aggregate as in STATIC: vr_max, vr_mean
       * Write one row per generated image to a CSV file.

"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Default (anonymized) configuration
# ---------------------------------------------------------------------

# These are placeholders. Replace them with the appropriate paths
# in your environment before running the script.

# STATIC setting
REFDIR_STATIC = Path("/path/to/static/reference_images")  # one subfolder per slug
MODEL_GENDIRS_STATIC: Dict[str, Path] = {
    "model_a": Path("/path/to/static/generated_images/model_a"),
    "model_b": Path("/path/to/static/generated_images/model_b"),
    # Add more models as needed
}
PROMPT_CSV_STATIC = Path("/path/to/static_prompts.csv")
PROMPT_SLUG_COL_STATIC = "label_clean"
OUTCSV_STATIC = Path("/path/to/results_vr_static.csv")
OUTMETA_STATIC = Path("/path/to/results_vr_static.meta.json")

# DYNAMIC setting
REFDIR_DYNAMIC = Path("/path/to/dynamic/reference_images")  # one subfolder per slug
MODEL_GENDIRS_DYNAMIC: Dict[str, Path] = {
    "model_a": Path("/path/to/dynamic/generated_images/model_a"),
    "model_b": Path("/path/to/dynamic/generated_images/model_b"),
    # Add more models as needed
}
PROMPT_CSV_DYNAMIC = Path("/path/to/dynamic_prompts.csv")
PROMPT_SLUG_COL_DYNAMIC = "label_clean"
OUTCSV_DYNAMIC = Path("/path/to/results_vr_dynamic.csv")
OUTMETA_DYNAMIC = Path("/path/to/results_vr_dynamic.meta.json")

# Image handling
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Limits
K_GEN_DEFAULT = 50   # default max generated images per slug (can be overridden)
MIN_GEN = 1

# Patch grid (can be overridden via CLI)
GRID_ROWS_DEFAULT = 4
GRID_COLS_DEFAULT = 4

# DINO-style backbone (placeholder name; adjust to your DINOv3 model if needed)
DINOBACKBONE_NAME = "vit_base_patch14_dinov2"  # example timm model; replace with DINOv3 if available

# pHash dedup threshold (Hamming distance <= this means "duplicate")
PHASH_DUP_THRESH = 18

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("vr_dino")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------


def p(msg: str) -> None:
    """Lightweight printing with flush."""
    print(msg, flush=True)


def norm_slug(s: str) -> str:
    """Normalize a slug (folder name) to a canonical form."""
    return s.strip().replace(" ", "_").lower()


def index_slug_dirs(root: Path) -> Dict[str, Path]:
    """
    Map normalized slug -> folder path under the given root.
    Folders starting with '.' are ignored.
    """
    out: Dict[str, Path] = {}
    if not root.exists():
        return out
    for q in sorted([q for q in root.iterdir() if q.is_dir() and not q.name.startswith(".")]):
        out[norm_slug(q.name)] = q
    return out


def list_images(folder: Path) -> List[Path]:
    """List image files in a folder, sorted by name."""
    if not folder.exists():
        return []
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS])


def load_first_ref_image(slug_ref_dir: Path) -> Tuple[Image.Image, Path]:
    """Load the first reference image in a slug folder (STATIC mode)."""
    paths = list_images(slug_ref_dir)
    if not paths:
        raise FileNotFoundError(f"No reference images in {slug_ref_dir}")
    return Image.open(paths[0]).convert("RGB"), paths[0]


def load_all_ref_images(slug_ref_dir: Path) -> Tuple[List[Image.Image], List[Path]]:
    """Load all reference images in a slug folder (DYNAMIC mode)."""
    paths = list_images(slug_ref_dir)
    if not paths:
        raise FileNotFoundError(f"No reference images in {slug_ref_dir}")
    imgs, kept_paths = [], []
    for pth in paths:
        try:
            imgs.append(Image.open(pth).convert("RGB"))
            kept_paths.append(pth)
        except Exception as e:
            p(f"[warn] Failed to load reference image {pth}: {e}")
    if not imgs:
        raise FileNotFoundError(f"Failed to load any reference images in {slug_ref_dir}")
    return imgs, kept_paths


def load_gen_images(slug_gen_dir: Path, k: int) -> Tuple[List[Image.Image], List[Path]]:
    """Load up to k generated images for a slug."""
    paths = list_images(slug_gen_dir)[:k]
    imgs, kept = [], []
    for pth in paths:
        try:
            imgs.append(Image.open(pth).convert("RGB"))
            kept.append(pth)
        except Exception as e:
            p(f"[warn] Failed to load generated image {pth}: {e}")
    return imgs, kept


def split_into_grid(img: Image.Image, rows: int, cols: int) -> List[Image.Image]:
    """Split an image into a regular rows×cols grid of rectangular patches."""
    W, H = img.size
    w = max(1, W // cols)
    h = max(1, H // rows)
    patches: List[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            left = c * w
            top = r * h
            right = (c + 1) * w if c < cols - 1 else W
            bottom = (r + 1) * h if r < rows - 1 else H
            patches.append(img.crop((left, top, right, bottom)))
    return patches


def read_slug_whitelist(csv_path: Path, slug_col: str) -> Set[str]:
    """
    Read a CSV file and return the set of normalized slugs from the given column.
    Only these slugs will be processed.
    """
    df = pd.read_csv(csv_path)
    if slug_col not in df.columns:
        raise ValueError(f"Column '{slug_col}' not found in {csv_path}")
    return set(df[slug_col].astype(str).map(norm_slug).tolist())


def dedup_refs_by_phash(
    ref_imgs: List[Image.Image],
    ref_paths: List[Path],
    thresh: int = PHASH_DUP_THRESH,
) -> Tuple[List[Image.Image], List[Path], int]:
    """
    Deduplicate reference images using perceptual hash (pHash).
    Keeps the first occurrence of near-duplicates; drops later ones if the
    Hamming distance between hashes is <= thresh.

    Returns (kept_images, kept_paths, num_removed).

    If the 'imagehash' library is not available, returns the input as-is.
    """
    try:
        import imagehash
    except Exception as e:
        p(f"[pHash warn] 'imagehash' not installed; skipping deduplication. ({e})")
        return ref_imgs, ref_paths, 0

    kept_imgs: List[Image.Image] = []
    kept_paths: List[Path] = []
    hashes: List[Optional[object]] = []  # imagehash.ImageHash, but kept generic here
    removed = 0

    for img, path in zip(ref_imgs, ref_paths):
        try:
            h = imagehash.phash(img)
        except Exception as e:
            p(f"[pHash warn] Failed to compute pHash for {path.name}: {e}")
            kept_imgs.append(img)
            kept_paths.append(path)
            hashes.append(None)
            continue

        is_dup = False
        for h_prev in hashes:
            if h_prev is None:
                continue
            if h - h_prev <= thresh:
                is_dup = True
                break
        if is_dup:
            removed += 1
        else:
            kept_imgs.append(img)
            kept_paths.append(path)
            hashes.append(h)

    return kept_imgs, kept_paths, removed


def ensure_outcsv_header(path: Path, mode: str) -> None:
    """Create the output CSV with the appropriate header if it does not exist."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    cols = [
        "mode",             # "static" or "dynamic"
        "model",
        "slug",
        "ref_image_path",   # static: first ref; dynamic: first kept ref
        "gen_image_path",
        "grid_rows",
        "grid_cols",
        "n_ref_patches",
        "n_gen_patches",
        "vr_max",           # max per-patch similarity (best-case reuse)
        "vr_mean",          # mean of per-patch maxima
        "vr_p90",           # 90th percentile of per-patch maxima
    ]

    if mode == "dynamic":
        cols += [
            "ref_images_kept",        # ';'-joined paths of kept refs after pHash
            "n_ref_before",           # number of refs before dedup
            "n_ref_kept",             # number of refs kept after dedup
            "n_ref_dups_removed",     # number removed as near-duplicates
        ]

    pd.DataFrame(columns=cols).to_csv(path, index=False)


def already_done_index(path: Path) -> Set[str]:
    """
    Return the set of generated image paths already present in the output CSV.
    Used to skip images on subsequent runs (auto-resume).
    """
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path, usecols=["gen_image_path"])
        return set(df["gen_image_path"].astype(str).tolist())
    except Exception:
        return set()

# ---------------------------------------------------------------------
# DINO-style embedder (for VR)
# ---------------------------------------------------------------------


class DINOEmbedder:
    """
    DINO-style vision encoder wrapper.

    This implementation uses timm to load a ViT backbone with DINO-style
    pretraining (e.g., DINOv2 / DINOv3). The model name is configurable
    via DINOBACKBONE_NAME.

    The forward_features output is collapsed to a single feature vector
    per image (e.g., using a pooled representation or token mean), and
    then L2-normalized.
    """

    def __init__(self, model_name: str) -> None:
        try:
            import timm
            from timm.data import resolve_data_config
            from timm.data.transforms_factory import create_transform

            self.model_name = model_name
            self.model = timm.create_model(model_name, pretrained=True)
            self.model.eval().to(DEVICE)

            cfg = getattr(self.model, "pretrained_cfg", None) or getattr(self.model, "default_cfg", {})
            data_cfg = resolve_data_config(cfg)
            self.preprocess = create_transform(**data_cfg)

            if hasattr(self.model, "reset_classifier"):
                self.model.reset_classifier(0)

            p(f"[VR] Loaded DINO-style backbone via timm: {model_name}")
        except Exception as e:
            raise RuntimeError("Install timm and ensure the model name is valid. Error: " + str(e))

    @torch.no_grad()
    def embed_images(self, images: List[Image.Image], batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Embed a list of PIL images and return an [N, D] tensor of L2-normalized features.
        """
        if len(images) == 0:
            return torch.empty(0, 0)

        if batch_size is None:
            batch_size = 64 if DEVICE == "cuda" else 16

        tensors = [self.preprocess(img) for img in images]
        x = torch.stack(tensors)  # [N, 3, H, W]
        outs = []
        for i in range(0, x.size(0), batch_size):
            xb = x[i: i + batch_size].to(DEVICE)
            fb = self.model.forward_features(xb)

            if isinstance(fb, dict):
                # Prefer pooled or average-pooled tokens if available
                if "pooled" in fb and fb["pooled"].dim() == 2:
                    z = fb["pooled"]  # [B, C]
                elif "avg_pool" in fb and fb["avg_pool"].dim() == 2:
                    z = fb["avg_pool"]  # [B, C]
                elif "xnorm" in fb and fb["xnorm"].dim() == 3:
                    z = fb["xnorm"].mean(dim=1)  # [B, T, C] -> [B, C]
                elif "x_norm" in fb and fb["x_norm"].dim() == 3:
                    z = fb["x_norm"].mean(dim=1)  # [B, T, C] -> [B, C]
                else:
                    v = next(iter(fb.values()))
                    z = v.mean(dim=1) if v.dim() == 3 else v
            else:
                # Some timm models may return a tensor directly
                z = fb.mean(dim=1) if fb.dim() == 3 else fb

            z = F.normalize(z, dim=-1)
            outs.append(z.detach().cpu())
        return torch.cat(outs, dim=0)


def compute_vr_stats(ref_patches: List[Image.Image], gen_patches: List[Image.Image], embedder: DINOEmbedder) -> Tuple[float, float, float, int, int]:
    """
    Compute patch-level VR statistics between reference and generated patches.

    - ref_patches: list of reference patches (possibly from one or many refs).
    - gen_patches: list of generated patches.

    Returns:
        vr_max, vr_mean, vr_p90, n_ref_patches, n_gen_patches

    vr_* are derived from the distribution of per-patch maxima, where for
    each generated patch we compute the maximum cosine similarity to any
    reference patch.
    """
    if len(ref_patches) == 0 or len(gen_patches) == 0:
        return float("nan"), float("nan"), float("nan"), len(ref_patches), len(gen_patches)

    ref_emb = embedder.embed_images(ref_patches)  # [R, D]
    gen_emb = embedder.embed_images(gen_patches)  # [G, D]
    if ref_emb.numel() == 0 or gen_emb.numel() == 0:
        return float("nan"), float("nan"), float("nan"), len(ref_patches), len(gen_patches)

    # Normalize again to be explicit
    ref_emb = F.normalize(ref_emb, dim=-1)
    gen_emb = F.normalize(gen_emb, dim=-1)

    sim = gen_emb @ ref_emb.T  # [G, R]
    max_per_patch = sim.max(dim=1).values  # [G]

    vr_max = float(max_per_patch.max().item())
    vr_mean = float(max_per_patch.mean().item())

    try:
        vr_p90 = float(torch.quantile(max_per_patch, torch.tensor(0.9)).item())
    except Exception:
        vr_p90 = float("nan")

    return vr_max, vr_mean, vr_p90, len(ref_patches), len(gen_patches)

# ---------------------------------------------------------------------
# Core computation: STATIC mode
# ---------------------------------------------------------------------


def run_static_for_model(
    model_name: str,
    gendir: Path,
    refdir: Path,
    dino_embedder: DINOEmbedder,
    whitelist: Set[str],
    k_gen: int,
    grid_rows: int,
    grid_cols: int,
    outcsv: Path,
) -> None:
    """
    Run VR computation for one model in STATIC mode.

    - refdir: root folder with one subfolder per slug, each containing reference images.
    - gendir: root folder with one subfolder per slug for generated images of this model.
    - whitelist: set of normalized slugs to consider.
    """
    gen_index = index_slug_dirs(gendir)   # norm_slug -> generated folder
    ref_index = index_slug_dirs(refdir)   # norm_slug -> reference folder

    eligible_norms = sorted([s for s in whitelist if s in gen_index and s in ref_index])

    p(f"\n=== STATIC | Model: {model_name} ===")
    p(f"Generated root: {gendir}")
    p(f"Reference root: {refdir}")
    p(f"[DIAG] slugs in whitelist: {len(whitelist)} | in gen: {len(gen_index)} | in ref: {len(ref_index)}")
    p(f"[DIAG] eligible intersection: {len(eligible_norms)}")

    if not eligible_norms:
        p(f"[STATIC | {model_name}] No eligible slugs after intersecting whitelist ∩ gen ∩ ref.")
        return

    done = already_done_index(outcsv)
    total_appended = 0
    total_checked = 0

    for idx, slug_norm in enumerate(eligible_norms, 1):
        ref_dir = ref_index[slug_norm]
        gen_dir = gen_index[slug_norm]
        slug = ref_dir.name

        p(f"\n[STATIC | {model_name}] ({idx}/{len(eligible_norms)}) slug: {slug} (norm: {slug_norm})")

        try:
            ref_img, ref_path = load_first_ref_image(ref_dir)
            p(f"  - First reference: {ref_path.name}")
        except Exception as e:
            p(f"  - [Skip] Failed to load first reference for slug '{slug}': {e}")
            continue

        gen_imgs, gen_paths = load_gen_images(gen_dir, k_gen)
        p(f"  - Found {len(gen_paths)} generated images.")
        if len(gen_imgs) < MIN_GEN:
            p(f"  - [Skip] Not enough generated images for slug '{slug}'.")
            continue

        ref_patches = split_into_grid(ref_img, grid_rows, grid_cols)

        appended_here = 0
        for gi, gp in zip(gen_imgs, gen_paths):
            total_checked += 1
            gp_str = str(gp)

            if gp_str in done:
                p(f"    - [ResumeSkip] {gp.name} already present in CSV.")
                continue

            gen_patches = split_into_grid(gi, grid_rows, grid_cols)
            vr_max, vr_mean, vr_p90, n_ref_patches, n_gen_patches = compute_vr_stats(
                ref_patches, gen_patches, dino_embedder
            )

            p(f"    - {gp.name}: VR_max={vr_max:.3f} | VR_mean={vr_mean:.3f} | VR_p90={vr_p90:.3f}")

            row = {
                "mode": "static",
                "model": model_name,
                "slug": slug,
                "ref_image_path": str(ref_path),
                "gen_image_path": gp_str,
                "grid_rows": int(grid_rows),
                "grid_cols": int(grid_cols),
                "n_ref_patches": int(n_ref_patches),
                "n_gen_patches": int(n_gen_patches),
                "vr_max": float(vr_max),
                "vr_mean": float(vr_mean),
                "vr_p90": float(vr_p90),
            }

            pd.DataFrame([row]).to_csv(outcsv, mode="a", index=False, header=False)
            appended_here += 1
            total_appended += 1

        p(f"  - Appended {appended_here} rows for slug '{slug}'.")

    p(f"\n[STATIC | {model_name}] Summary: checked images = {total_checked} | "
      f"rows appended = {total_appended}")

# ---------------------------------------------------------------------
# Core computation: DYNAMIC mode
# ---------------------------------------------------------------------


def run_dynamic_for_model(
    model_name: str,
    gendir: Path,
    refdir: Path,
    dino_embedder: DINOEmbedder,
    whitelist: Set[str],
    k_gen: int,
    grid_rows: int,
    grid_cols: int,
    outcsv: Path,
) -> None:
    """
    Run VR computation for one model in DYNAMIC mode.

    - refdir: root folder with one subfolder per slug, each containing multiple references.
    - gendir: root folder with one subfolder per slug for generated images of this model.
    - whitelist: set of normalized slugs to consider.
    """
    gen_index = index_slug_dirs(gendir)   # norm_slug -> generated folder
    ref_index = index_slug_dirs(refdir)   # norm_slug -> reference folder

    eligible_norms = sorted([s for s in whitelist if s in gen_index and s in ref_index])

    p(f"\n=== DYNAMIC | Model: {model_name} ===")
    p(f"Generated root: {gendir}")
    p(f"Reference root: {refdir}")
    p(f"[DIAG] slugs in whitelist: {len(whitelist)} | in gen: {len(gen_index)} | in ref: {len(ref_index)}")
    p(f"[DIAG] eligible intersection: {len(eligible_norms)}")

    if not eligible_norms:
        p(f"[DYNAMIC | {model_name}] No eligible slugs after intersecting whitelist ∩ gen ∩ ref.")
        return

    done = already_done_index(outcsv)
    total_appended = 0
    total_checked = 0

    for idx, slug_norm in enumerate(eligible_norms, 1):
        ref_dir = ref_index[slug_norm]
        gen_dir = gen_index[slug_norm]
        slug = ref_dir.name

        p(f"\n[DYNAMIC | {model_name}] ({idx}/{len(eligible_norms)}) slug: {slug} (norm: {slug_norm})")

        # Load & deduplicate references
        try:
            ref_imgs_all, ref_paths_all = load_all_ref_images(ref_dir)
            n_ref_before = len(ref_imgs_all)
            ref_imgs, ref_paths, n_removed = dedup_refs_by_phash(ref_imgs_all, ref_paths_all, PHASH_DUP_THRESH)
            n_ref_kept = len(ref_imgs)
            p(f"  - References: {n_ref_before} total | kept = {n_ref_kept} | "
              f"removed by pHash ≤ {PHASH_DUP_THRESH} = {n_removed}")
            if n_ref_kept == 0:
                p("  - [Skip] No reference images left after deduplication.")
                continue
        except Exception as e:
            p(f"  - [Skip] Failed to load references for slug '{slug}': {e}")
            continue

        gen_imgs, gen_paths = load_gen_images(gen_dir, k_gen)
        p(f"  - Found {len(gen_paths)} generated images.")
        if len(gen_imgs) < MIN_GEN:
            p(f"  - [Skip] Not enough generated images for slug '{slug}'.")
            continue

        # Build combined reference patch bank
        ref_patches: List[Image.Image] = []
        for ri in ref_imgs:
            ref_patches.extend(split_into_grid(ri, grid_rows, grid_cols))

        ref_paths_kept_str = ";".join(str(pth) for pth in ref_paths)
        first_ref_path = str(ref_paths[0])

        appended_here = 0
        for gi, gp in zip(gen_imgs, gen_paths):
            total_checked += 1
            gp_str = str(gp)

            if gp_str in done:
                p(f"    - [ResumeSkip] {gp.name} already present in CSV.")
                continue

            gen_patches = split_into_grid(gi, grid_rows, grid_cols)
            vr_max, vr_mean, vr_p90, n_ref_patches, n_gen_patches = compute_vr_stats(
                ref_patches, gen_patches, dino_embedder
            )

            p(f"    - {gp.name}: VR_max={vr_max:.3f} | VR_mean={vr_mean:.3f} | VR_p90={vr_p90:.3f}")

            row = {
                "mode": "dynamic",
                "model": model_name,
                "slug": slug,
                "ref_image_path": first_ref_path,
                "gen_image_path": gp_str,
                "grid_rows": int(grid_rows),
                "grid_cols": int(grid_cols),
                "n_ref_patches": int(n_ref_patches),
                "n_gen_patches": int(n_gen_patches),
                "vr_max": float(vr_max),
                "vr_mean": float(vr_mean),
                "vr_p90": float(vr_p90),
                "ref_images_kept": ref_paths_kept_str,
                "n_ref_before": int(n_ref_before),
                "n_ref_kept": int(n_ref_kept),
                "n_ref_dups_removed": int(n_removed),
            }

            pd.DataFrame([row]).to_csv(outcsv, mode="a", index=False, header=False)
            appended_here += 1
            total_appended += 1

        p(f"  - Appended {appended_here} rows for slug '{slug}'.")

    p(f"\n[DYNAMIC | {model_name}] Summary: checked images = {total_checked} | "
      f"rows appended = {total_appended}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute patch-level visual reuse (VR) with a DINO-style backbone.")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["static", "dynamic"],
        required=True,
        help="Which setting to run: 'static' (first ref per slug) or 'dynamic' (all refs, max over patch pool)."
    )
    ap.add_argument(
        "--model",
        type=str,
        help="If provided, run only this model (must match a key in the corresponding MODEL_GENDIRS dict). "
             "If omitted, runs all models defined there."
    )
    ap.add_argument(
        "--k-gen",
        type=int,
        default=K_GEN_DEFAULT,
        help="Maximum number of generated images per slug."
    )
    ap.add_argument(
        "--grid-rows",
        type=int,
        default=GRID_ROWS_DEFAULT,
        help="Number of rows in the patch grid."
    )
    ap.add_argument(
        "--grid-cols",
        type=int,
        default=GRID_COLS_DEFAULT,
        help="Number of columns in the patch grid."
    )
    ap.add_argument(
        "--prompt-csv",
        type=Path,
        default=None,
        help="Path to the prompt CSV with a slug column. If not set, a mode-specific default is used."
    )
    ap.add_argument(
        "--slug-col",
        type=str,
        default=None,
        help="Name of the slug column in the prompt CSV. If not set, a mode-specific default is used."
    )
    ap.add_argument(
        "--refdir",
        type=Path,
        default=None,
        help="Root directory for reference images. If not set, a mode-specific default is used."
    )
    ap.add_argument(
        "--outcsv",
        type=Path,
        default=None,
        help="Output CSV path. If not set, a mode-specific default is used."
    )
    ap.add_argument(
        "--outmeta",
        type=Path,
        default=None,
        help="Metadata JSON path. If not set, a mode-specific default is used."
    )
    ap.add_argument(
        "--dino-model",
        type=str,
        default=DINOBACKBONE_NAME,
        help="Name of the DINO-style backbone for timm.create_model (e.g., a DINOv3 ViT)."
    )

    args = ap.parse_args()

    mode = args.mode

    # Mode-specific configuration (can be overridden by CLI)
    if mode == "static":
        refdir = args.refdir or REFDIR_STATIC
        model_gendirs = MODEL_GENDIRS_STATIC
        prompt_csv = args.prompt_csv or PROMPT_CSV_STATIC
        slug_col = args.slug_col or PROMPT_SLUG_COL_STATIC
        outcsv = args.outcsv or OUTCSV_STATIC
        outmeta = args.outmeta or OUTMETA_STATIC
    else:
        refdir = args.refdir or REFDIR_DYNAMIC
        model_gendirs = MODEL_GENDIRS_DYNAMIC
        prompt_csv = args.prompt_csv or PROMPT_CSV_DYNAMIC
        slug_col = args.slug_col or PROMPT_SLUG_COL_DYNAMIC
        outcsv = args.outcsv or OUTCSV_DYNAMIC
        outmeta = args.outmeta or OUTMETA_DYNAMIC

    grid_rows = args.grid_rows
    grid_cols = args.grid_cols

    p(f"[Device] {DEVICE}")
    p(f"[Mode] {mode}")
    p(f"[VR] DINO-style backbone: {args.dino_model}")
    p(f"[Refdir] {refdir}")
    p(f"[Prompt CSV] {prompt_csv} (slug column: '{slug_col}')")
    p(f"[OutCSV] {outcsv}")
    p(f"[OutMeta] {outmeta}")
    p(f"[Grid] {grid_rows}x{grid_cols}")

    # Ensure output CSV has header
    ensure_outcsv_header(outcsv, mode)

    # Load slug whitelist
    whitelist = read_slug_whitelist(prompt_csv, slug_col)
    p(f"[Whitelist] Loaded {len(whitelist)} slugs from '{prompt_csv.name}'")

    # Initialize DINO embedder once
    dino_embedder = DINOEmbedder(args.dino_model)

    # Prepare metadata
    meta = {
        "device": DEVICE,
        "mode": mode,
        "dino_backbone": args.dino_model,
        "k_gen": int(args.k_gen),
        "min_gen": int(MIN_GEN),
        "grid_rows": int(grid_rows),
        "grid_cols": int(grid_cols),
        "refdir": str(refdir),
        "gendirs": {k: str(v) for k, v in model_gendirs.items()},
        "prompt_csv": str(prompt_csv),
        "slug_column": slug_col,
        "outcsv": str(outcsv),
        "note": "VR scores are patch-level similarities; higher values indicate stronger local visual reuse.",
    }

    if mode == "dynamic":
        meta["dedup"] = {
            "method": "pHash",
            "hamming_thresh": int(PHASH_DUP_THRESH),
        }

    outmeta.parent.mkdir(parents=True, exist_ok=True)
    with open(outmeta, "w") as f:
        json.dump(meta, f, indent=2)
    p(f"[Meta] Saved → {outmeta}")

    # Run for one or all models
    if args.model:
        if args.model not in model_gendirs:
            raise ValueError(f"Model '{args.model}' not found in model directory configuration.")
        run_fn = run_static_for_model if mode == "static" else run_dynamic_for_model
        run_fn(
            args.model,
            model_gendirs[args.model],
            refdir,
            dino_embedder,
            whitelist,
            args.k_gen,
            grid_rows,
            grid_cols,
            outcsv,
        )
    else:
        for model_name, gendir in model_gendirs.items():
            run_fn = run_static_for_model if mode == "static" else run_dynamic_for_model
            run_fn(
                model_name,
                gendir,
                refdir,
                dino_embedder,
                whitelist,
                args.k_gen,
                grid_rows,
                grid_cols,
                outcsv,
            )

    p(f"\n[Done] VR results written to {outcsv}")


if __name__ == "__main__":
    main()
