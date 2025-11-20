#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

Modes
-----
1) STATIC mode (mode = "static"):
   - Each concept (slug) has a folder with a reference images.
   - For each model and slug:
       * Compute CRA as the cosine similarity between the first reference image
         and each generated image in the corresponding folder.
       * Keep only generated images with CRA >= ic_tau.
       * Write one row per CRA-pass image to a CSV file.

2) DYNAMIC mode (mode = "dynamic"):
   - Each concept (slug) has a folder of multiple reference images.
   - All reference images are loaded and optionally deduplicated using
     perceptual hashing (pHash).
   - For each model and slug:
       * Compute CRA as the MAX cosine similarity between the generated image
         and ALL kept reference images.
       * Keep only generated images with CRA >= ic_tau.
       * Write one row per CRA-pass image to a CSV file.
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
OUTCSV_STATIC = Path("/path/to/results_cra_static.csv")
OUTMETA_STATIC = Path("/path/to/results_cra_static.meta.json")

# DYNAMIC setting
REFDIR_DYNAMIC = Path("/path/to/dynamic/reference_images")  # one subfolder per slug
MODEL_GENDIRS_DYNAMIC: Dict[str, Path] = {
    "model_a": Path("/path/to/dynamic/generated_images/model_a"),
    "model_b": Path("/path/to/dynamic/generated_images/model_b"),
    # Add more models as needed
}
PROMPT_CSV_DYNAMIC = Path("/path/to/dynamic_prompts.csv")
PROMPT_SLUG_COL_DYNAMIC = "label_clean"
OUTCSV_DYNAMIC = Path("/path/to/results_cra_dynamic.csv")
OUTMETA_DYNAMIC = Path("/path/to/results_cra_dynamic.meta.json")

# Image handling
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Limits
K_GEN_DEFAULT = 50   # default max generated images per slug (can be overridden)
MIN_GEN = 1

# Threshold (CRA; can be overridden via CLI; if None, set per mode)
IC_TAU_STATIC_DEFAULT = 0.70
IC_TAU_DYNAMIC_DEFAULT = 0.75

# Embedding models
STCLIP_MODEL_NAME = "sentence-transformers/st-clip-vit-b-16"
HF_CLIP_MODEL_NAME = "openai/clip-vit-base-patch16"

# Optional: if the environment needs explicit token passing for Hugging Face
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# pHash dedup threshold (Hamming distance <= this means "duplicate")
PHASH_DUP_THRESH = 18

# Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("cra_static_dynamic")

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
    hashes: List[Optional[imagehash.ImageHash]] = []
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

    # Common columns for both modes
    cols = [
        "mode",             # "static" or "dynamic"
        "model",
        "slug",
        "ref_image_path",   # static: first ref; dynamic: first kept ref
        "gen_image_path",
        "cra_sim",          # CLIP-based reference alignment score
        "cra_pass",         # True if cra_sim >= ic_tau
    ]

    # Additional columns used in dynamic mode
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
# CRA embedder (ST-CLIP / CLIP)
# ---------------------------------------------------------------------


class ImageEmbedderCRA:
    """
    Image embedder for CLIP-based reference alignment (CRA).

    Primary backend:
        - SentenceTransformers ST-CLIP (image encoder)
    Fallback backend:
        - Hugging Face CLIP (image encoder)

    Both backends return L2-normalized embeddings suitable for cosine similarity.
    """

    def __init__(self) -> None:
        self.backend: Optional[str] = None
        self.model_name: Optional[str] = None
        self._st_model = None
        self._hf_model = None
        self._hf_processor = None

        # Try ST-CLIP
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            kwargs = {}
            if HF_TOKEN:
                # Older versions use 'use_auth_token', newer ones may ignore it.
                kwargs["use_auth_token"] = HF_TOKEN

            self._st_model = SentenceTransformer(STCLIP_MODEL_NAME, device=DEVICE, **kwargs)
            self._st_model.eval()
            self.backend = "sentence-transformers"
            self.model_name = STCLIP_MODEL_NAME
            p(f"[CRA] Loaded ST-CLIP: {STCLIP_MODEL_NAME}")
            return
        except Exception as e:
            p(f"[CRA warn] ST-CLIP failed ({e}); falling back to Hugging Face CLIP...")

        # Fallback: Hugging Face CLIP
        try:
            from transformers import CLIPProcessor, CLIPModel  # type: ignore

            tkwargs = {}
            if HF_TOKEN:
                # Newer transformers use 'token'; older ones use 'use_auth_token'.
                try:
                    tkwargs["token"] = HF_TOKEN
                except TypeError:
                    tkwargs["use_auth_token"] = HF_TOKEN

            self._hf_model = CLIPModel.from_pretrained(HF_CLIP_MODEL_NAME, **tkwargs).to(DEVICE)
            self._hf_model.eval()
            self._hf_processor = CLIPProcessor.from_pretrained(HF_CLIP_MODEL_NAME, **tkwargs)
            self.backend = "hf-clip"
            self.model_name = HF_CLIP_MODEL_NAME
            p(f"[CRA] Loaded Hugging Face CLIP: {HF_CLIP_MODEL_NAME}")
        except Exception as e:
            raise RuntimeError(
                "Failed to load both ST-CLIP and Hugging Face CLIP. "
                "If you see a 401 error, ensure appropriate authentication is configured. "
                f"Underlying error: {e}"
            )

    @torch.no_grad()
    def embed(self, images: List[Image.Image], batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Embed a list of PIL images and return an [N, D] tensor of L2-normalized features.
        """
        if len(images) == 0:
            return torch.empty(0, 0)

        if self.backend == "sentence-transformers":
            if batch_size is None:
                batch_size = 64 if DEVICE == "cuda" else 16
            embs = self._st_model.encode(
                images=images,
                batch_size=batch_size,
                convert_to_tensor=True,
                device=DEVICE,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            return F.normalize(embs, dim=-1).detach().cpu()
        else:
            # Hugging Face CLIP fallback
            if batch_size is None:
                batch_size = 32 if DEVICE == "cuda" else 8
            outs = []
            for i in range(0, len(images), batch_size):
                batch = images[i: i + batch_size]
                inputs = self._hf_processor(images=batch, return_tensors="pt").to(DEVICE)
                feats = self._hf_model.get_image_features(**inputs)
                outs.append(F.normalize(feats, dim=-1))
            return torch.cat(outs, dim=0).detach().cpu()

# ---------------------------------------------------------------------
# Core computation: STATIC mode
# ---------------------------------------------------------------------


def run_static_for_model(
    model_name: str,
    gendir: Path,
    refdir: Path,
    cra_embedder: ImageEmbedderCRA,
    whitelist: Set[str],
    k_gen: int,
    ic_tau: float,
    outcsv: Path,
) -> None:
    """
    Run CRA computation for one model in STATIC mode.

    - refdir: root folder with one subfolder per slug, each containing reference images.
    - gendir: root folder with one subfolder per slug for generated images of this model.
    - whitelist: set of normalized slugs to consider.
    - ic_tau: CRA threshold.
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
    total_cra_pass = 0

    for idx, slug_norm in enumerate(eligible_norms, 1):
        ref_dir = ref_index[slug_norm]
        gen_dir = gen_index[slug_norm]
        slug = ref_dir.name

        p(f"\n[STATIC | {model_name}] ({idx}/{len(eligible_norms)}) slug: {slug} (norm: {slug_norm})")

        # Load first reference for this slug
        try:
            ref_img, ref_path = load_first_ref_image(ref_dir)
            p(f"  - First reference: {ref_path.name}")
        except Exception as e:
            p(f"  - [Skip] Failed to load first reference for slug '{slug}': {e}")
            continue

        # Load generated images for this slug
        gen_imgs, gen_paths = load_gen_images(gen_dir, k_gen)
        p(f"  - Found {len(gen_paths)} generated images.")
        if len(gen_imgs) < MIN_GEN:
            p(f"  - [Skip] Not enough generated images for slug '{slug}'.")
            continue

        # Compute CRA = cosine similarity between ref and each gen image
        try:
            img_list = [ref_img] + gen_imgs
            embs = cra_embedder.embed(img_list)  # [1 + N, D]
        except Exception as e:
            p(f"  - [Skip] CRA embedding failed for slug '{slug}': {e}")
            continue

        if embs.numel() == 0 or embs.shape[0] != (1 + len(gen_imgs)):
            p(f"  - [Skip] Empty or misaligned CRA embeddings for slug '{slug}'.")
            continue

        ref_emb = embs[0:1, :]
        gen_emb = embs[1:, :]
        cra_sims = (gen_emb @ ref_emb.T).squeeze(1).numpy()  # [N]

        appended_here = 0
        for gi, gp, cra_sim in zip(gen_imgs, gen_paths, cra_sims):
            total_checked += 1
            gp_str = str(gp)

            if gp_str in done:
                p(f"    - [ResumeSkip] {gp.name} already present in CSV.")
                continue

            passed = bool(cra_sim >= ic_tau)
            p(f"    - {gp.name}: CRA={cra_sim:.3f} (τ={ic_tau}) → {'PASS' if passed else 'FAIL'}")

            if not passed:
                continue

            total_cra_pass += 1

            row = {
                "mode": "static",
                "model": model_name,
                "slug": slug,
                "ref_image_path": str(ref_path),
                "gen_image_path": gp_str,
                "cra_sim": float(cra_sim),
                "cra_pass": True,
            }

            # Append row immediately (incremental CSV)
            pd.DataFrame([row]).to_csv(outcsv, mode="a", index=False, header=False)
            appended_here += 1
            total_appended += 1

        p(f"  - Appended {appended_here} rows for slug '{slug}'.")

    p(f"\n[STATIC | {model_name}] Summary: checked images = {total_checked} | "
      f"CRA-pass = {total_cra_pass} | rows appended = {total_appended}")

# ---------------------------------------------------------------------
# Core computation: DYNAMIC mode
# ---------------------------------------------------------------------


def run_dynamic_for_model(
    model_name: str,
    gendir: Path,
    refdir: Path,
    cra_embedder: ImageEmbedderCRA,
    whitelist: Set[str],
    k_gen: int,
    ic_tau: float,
    outcsv: Path,
) -> None:
    """
    Run CRA computation for one model in DYNAMIC mode.

    - refdir: root folder with one subfolder per slug, each containing multiple references.
    - gendir: root folder with one subfolder per slug for generated images of this model.
    - whitelist: set of normalized slugs to consider.
    - ic_tau: CRA threshold, applied to the maximum similarity vs all kept references.
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
    total_cra_pass = 0

    for idx, slug_norm in enumerate(eligible_norms, 1):
        ref_dir = ref_index[slug_norm]
        gen_dir = gen_index[slug_norm]
        slug = ref_dir.name

        p(f"\n[DYNAMIC | {model_name}] ({idx}/{len(eligible_norms)}) slug: {slug} (norm: {slug_norm})")

        # Load all references for this slug
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

        # Load generated images
        gen_imgs, gen_paths = load_gen_images(gen_dir, k_gen)
        p(f"  - Found {len(gen_paths)} generated images.")
        if len(gen_imgs) < MIN_GEN:
            p(f"  - [Skip] Not enough generated images for slug '{slug}'.")
            continue

        # Embed ALL kept references once
        try:
            ref_embs = cra_embedder.embed(ref_imgs)   # [R, D]
            if ref_embs.numel() == 0 or ref_embs.shape[0] != len(ref_imgs):
                p("  - [Skip] Empty or misaligned CRA embeddings for references.")
                continue
        except Exception as e:
            p(f"  - [Skip] CRA embedding failed on references for slug '{slug}': {e}")
            continue

        # Embed all generated images
        try:
            gen_embs = cra_embedder.embed(gen_imgs)  # [N, D]
        except Exception as e:
            p(f"  - [Skip] CRA embedding failed on generated images for slug '{slug}': {e}")
            continue

        if gen_embs.numel() == 0 or gen_embs.shape[0] != len(gen_imgs):
            p("  - [Skip] Empty or misaligned CRA embeddings for generated images.")
            continue

        # Similarity matrix [N, R] and per-image max similarity over references
        cra_sims_mat = gen_embs @ ref_embs.T
        cra_sims_max = cra_sims_mat.max(dim=1).values  # [N]

        ref_paths_kept_str = ";".join(str(pth) for pth in ref_paths)
        first_ref_path = str(ref_paths[0])

        appended_here = 0
        for i, (gi, gp) in enumerate(zip(gen_imgs, gen_paths)):
            total_checked += 1
            gp_str = str(gp)

            if gp_str in done:
                p(f"    - [ResumeSkip] {gp.name} already present in CSV.")
                continue

            cra_sim = float(cra_sims_max[i].item())
            passed = bool(cra_sim >= ic_tau)
            p(f"    - {gp.name}: CRA(max over {len(ref_imgs)} refs)={cra_sim:.3f} (τ={ic_tau}) → "
              f"{'PASS' if passed else 'FAIL'}")

            if not passed:
                continue

            total_cra_pass += 1

            row = {
                "mode": "dynamic",
                "model": model_name,
                "slug": slug,
                "ref_image_path": first_ref_path,
                "gen_image_path": gp_str,
                "cra_sim": float(cra_sim),
                "cra_pass": True,
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
      f"CRA-pass = {total_cra_pass} | rows appended = {total_appended}")

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute CLIP-based reference alignment (CRA) for static or dynamic settings.")
    ap.add_argument(
        "--mode",
        type=str,
        choices=["static", "dynamic"],
        required=True,
        help="Which setting to run: 'static' (first ref per slug) or 'dynamic' (all refs, max similarity)."
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
        "--ic-tau",
        type=float,
        default=None,
        help="CRA threshold. If not set, a mode-specific default is used (static: 0.70, dynamic: 0.75)."
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
        ic_tau = args.ic_tau if args.ic_tau is not None else IC_TAU_STATIC_DEFAULT
    else:
        refdir = args.refdir or REFDIR_DYNAMIC
        model_gendirs = MODEL_GENDIRS_DYNAMIC
        prompt_csv = args.prompt_csv or PROMPT_CSV_DYNAMIC
        slug_col = args.slug_col or PROMPT_SLUG_COL_DYNAMIC
        outcsv = args.outcsv or OUTCSV_DYNAMIC
        outmeta = args.outmeta or OUTMETA_DYNAMIC
        ic_tau = args.ic_tau if args.ic_tau is not None else IC_TAU_DYNAMIC_DEFAULT

    p(f"[Device] {DEVICE}")
    p(f"[Mode] {mode}")
    p(f"[CRA] Primary encoder: ST-CLIP | Fallback: Hugging Face CLIP | τ={ic_tau}")
    p(f"[Refdir] {refdir}")
    p(f"[Prompt CSV] {prompt_csv} (slug column: '{slug_col}')")
    p(f"[OutCSV] {outcsv}")
    p(f"[OutMeta] {outmeta}")

    # Ensure output CSV has header
    ensure_outcsv_header(outcsv, mode)

    # Load slug whitelist
    whitelist = read_slug_whitelist(prompt_csv, slug_col)
    p(f"[Whitelist] Loaded {len(whitelist)} slugs from '{prompt_csv.name}'")

    # Initialize CRA embedder once
    cra_embedder = ImageEmbedderCRA()

    # Prepare metadata
    meta = {
        "device": DEVICE,
        "mode": mode,
        "cra_model": {
            "primary": STCLIP_MODEL_NAME,
            "fallback": HF_CLIP_MODEL_NAME,
        },
        "thresholds": {
            "ic_tau": float(ic_tau),
        },
        "k_gen": int(args.k_gen),
        "min_gen": int(MIN_GEN),
        "refdir": str(refdir),
        "gendirs": {k: str(v) for k, v in model_gendirs.items()},
        "prompt_csv": str(prompt_csv),
        "slug_column": slug_col,
        "outcsv": str(outcsv),
        "note": "Rows are appended only for images with CRA >= ic_tau.",
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
            cra_embedder,
            whitelist,
            args.k_gen,
            ic_tau,
            outcsv,
        )
    else:
        for model_name, gendir in model_gendirs.items():
            run_fn = run_static_for_model if mode == "static" else run_dynamic_for_model
            run_fn(
                model_name,
                gendir,
                refdir,
                cra_embedder,
                whitelist,
                args.k_gen,
                ic_tau,
                outcsv,
            )

    p(f"\n[Done] CRA results written to {outcsv}")


if __name__ == "__main__":
    main()
