#!/usr/bin/env python3
"""
qe_layers.py — extract interlayer spacings from a Quantum ESPRESSO pw.x output.

Usage (CLI):
  python qe_layers.py pw.out --tol 0.30 --csv out.csv --png out.png

Exports:
  - parse_pw_last_structure(path): returns (cart_coords_A, species, cell_A)
  - layers_from_z(zvals_A, tol=0.30): returns DataFrame with layer means & spacings
"""
from __future__ import annotations
import re, sys, argparse, io
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BOHR_TO_ANG = 0.529177210903

def _last_match(pattern: str, text: str, flags=0):
    m = None
    for mm in re.finditer(pattern, text, flags):
        m = mm
    return m

def _read_text(path: Path) -> str:
    return Path(path).read_text(errors="ignore")

def _parse_alat_au(text: str) -> float | None:
    m = re.search(r"lattice parameter \(alat\)\s*=\s*([0-9.]+)\s*a\.u\.", text)
    return float(m.group(1)) if m else None

def _parse_cell_from_CELL_PARAMETERS(text: str) -> Tuple[np.ndarray|None, str|None]:
    m = _last_match(r"^CELL_PARAMETERS\s*\(([^)]+)\)\s*$", text, re.MULTILINE)
    if not m: return None, None
    unit = m.group(1).strip().lower()
    start = m.end()
    lines = text[start:].splitlines()
    rows = []
    for ln in lines[:3]:
        toks = ln.split()
        if len(toks) >= 3:
            rows.append([float(toks[0]), float(toks[1]), float(toks[2])])
    if len(rows) != 3: return None, None
    return np.array(rows, float), unit

def _parse_cell_from_crystal_axes(text: str) -> Tuple[np.ndarray|None, str|None]:
    # QE also prints a simpler block:
    # crystal axes: (cart. coord. in units of alat)
    idx = text.lower().rfind("crystal axes: (cart. coord. in units of")
    if idx == -1: return None, None
    sub = text[idx: idx+1200]
    vec_pat = re.compile(r"a\(\s*[123]\s*\)\s*=\s*\(\s*([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s+([-\d\.Ee+]+)\s*\)")
    vecs = vec_pat.findall(sub)
    if len(vecs) != 3: return None, None
    cell = np.array([[float(x) for x in v] for v in vecs], float)
    # unit is always “alat” in this block text; keep string for clarity
    return cell, "alat"

def _unitize_cell(cell: np.ndarray, unit: str, alat_au: float|None) -> np.ndarray:
    u = (unit or "").lower()
    if u.startswith("angstrom"):
        return cell
    if u.startswith("bohr") or u.startswith("a.u."):
        return cell * BOHR_TO_ANG
    if u.startswith("alat"):
        if alat_au is None:
            raise RuntimeError("CELL/axes in 'alat' but lattice parameter not found.")
        return cell * (alat_au * BOHR_TO_ANG)
    raise RuntimeError(f"Unhandled CELL_PARAMETERS/crystal axes unit: {unit}")

def _parse_last_atomic_positions(text: str) -> Tuple[str, List[Tuple[str,float,float,float]]]:
    m = _last_match(r"^ATOMIC_POSITIONS\s*\(([^)]+)\)\s*$", text, re.MULTILINE)
    if not m:
        raise RuntimeError("ATOMIC_POSITIONS block not found.")
    unit = m.group(1).strip().lower()
    lines = text[m.end():].splitlines()

    # Robustly read only the final coordinates block (QE often prints: Begin/End final coordinates)
    rows = []
    started = False
    for ln in lines:
        if ln.strip().startswith("End final coordinates"):  # QE marker
            break
        if not ln.strip():
            if started: break
            else: continue
        parts = ln.split()
        if len(parts) < 4:
            if started: break
            else: continue
        try:
            sp = parts[0]
            x, y, z = map(float, parts[1:4])
            rows.append((sp, x, y, z))
            started = True
        except ValueError:
            if started: break
            else: continue

    if not rows:
        raise RuntimeError("No atomic positions parsed from final ATOMIC_POSITIONS.")
    return unit, rows

def parse_pw_last_structure(path: str | Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """
    Returns:
      cart_coords_A: (N,3) Cartesian in Å
      species: list[str] length N
      cell_A: (3,3) cell vectors in Å (rows a1,a2,a3)
    """
    text = _read_text(Path(path))
    alat_au = _parse_alat_au(text)

    # get cell (prefer CELL_PARAMETERS; fallback to crystal axes)
    cell, unit = _parse_cell_from_CELL_PARAMETERS(text)
    if cell is None:
        cell, unit = _parse_cell_from_crystal_axes(text)
    if cell is None:
        raise RuntimeError("Could not find CELL_PARAMETERS or 'crystal axes' block.")
    cell_A = _unitize_cell(cell, unit, alat_au)

    # positions
    unit_pos, rows = _parse_last_atomic_positions(text)
    species = [r[0] for r in rows]
    pos = np.array([[r[1], r[2], r[3]] for r in rows], float)

    # convert to Cartesian Å
    up = unit_pos.lower()
    if up.startswith("angstrom"):
        cart = pos
    elif up.startswith("bohr") or up.startswith("a.u."):
        cart = pos * BOHR_TO_ANG
    elif up.startswith("crystal"):
        cart = pos @ cell_A
    elif up.startswith("alat"):
        if alat_au is None:
            raise RuntimeError("ATOMIC_POSITIONS in 'alat' but alat not found.")
        cart = pos * (alat_au * BOHR_TO_ANG)
    else:
        raise RuntimeError(f"Unhandled ATOMIC_POSITIONS unit: {unit_pos}")
    return cart, species, cell_A

def layers_from_z(zvals_A: np.ndarray, tol: float = 0.30) -> pd.DataFrame:
    """Cluster atoms by z (Å) with a tolerance, compute mean z per layer and spacing to next."""
    idx_sorted = np.argsort(zvals_A)
    z_sorted = zvals_A[idx_sorted]

    layers: List[List[int]] = []
    current = [int(idx_sorted[0])]
    for idx in idx_sorted[1:]:
        if abs(zvals_A[idx] - zvals_A[current[-1]]) <= tol:
            current.append(int(idx))
        else:
            layers.append(current)
            current = [int(idx)]
    layers.append(current)

    layer_means = np.array([zvals_A[idxs].mean() for idxs in layers])
    order = np.argsort(layer_means)
    layer_means = layer_means[order]
    layers = [layers[i] for i in order]
    interlayer = np.diff(layer_means)

    rows = []
    for i, atom_idxs in enumerate(layers, start=1):
        rows.append({
            "Layer # (bottom→top)": i,
            "Mean Z (Å)": layer_means[i-1],
            "Atoms in layer": len(atom_idxs),
            "Atom indices (0-based)": atom_idxs,
            "Spacing to next (Å)": interlayer[i-1] if i-1 < len(interlayer) else None
        })
    return pd.DataFrame(rows)

def plot_spacings(df: pd.DataFrame, png_path: str|Path|None = None):
    d = df["Spacing to next (Å)"].dropna().to_numpy()
    x = np.arange(1, len(d)+1)
    plt.figure(figsize=(6,4))
    if len(d) > 0:
        plt.plot(x, d, marker="o")
    plt.xlabel("Layer index (i)")
    plt.ylabel("d(i→i+1) (Å)")
    plt.title("Interlayer distances")
    plt.grid(True)
    plt.tight_layout()
    if png_path:
        plt.savefig(png_path, dpi=200)
    return plt.gcf()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pw_out", help="Quantum ESPRESSO pw.x output file")
    ap.add_argument("--tol", type=float, default=0.30, help="Z clustering tolerance in Å (default 0.30)")
    ap.add_argument("--csv", type=Path, default=Path("layers_z_and_spacing.csv"))
    ap.add_argument("--png", type=Path, default=Path("interlayer_distances.png"))
    args = ap.parse_args()

    cart, species, cell = parse_pw_last_structure(args.pw_out)
    z = cart[:,2]
    df = layers_from_z(z, tol=args.tol)
    df.to_csv(args.csv, index=False)
    plot_spacings(df, png_path=args.png)
    print(f"Saved: {args.csv} and {args.png}")

if __name__ == "__main__":
    main()
