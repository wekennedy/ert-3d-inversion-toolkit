# -*- coding: utf-8 -*-
"""
True3DInversionUtilities.py
----------------------------
Utilities for true 3D ERT inversion with pyGIMLi, used by
True3DInversionMain.py and True3DTimelapseInversionMain.py.

Difference from an older per-line 2D workflow (ERTutilities_3D_v2 /
ERTmain_3D_v5):
    That workflow read per-line 2D files, *reconstructed* Wenner electrode
    indices from (x-position, level), then stitched lines together by
    offsetting electrode numbers and inventing y-coordinates - a pseudo-3D
    assembly.

    This module instead goes straight from a native 3D Terrameter LS/LS2
    export to a pyGIMLi DataContainerERT:

        raw 3D export  ->  (x,y,z) per A,B,M,N  ->  global electrode table
                       ->  a/b/m/n indices      ->  DataContainerERT
                       ->  numerical/analytical k in 3D
                       ->  3D PLC + tetrahedral mesh -> ERTManager.invert()

    Electrode identity comes from the *coordinates*, so merging several
    protocol files, roll-alongs or crossing lines is just concatenation - no
    index offsets, no manual ordering tables, no y-coordinate bookkeeping.

Requires: pygimli, numpy, pandas, matplotlib, and TetGen on PATH for meshing.
scipy is optional for most of the module (KD-tree electrode clustering and
the neighbour-outlier filter both fall back to a plain grid/binning method
without it), but it is a hard requirement for the interactive QC plot, which
always applies a scipy median filter to its rolling-median guide line.
"""

__version__ = "2.0"

import os
import re
import sys
import gc
import glob
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import pygimli as pg
import pygimli.meshtools as mt
import pygimli.physics.ert as ert

import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:  # QC falls back to bin/MAD filtering
    _HAS_SCIPY = False


# =====================================================================
# ------------------------ 1. Column detection ------------------------
# =====================================================================
#
# The LS/LS2 text export changes between firmware versions and between
# "Export data" presets, so columns are matched by NAME rather than by
# position. Anything unrecognised is simply ignored.

def _norm(token):
    """Normalise a header token: 'A(x) [m]' -> 'ax', 'App.R.(Ohmm)' -> 'approhmm'."""
    return re.sub(r'[^a-z0-9]', '', str(token).strip().lower())


# electrode role aliases -> canonical role
_ROLE = {'a': 'a', 'c1': 'a', 'cur1': 'a',
         'b': 'b', 'c2': 'b', 'cur2': 'b',
         'm': 'm', 'p1': 'm', 'pot1': 'm',
         'n': 'n', 'p2': 'n', 'pot2': 'n'}

_ROLE_ALT = '|'.join(sorted(_ROLE.keys(), key=len, reverse=True))

# 'A(x)' -> ax   and the reversed convention 'X(A)' -> xa
_RE_COORD_FWD = re.compile(r'^(%s)(x|y|z)m?$' % _ROLE_ALT)
_RE_COORD_REV = re.compile(r'^(x|y|z)(%s)m?$' % _ROLE_ALT)
_RE_ELEC_ID = re.compile(r'^(%s)(no|num|id|elec|electrode)?$' % _ROLE_ALT)
# 'A(adr)' -> aadr : the Terrameter switch/relay address of that electrode
_RE_ADDR = re.compile(r'^(%s)(adr|addr|address|switch|relay)$' % _ROLE_ALT)


def _classify_column(header_token):
    """Return canonical name for a header token, or None.

    Canonical names:
        'ax','ay','az', 'bx',...            electrode coordinates
        'aadr','badr','madr','nadr'         Terrameter switch addresses
        'a','b','m','n'                     electrode ID (index) columns
        'time'  acquisition timestamp (kept as text, parsed later)
        'r'     measured resistance (Ohm)
        'rhoa'  instrument apparent resistivity (Ohm m)  - NOT used for inversion
        'err'   repeatability / variance (%)
        'i'     current (mA), 'u' voltage (V), 'k' instrument geometric factor
    """
    s = _norm(header_token)
    if not s:
        return None

    m = _RE_COORD_FWD.match(s)
    if m:
        return _ROLE[m.group(1)] + m.group(2)
    m = _RE_COORD_REV.match(s)
    if m:
        return _ROLE[m.group(2)] + m.group(1)

    if s in ('time', 'date', 'datetime', 'timestamp', 'acquired'):
        return 'time'

    m = _RE_ADDR.match(s)
    if m:
        return _ROLE[m.group(1)] + 'adr'

    # value columns (order matters: 'rhoaohmm' must hit rhoa before r)
    if re.match(r'^(rhoa|rho|appres|appr|apparentresist).*$', s):
        return 'rhoa'
    if re.match(r'^(r|res|resistance)(ohm|ohms)?$', s):
        return 'r'
    if re.match(r'^(var|err|error|stddev|std|dev|repeat).*$', s):
        return 'err'
    if re.match(r'^(i|current)(ma|a)?$', s):
        return 'i'
    if re.match(r'^(u|v|volt|voltage)(v|mv)?$', s):
        return 'u'
    if re.match(r'^(k|geomfactor|geometricfactor)(m)?$', s):
        return 'k'

    m = _RE_ELEC_ID.match(s)
    if m:
        return _ROLE[m.group(1)]

    return None


def sniff_table(filepath, max_scan=80):
    """Find the delimiter and header row of an LS/LS2 ASCII export.

    Returns (delimiter, header_row_index, colmap) where colmap maps
    canonical name -> column position.
    """
    with open(filepath, 'r', errors='replace') as f:
        lines = [next(f, '') for _ in range(max_scan)]

    best = (None, None, {}, 0)  # delim, row, colmap, score
    for delim in [',', '\t', ';', '|', None]:      # None = any whitespace
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            toks = line.split() if delim is None else line.split(delim)
            colmap = {}
            for j, t in enumerate(toks):
                c = _classify_column(t)
                if c and c not in colmap:
                    colmap[c] = j
            # score = how many canonical fields this row explains
            score = len(colmap)
            if score > best[3]:
                best = (delim, i, colmap, score)

    delim, row, colmap, score = best
    if score < 4:
        raise ValueError(
            f"Could not find a recognisable header row in {os.path.basename(filepath)}.\n"
            f"Best guess explained only {score} columns. Pass an explicit "
            f"`colmap` to read_ls_3d(), e.g. "
            f"{{'ax':6,'ay':7,'az':8,...,'r':20,'err':21}} (0-based positions).")
    return delim, row, colmap


def read_ls_3d(filepath, colmap=None, delimiter=None, header_row=None,
               verbose=True):
    """Read one native 3D Terrameter LS/LS2 export into a tidy DataFrame.

    Handles both flavours of export:
      (a) coordinate export - columns A(x),A(y),A(z),B(x)...  -> used directly
      (b) index export      - columns A,B,M,N electrode numbers -> coordinates
                              must be supplied later via `elec_lookup`

    Returns (df, colmap, mode) where mode is 'coords' or 'index'.
    """
    if colmap is None or delimiter is None or header_row is None:
        d, h, c = sniff_table(filepath)
        delimiter = delimiter if delimiter is not None else d
        header_row = header_row if header_row is not None else h
        colmap = colmap if colmap is not None else c

    sep = r'\s+' if delimiter is None else re.escape(delimiter)
    df_raw = pd.read_csv(filepath, sep=sep, skiprows=header_row + 1,
                         header=None, engine='python',
                         on_bad_lines='skip', comment=None)

    coord_keys = [r + ax for r in 'abmn' for ax in 'xyz']
    have_coords = all(k in colmap for k in ['ax', 'ay', 'mx', 'my'])
    mode = 'coords' if have_coords else 'index'

    out = {}
    for name, pos in colmap.items():
        if pos >= df_raw.shape[1]:
            continue
        if name == 'time':
            out[name] = df_raw.iloc[:, pos].astype(str)
        else:
            out[name] = pd.to_numeric(df_raw.iloc[:, pos], errors='coerce')
    df = pd.DataFrame(out)

    # z is optional in a flat survey / soil box
    if mode == 'coords':
        for r in 'abmn':
            for ax in 'xyz':
                k = r + ax
                if k not in df.columns:
                    df[k] = 0.0
        required = coord_keys
    else:
        required = [c for c in 'abmn' if c in df.columns]

    n0 = len(df)
    df = df.dropna(subset=[c for c in required if c in df.columns])
    df = df.reset_index(drop=True)

    if verbose:
        print(f"  [read] {os.path.basename(filepath)}: {len(df)} rows "
              f"({n0 - len(df)} dropped as non-numeric), mode='{mode}'")
        print(f"  [read] columns found: {sorted(colmap.keys())}")

    if len(df) == 0:
        raise ValueError(f"No numeric data rows parsed from {filepath}. "
                         f"Check delimiter/header detection.")
    return df, colmap, mode


# =====================================================================
# --------------- 2. Electrode coordinate lookup (optional) -----------
# =====================================================================

def load_elec_lookup(path):
    """Electrode-number -> (x, y, z) table.

    Accepts a CSV/TXT with columns (id, x, y, z) or (x, y, z), or an ABEM
    Spread XML. Returns a dict {electrode_number(int): np.array([x, y, z])}.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == '.xml':
        return _load_spread_xml(path)

    arr = np.genfromtxt(path, delimiter=None if _looks_whitespace(path) else ',',
                        dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] >= 4:
        return {int(r[0]): np.array(r[1:4], float) for r in arr}
    if arr.shape[1] == 3:
        return {i + 1: np.array(r, float) for i, r in enumerate(arr)}
    if arr.shape[1] == 2:                      # x,y only - assume flat
        return {i + 1: np.array([r[0], r[1], 0.0]) for i, r in enumerate(arr)}
    raise ValueError(f"Unrecognised electrode table shape {arr.shape} in {path}")


def _looks_whitespace(path):
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if line.strip():
                return ',' not in line
    return True


def _load_spread_xml(path):
    """Best-effort reader for an ABEM Spread XML.

    Walks the tree looking for elements that carry an electrode number and
    x/y/z attributes or child tags. Prints what it found so the mapping can be
    checked against the file before it is trusted.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    lookup = {}

    def _num(el, keys):
        for k in keys:
            if k in el.attrib:
                try:
                    return float(el.attrib[k])
                except ValueError:
                    pass
            child = el.find(k)
            if child is not None and child.text:
                try:
                    return float(child.text)
                except ValueError:
                    pass
        return None

    for el in root.iter():
        eid = _num(el, ['Electrode', 'ElectrodeNumber', 'Id', 'ID',
                        'Number', 'number', 'index'])
        x = _num(el, ['X', 'x', 'XCoord', 'PosX'])
        y = _num(el, ['Y', 'y', 'YCoord', 'PosY'])
        z = _num(el, ['Z', 'z', 'ZCoord', 'PosZ', 'Elevation'])
        if eid is not None and x is not None and y is not None:
            lookup[int(eid)] = np.array([x, y, z if z is not None else 0.0])

    if not lookup:
        raise ValueError(
            f"No electrode coordinates recovered from {path}. Inspect the XML "
            f"and either extend _load_spread_xml() or export a plain "
            f"'id,x,y,z' CSV instead.")
    print(f"  [xml] recovered {len(lookup)} electrode positions from "
          f"{os.path.basename(path)}")
    return lookup


# =====================================================================
# ------------------ 3. Global electrode table ------------------------
# =====================================================================

def build_electrode_table(pts, tol=0.02, sort=True):
    """Collapse an (N, 3) cloud of electrode coordinates onto unique electrodes.

    Points closer together than `tol` are treated as the same physical
    electrode. Clustering is single-linkage on a KD-tree rather than rounding
    onto a grid: a grid puts coordinates that land exactly on a cell boundary
    into neighbouring cells, which silently splits one electrode into two.

    Returns (positions (E, 3), index (N,)).
    """
    pts = np.asarray(pts, float)
    n = len(pts)

    if _HAS_SCIPY and n:
        parent = np.arange(n)

        def find(i):
            root = i
            while parent[root] != root:
                root = parent[root]
            while parent[i] != root:      # path compression
                parent[i], i = root, parent[i]
            return root

        for i, j in cKDTree(pts).query_pairs(r=tol, output_type='ndarray'):
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        roots = np.array([find(i) for i in range(n)])
        _, inv = np.unique(roots, return_inverse=True)
    else:                                  # grid fallback without scipy
        key = np.round(pts / tol).astype(np.int64)
        _, inv = np.unique(key, axis=0, return_inverse=True)

    ne = inv.max() + 1 if n else 0
    counts = np.bincount(inv, minlength=ne).astype(float)
    pos = np.zeros((ne, 3))
    for d in range(3):
        pos[:, d] = np.bincount(inv, weights=pts[:, d], minlength=ne) / counts

    if sort:
        order = np.lexsort((pos[:, 2], pos[:, 1], pos[:, 0]))
        pos = pos[order]
        remap = np.empty(len(order), dtype=np.int64)
        remap[order] = np.arange(len(order))
        inv = remap[inv]

    return pos, inv


def _split_abmn(df, mode, elec_lookup=None, tol=0.02, inf_threshold=1e4,
                verbose=True):
    """Return (positions (E,3), abmn (N,4) 0-based indices).

    Remote ('infinite') electrodes - NaN or |coord| > inf_threshold - are given
    index -1, which is how pyGIMLi encodes pole electrodes.
    """
    n = len(df)
    if mode == 'coords':
        stacks, absent = [], []
        for r in 'abmn':
            xyz = df[[r + 'x', r + 'y', r + 'z']].to_numpy(float)
            miss = ~np.isfinite(xyz).all(axis=1) | \
                   (np.abs(xyz) > inf_threshold).any(axis=1)
            xyz[miss] = 0.0
            stacks.append(xyz)
            absent.append(miss)
        allpts = np.vstack(stacks)
        keep = ~np.vstack(absent).reshape(-1)

        pos, inv_kept = build_electrode_table(allpts[keep], tol=tol)
        inv = np.full(len(allpts), -1, dtype=np.int64)
        inv[keep] = inv_kept
        abmn = inv.reshape(4, n).T

    else:  # electrode-index export
        if elec_lookup is None:
            raise ValueError(
                "This export gives electrode NUMBERS, not coordinates. Supply "
                "ELEC_LOOKUP_FILE (an 'id,x,y,z' CSV or the Spread XML) so the "
                "numbers can be placed in 3D.")
        ids = df[['a', 'b', 'm', 'n']].to_numpy()
        pts = []
        for col in range(4):
            for v in ids[:, col]:
                if not np.isfinite(v) or int(v) not in elec_lookup:
                    pts.append([np.nan] * 3)
                else:
                    pts.append(elec_lookup[int(v)])
        pts = np.asarray(pts, float)
        keep = np.isfinite(pts).all(axis=1)
        pos, inv_kept = build_electrode_table(pts[keep], tol=tol)
        inv = np.full(len(pts), -1, dtype=np.int64)
        inv[keep] = inv_kept
        abmn = inv.reshape(4, n).T

    if verbose:
        npole = int((abmn < 0).sum())
        print(f"  [geom] {len(pos)} unique electrodes from {4 * n} coordinate "
              f"entries (tol = {tol} m)" +
              (f", {npole} remote/pole entries" if npole else ""))
    return pos, abmn


# =====================================================================
# ------------------ 4. The Survey object -----------------------------
# =====================================================================
#
# Everything between reading the file and building the mesh is held in plain
# numpy arrays inside a Survey. Measurements are never deleted - they are
# flagged in `valid`, so any rejection can be undone, inspected or saved and
# reloaded. A pyGIMLi DataContainerERT is built once, at the end, from the
# surviving data only.
#
# This is deliberately different from mutating a DataContainerERT in place.
# pyGIMLi's DataContainer.remove() has version-dependent semantics for whether
# its argument is a boolean mask or an index list, and getting it wrong silently
# empties the container.

def _writable(a, dtype=float):
    """A writable, owned copy of `a`.

    pandas 2.x returns read-only views from .to_numpy() under copy-on-write.
    Storing one of those in a Survey makes any later in-place update raise
    "assignment destination is read-only", so everything is copied on entry.
    """
    if a is None:
        return None
    arr = np.array(a, dtype=dtype, copy=True)
    arr.setflags(write=True)
    return arr


class Survey:
    """A 3D ERT dataset as numpy arrays, with a reversible validity mask."""

    FIELDS = ('r', 'rhoa', 'k', 'err', 'i', 'u', 'rhoa_instrument')

    def __init__(self, pos, abmn, src=None, **fields):
        self.pos = _writable(pos)                  # (E, 3) electrode positions
        self.abmn = _writable(abmn, np.int64)      # (N, 4) 0-based, -1 = remote
        n = len(self.abmn)
        self.src = np.array(src if src is not None else ['?'] * n,
                            dtype=object, copy=True)
        for f in self.FIELDS:
            setattr(self, f, _writable(fields.get(f)))
        self.valid = np.ones(n, bool)
        self.acquired = None        # median acquisition time, if the export has one
        self.acquired_span = None   # (first, last) reading

    # ---- basic properties ------------------------------------------
    @property
    def n_data(self):
        return len(self.abmn)

    @property
    def n_valid(self):
        return int(self.valid.sum())

    @property
    def n_elec(self):
        return len(self.pos)

    def has(self, field):
        return getattr(self, field, None) is not None

    def __repr__(self):
        return (f"<Survey {self.n_valid}/{self.n_data} measurements, "
                f"{self.n_elec} electrodes>")

    # ---- geometry helpers ------------------------------------------
    def elec_xyz(self, role):
        """(N, 3) positions of the 'a'/'b'/'m'/'n' electrode of each datum."""
        j = 'abmn'.index(role)
        idx = self.abmn[:, j]
        out = np.full((self.n_data, 3), np.nan)
        ok = idx >= 0
        out[ok] = self.pos[idx[ok]]
        return out

    def centres(self):
        """Mean position of the electrodes of each measurement (N, 3)."""
        acc = np.zeros((self.n_data, 3))
        cnt = np.zeros((self.n_data, 1))
        for r in 'abmn':
            p = self.elec_xyz(r)
            ok = np.isfinite(p).all(axis=1)
            acc[ok] += p[ok]
            cnt[ok, 0] += 1
        cnt[cnt == 0] = 1
        return acc / cnt

    def spacings(self):
        """Largest electrode separation within each measurement (N,)."""
        pts = np.stack([self.elec_xyz(r) for r in 'abmn'], axis=1)  # (N,4,3)
        d = np.linalg.norm(pts[:, :, None, :] - pts[:, None, :, :], axis=-1)
        return np.nanmax(d.reshape(self.n_data, -1), axis=1)

    def principal_axes(self):
        """Right-handed frame aligned with the electrode layout."""
        p = self.pos - self.pos.mean(axis=0)
        _, _, vt = np.linalg.svd(p, full_matrices=True)
        return vt                                    # rows: along, perp1, perp2

    def line_key(self, decimals=2):
        """Label identifying which line/offset each measurement sits on."""
        ax = self.principal_axes()
        c = (self.centres() - self.pos.mean(axis=0)) @ ax.T
        return [tuple(np.round(row[1:], decimals)) for row in c]

    def along(self):
        """Distance of each measurement centre along the survey's main axis."""
        ax = self.principal_axes()
        c = (self.centres() - self.pos.mean(axis=0)) @ ax.T
        return c[:, 0]

    # ---- mask handling ---------------------------------------------
    def reject(self, mask, reason='', verbose=True):
        """Flag measurements invalid. Reversible - nothing is deleted."""
        mask = np.asarray(mask, bool)
        new = mask & self.valid
        self.valid &= ~mask
        if verbose and new.any():
            print(f"  [qc] flagged {int(new.sum()):6d}  ({reason})")
        return int(new.sum())

    def take(self, idx):
        """Keep and reorder rows by index. Used to align timesteps."""
        idx = np.asarray(idx, np.int64)
        self.abmn = self.abmn[idx]
        self.src = self.src[idx]
        self.valid = self.valid[idx]
        for f in self.FIELDS:
            if self.has(f):
                setattr(self, f, getattr(self, f)[idx])
        return self

    def accept_all(self):
        self.valid[:] = True
        return self

    def save_mask(self, path):
        np.savetxt(path, self.valid.astype(int), fmt='%d')
        print(f"  [qc] validity mask saved to {path}")

    def load_mask(self, path):
        m = np.loadtxt(path).astype(bool)
        if len(m) != self.n_data:
            raise ValueError(f"mask has {len(m)} entries, survey has "
                             f"{self.n_data}")
        self.valid = m
        print(f"  [qc] mask loaded: {self.n_valid}/{self.n_data} kept")
        return self

    # ---- export ----------------------------------------------------
    def to_container(self):
        """Build a pyGIMLi DataContainerERT from the valid measurements."""
        if self.n_valid == 0:
            raise ValueError("No valid measurements left - nothing to invert.")
        abmn = self.abmn[self.valid]

        used = np.unique(abmn[abmn >= 0])
        remap = np.full(self.n_elec, -1, np.int64)
        remap[used] = np.arange(len(used))
        safe = np.where(abmn >= 0, abmn, 0)
        new = np.where(abmn >= 0, remap[safe], -1)

        data = pg.DataContainerERT()
        for p in self.pos[used]:
            data.createSensor(pg.Pos(float(p[0]), float(p[1]), float(p[2])))
        data.resize(len(new))
        for j, tok in enumerate('abmn'):
            try:
                data.registerSensorIndex(tok)
            except Exception:
                pass
            data[tok] = np.asarray(new[:, j], dtype=float)

        for f in ('r', 'rhoa', 'k', 'err', 'i', 'u'):
            if self.has(f):
                data[f] = getattr(self, f)[self.valid]
        data['valid'] = np.ones(len(new))
        return data

    def save(self, path):
        data = self.to_container()
        data.save(path)
        print(f"  [save] {self.n_valid} measurements, {data.sensorCount()} "
              f"electrodes -> {path}")
        return data

    # ---- reporting -------------------------------------------------
    def summary(self):
        p = self.pos
        ext = p.max(axis=0) - p.min(axis=0)
        print("\n  ---- survey summary ----")
        print(f"  electrodes      : {self.n_elec}")
        print(f"  measurements    : {self.n_valid} valid of {self.n_data}")
        for d, name in enumerate('xyz'):
            print(f"  {name} extent        : {p[:,d].min():.3f} .. "
                  f"{p[:,d].max():.3f} m  ({ext[d]:.3f} m)")
        if _HAS_SCIPY and self.n_elec > 1:
            dmin = cKDTree(p).query(p, k=2)[0][:, 1]
            print(f"  nearest-neighbour spacing: min {dmin.min():.3f} m, "
                  f"median {np.median(dmin):.3f} m")
        sp = self.spacings()
        print(f"  configuration size: {np.nanmin(sp):.3f} .. "
              f"{np.nanmax(sp):.3f} m")
        srcs, counts = np.unique(self.src, return_counts=True)
        if len(srcs) > 1:
            print("  per file        :")
            for s, c in zip(srcs, counts):
                print(f"      {s}: {c}")
        print("  ------------------------\n")
        return self


# =====================================================================
# ------------------ 5. Loading into a Survey -------------------------
# =====================================================================

def load_3d_survey(files, elec_lookup_file=None, tol=0.02, colmap=None,
                   err_is_percent=True, transforms=None, coordinate_scale=1.0,
                   layout=None, verbose=True):
    """Read one or many native 3D exports and merge them into one Survey.

    Merging is by COORDINATE, so protocol files, roll-alongs and crossing lines
    can be listed in any order and shared electrodes are recognised
    automatically.

    coordinate_scale multiplies every electrode coordinate. Use it when the
    spread definition in the instrument did not match the physical layout - for
    example scale=2.0 if electrodes were planted every 0.5 m but the spread was
    set up as 0.25 m. This changes k, and therefore rhoa, proportionally.

    transforms repositions individual files before merging, keyed by basename:
        {'line_B.txt': dict(offset=(0, 1.5, 0)),
         'line_C.txt': dict(rotation_deg=90, offset=(3, 0, 0))}
    """
    if isinstance(files, str):
        files = sorted(glob.glob(files)) if any(c in files for c in '*?[') \
            else [files]
    if not files:
        raise ValueError("No input data files given.")

    elec_lookup = load_elec_lookup(elec_lookup_file) if elec_lookup_file else None
    transforms = transforms or {}

    frames, mode = [], None
    for f in files:
        df, cmap, m = read_ls_3d(f, colmap=colmap, verbose=verbose)
        mode = m if mode is None else mode
        if m != mode:
            raise ValueError("Mixed coordinate and index exports in one run - "
                             "process them separately.")
        base = os.path.basename(f)
        if mode == 'coords':
            if layout:
                df = apply_electrode_layout(df, layout, verbose=verbose)
            if coordinate_scale != 1.0:
                for r in 'abmn':
                    for ax in 'xyz':
                        df[r + ax] = df[r + ax].to_numpy(float) * coordinate_scale
            if base in transforms:
                df = apply_transform(df, **transforms[base])
                if verbose:
                    print(f"  [xform] applied {transforms[base]} to {base}")
        df['_src'] = base
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    if verbose and coordinate_scale != 1.0:
        print(f"  [scale] all coordinates multiplied by {coordinate_scale}")

    if mode == 'coords':
        report_address_blocks(df, verbose=verbose)
        report_addresses(df, tol=tol, verbose=verbose)
        _warn_identical_files(df, verbose=verbose)

    pos, abmn = _split_abmn(df, mode, elec_lookup=elec_lookup, tol=tol,
                            verbose=verbose)

    # measured resistance is the primary observable
    if 'r' in df:
        r = df['r'].to_numpy(float)
    elif 'rhoa' in df and 'k' in df:
        r = df['rhoa'].to_numpy(float) / df['k'].to_numpy(float)
    elif 'u' in df and 'i' in df:
        r = df['u'].to_numpy(float) / (df['i'].to_numpy(float) * 1e-3)
        if verbose:
            print("  [note] resistance derived from U/I")
    else:
        raise ValueError(
            "No resistance column found. The inversion needs measured R (Ohm); "
            "apparent resistivity from the instrument assumes a collinear "
            "geometric factor and must not be used for a true 3D layout.")

    err = None
    if 'err' in df:
        err = np.abs(df['err'].to_numpy(float))
        if err_is_percent:
            err = err / 100.0

    sv_time = _parse_acquisition(df, verbose=verbose)

    sv = Survey(pos, abmn, src=df['_src'].to_numpy(object), r=r, err=err,
                rhoa_instrument=df['rhoa'].to_numpy(float) if 'rhoa' in df else None,
                i=df['i'].to_numpy(float) if 'i' in df else None,
                u=df['u'].to_numpy(float) if 'u' in df else None)
    sv.acquired, sv.acquired_span = sv_time
    if verbose:
        sv.summary()
    return sv


def _parse_acquisition(df, verbose=True):
    """Median acquisition time and (first, last) from the export's Time column."""
    if 'time' not in df.columns:
        return None, None
    try:
        t = pd.to_datetime(df['time'], errors='coerce', format='mixed')
    except (TypeError, ValueError):
        t = pd.to_datetime(df['time'], errors='coerce')
    t = t.dropna()
    if t.empty:
        return None, None
    med = t.median().to_pydatetime()
    span = (t.min().to_pydatetime(), t.max().to_pydatetime())
    if verbose:
        mins = (span[1] - span[0]).total_seconds() / 60.0
        print(f"  [time] acquired {span[0]:%Y-%m-%d %H:%M} to "
              f"{span[1]:%H:%M} ({mins:.0f} min), median "
              f"{med:%Y-%m-%d %H:%M}")
    return med, span


def report_addresses(df, tol=0.02, verbose=True):
    """Report how switch addresses map onto positions, per source file.

    Within one spread the mapping must be one-to-one. When it is not, the usual
    causes are (a) two different spreads merged, or (b) the instrument recorded
    along-cable coordinates so that separate lines came back with identical
    coordinates. The per-file breakdown below tells the two apart.
    """
    if not all(f'{r}adr' in df.columns for r in 'abmn'):
        return None
    if not all(f'{r}x' in df.columns for r in 'abmn'):
        return None

    rows = []
    for r in 'abmn':
        rows.append(pd.DataFrame({
            'src': df['_src'] if '_src' in df else '?',
            'adr': df[f'{r}adr'].to_numpy(),
            'x': df[f'{r}x'].to_numpy(float),
            'y': df[f'{r}y'].to_numpy(float),
            'z': df[f'{r}z'].to_numpy(float),
            'kx': np.round(df[f'{r}x'].to_numpy(float) / tol),
            'ky': np.round(df[f'{r}y'].to_numpy(float) / tol),
            'kz': np.round(df[f'{r}z'].to_numpy(float) / tol)}))
    E = pd.concat(rows, ignore_index=True).dropna().drop_duplicates(
        subset=['src', 'adr', 'kx', 'ky', 'kz'])

    if verbose:
        print("  [addr] per file:")
        for src, g in E.groupby('src'):
            ys = np.sort(g.y.unique())
            print(f"      {src}: {len(g)} electrodes, addresses "
                  f"{int(g.adr.min())}-{int(g.adr.max())}, "
                  f"x {g.x.min():.2f}..{g.x.max():.2f}, "
                  f"y {' '.join(f'{v:.2f}' for v in ys)}")

    per_pos = E.groupby(['kx', 'ky', 'kz']).adr.nunique()
    per_adr = E.groupby('adr')[['kx', 'ky', 'kz']].nunique().max(axis=1)
    ok = (per_pos.max() <= 1) and (per_adr.max() <= 1)

    if verbose and not ok:
        print(f"  !! [addr] {int((per_pos > 1).sum())} positions carry more "
              f"than one address and {int((per_adr > 1).sum())} addresses "
              f"appear at more than one position.")
        print("     If the files above use DIFFERENT address ranges over the "
              "SAME coordinates, they are different physical lines that the "
              "instrument logged in local coordinates. Separate them with "
              "`transforms` before merging, or they will be stacked on top of "
              "each other.")
    elif verbose:
        print(f"  [addr] address <-> position mapping is one-to-one "
              f"({len(E)} electrodes)")
    return ok


def _warn_identical_files(df, verbose=True):
    """Warn when two source files occupy exactly the same electrode footprint."""
    if '_src' not in df.columns or df['_src'].nunique() < 2:
        return
    foot = {}
    for src, g in df.groupby('_src'):
        pts = np.vstack([g[[r + 'x', r + 'y', r + 'z']].to_numpy(float)
                         for r in 'abmn'])
        foot[src] = frozenset(map(tuple, np.round(pts, 3)))
    srcs = list(foot)
    clashes = [(a, b) for i, a in enumerate(srcs) for b in srcs[i + 1:]
               if foot[a] == foot[b]]
    if clashes and verbose:
        print("  !! [merge] identical electrode footprints:")
        for a, b in clashes[:5]:
            print(f"       {a}  ==  {b}")


def apply_electrode_layout(df, layout, verbose=True):
    """Overwrite electrode coordinates from the switch addresses.

    Use this when the instrument's spread definition did not describe the real
    layout - the classic case being a multi-line 3D survey where each cable was
    set up in its own local coordinates, so several lines come back sharing the
    same x and y.

    The switch address is the one thing in the export that is always reliable:
    it identifies the physical take-out. `layout` maps an inclusive address
    range onto the positions of its first and last electrode, and everything in
    between is interpolated. This handles line offsets, spacing and direction
    (including boustrophedon runs, where the address order reverses) in one
    statement:

        layout = {
            (1, 16):  ((0.25, 0.00, 0.0), (4.00, 0.00, 0.0)),   # forward
            (17, 32): ((4.00, 0.25, 0.0), (0.25, 0.25, 0.0)),   # reversed
            (33, 48): ((0.25, 0.50, 0.0), (4.00, 0.50, 0.0)),
            (49, 64): ((4.00, 0.75, 0.0), (0.25, 0.75, 0.0)),
        }
    """
    if not all(f'{r}adr' in df.columns for r in 'abmn'):
        raise ValueError("This export has no address columns, so the layout "
                         "cannot be applied by address.")

    table = {}
    for (lo, hi), (p0, p1) in layout.items():
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        span = max(hi - lo, 1)
        for a in range(int(lo), int(hi) + 1):
            table[a] = p0 + (a - lo) / span * (p1 - p0)

    covered, missing, worst = 0, set(), 0.0
    for r in 'abmn':
        adr = df[f'{r}adr'].to_numpy()
        old = df[[f'{r}x', f'{r}y', f'{r}z']].to_numpy(float)
        new = old.copy()
        for i, a in enumerate(adr):
            if not np.isfinite(a):
                continue
            p = table.get(int(a))
            if p is None:
                missing.add(int(a))
            else:
                new[i] = p
                covered += 1
                worst = max(worst, float(np.abs(new[i] - old[i]).max()))
        df[f'{r}x'], df[f'{r}y'], df[f'{r}z'] = new[:, 0], new[:, 1], new[:, 2]

    if verbose:
        print(f"  [layout] repositioned {covered} electrode entries from "
              f"{len(table)} defined addresses")
        print(f"  [layout] largest change from the recorded coordinates: "
              f"{worst:.3f} m")
        if missing:
            print(f"  !! [layout] no layout entry for addresses "
                  f"{sorted(missing)} - those coordinates were left as recorded")
    return df


def make_line_layout(runs, elec_spacing, line_spacing, origin=(0.0, 0.0, 0.0),
                     along='x', across='y', verbose=True):
    """Build an ELECTRODE_LAYOUT for a regular grid of parallel lines.

    `runs` lists the address at each END of each line, in the order you walk
    along it, one tuple per line, in the order the lines are offset. It is the
    notation you would write on a field sheet:

        runs = [(1, 16), (32, 17), (33, 48), (64, 49)]

    means line 1 runs 1->16 along the line, line 2 runs 32->17 (the cable
    doubled back, so addresses count down), line 3 runs 33->48, line 4 runs
    64->49. Boustrophedon layouts need no special handling: reversing a tuple
    reverses that line.

    Every line starts at the same `along` coordinate and is offset from its
    neighbour by `line_spacing` in the `across` direction.
    """
    ax = {'x': 0, 'y': 1, 'z': 2}
    ia, ic = ax[along], ax[across]
    layout = {}
    for line_no, (first, last) in enumerate(runs):
        n = abs(int(last) - int(first)) + 1
        p_first = np.array(origin, float)
        p_first[ic] += line_no * line_spacing
        p_last = p_first.copy()
        p_last[ia] += (n - 1) * elec_spacing
        lo, hi = min(int(first), int(last)), max(int(first), int(last))
        # key is (low address, high address); value is their positions in that order
        layout[(lo, hi)] = (tuple(p_first), tuple(p_last)) if first < last \
            else (tuple(p_last), tuple(p_first))
    if verbose:
        print(f"  [layout] {len(runs)} lines x "
              f"{abs(runs[0][1] - runs[0][0]) + 1} electrodes, "
              f"{elec_spacing} m along, {line_spacing} m between lines")
        for (lo, hi), (p0, p1) in sorted(layout.items()):
            d = "forward" if p1[ia] > p0[ia] else "reversed"
            print(f"      adr {lo:>3d}-{hi:<3d}  {along}="
                  f"{min(p0[ia], p1[ia]):.2f}..{max(p0[ia], p1[ia]):.2f}  "
                  f"{across}={p0[ic]:.2f}  ({d} with address)")
    return layout


def report_address_blocks(df, verbose=True):
    """Print how each contiguous block of switch addresses is laid out.

    Reveals line offsets, electrode spacing and the direction each cable runs,
    which is how you confirm a boustrophedon layout matches what you planted.
    """
    if not all(f'{r}adr' in df.columns for r in 'abmn'):
        return None
    rows = []
    for r in 'abmn':
        rows.append(pd.DataFrame({
            'src': df['_src'] if '_src' in df else '?',
            'adr': df[f'{r}adr'].to_numpy(),
            'x': df[f'{r}x'].to_numpy(float),
            'y': df[f'{r}y'].to_numpy(float),
            'z': df[f'{r}z'].to_numpy(float)}))
    E = pd.concat(rows, ignore_index=True).dropna().drop_duplicates()

    out = []
    for src, g in E.groupby('src'):
        g = g.sort_values('adr')
        # split where the line (y, z) changes or the addresses jump
        brk = ((g.y.diff().abs() > 1e-6) | (g.z.diff().abs() > 1e-6) |
               (g.adr.diff() > 1)).to_numpy().copy()
        brk[0] = True
        block = np.cumsum(brk)
        for _, b in g.groupby(block):
            step = np.diff(np.sort(b.x.unique()))
            out.append(dict(src=src, lo=int(b.adr.min()), hi=int(b.adr.max()),
                            n=len(b), y=float(b.y.iloc[0]),
                            x0=float(b.x.iloc[0]), x1=float(b.x.iloc[-1]),
                            step=float(np.median(step)) if len(step) else 0.0))
    if verbose:
        print("  [addr] address blocks:")
        for o in out:
            direction = "forward" if o['x1'] > o['x0'] else "reversed"
            print(f"      {o['src']}  adr {o['lo']:>3d}-{o['hi']:<3d} "
                  f"{o['n']:2d} electrodes  y={o['y']:.2f}  "
                  f"x {o['x0']:.2f} -> {o['x1']:.2f}  "
                  f"step {o['step']:.3f} m  ({direction})")
        ys = sorted({o['y'] for o in out})
        print(f"  [addr] {len(out)} line(s) across {len(ys)} distinct y "
              f"offset(s): {', '.join(f'{v:.2f}' for v in ys)}")
    return out


def apply_transform(df, offset=(0.0, 0.0, 0.0), rotation_deg=0.0,
                    origin=(0.0, 0.0)):
    """Rotate about `origin` then translate every electrode coordinate."""
    th = np.deg2rad(rotation_deg)
    ct, st = np.cos(th), np.sin(th)
    ox, oy = origin
    for r in 'abmn':
        x = df[f'{r}x'].to_numpy(float) - ox
        y = df[f'{r}y'].to_numpy(float) - oy
        df[f'{r}x'] = ox + ct * x - st * y + offset[0]
        df[f'{r}y'] = oy + st * x + ct * y + offset[1]
        df[f'{r}z'] = df[f'{r}z'].to_numpy(float) + offset[2]
    return df


def check_3d_coupling(survey, verbose=True):
    """Fraction of measurements using non-collinear electrode configurations.

    A measurement whose four electrodes lie on one straight line carries no
    information about resistivity variation across that line.
    """
    n_cross = 0
    for row in survey.abmn:
        q = survey.pos[row[row >= 0]]
        if len(q) < 3:
            continue
        q = q - q.mean(axis=0)
        sv = np.linalg.svd(q, compute_uv=False)
        if sv[0] > 0 and sv[1] / sv[0] > 1e-3:
            n_cross += 1

    frac = n_cross / max(survey.n_data, 1)
    if verbose:
        print(f"  [3D] {n_cross} / {survey.n_data} measurements "
              f"({100 * frac:.1f}%) use non-collinear configurations")
        if n_cross == 0:
            print("  !! [3D] every measurement lies within a single straight "
                  "line. A 3D inversion will interpolate between the lines "
                  "rather than resolve between them. Cross-line configurations "
                  "in the protocol are what buy genuine 3D resolution.")
        elif frac < 0.1:
            print("  !  [3D] few cross-line configurations - expect weak "
                  "resolution perpendicular to the lines.")
    return frac


# =====================================================================
# ------------------ 6. Geometric factors -----------------------------
# =====================================================================

def compute_geometric_factors(survey, numerical=False, mesh=None, verbose=True):
    """Compute k for the real 3D layout and set rhoa = k * R.

    numerical=False : analytical half-space solution, dim=3. Correct for a flat
                      surface and the right default for level ground.
    numerical=True  : k from a forward run on a homogeneous mesh. Needed when
                      topography is a significant fraction of the spacing.
    """
    tmp = Survey(survey.pos, survey.abmn, r=survey.r).to_container()
    if numerical:
        if verbose:
            print("  [k] numerical geometric factors (slow)...")
        k = np.asarray(ert.createGeometricFactors(tmp, numerical=True,
                                                  mesh=mesh, verbose=verbose))
    else:
        k = np.asarray(ert.createGeometricFactors(tmp, dim=3))

    survey.k = k
    survey.rhoa = k * survey.r

    if verbose:
        ka = np.abs(k)
        print(f"  [k] |k| range {ka.min():.4g} .. {ka.max():.4g} m, "
              f"median {np.median(ka):.4g} m")
        if survey.has('rhoa_instrument'):
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio = survey.rhoa / survey.rhoa_instrument
            ratio = ratio[np.isfinite(ratio)]
            if len(ratio):
                print(f"  [k] rhoa(3D)/rhoa(instrument): median "
                      f"{np.median(ratio):.4f}")
    return survey


# =====================================================================
# ------------------ 7. Reciprocals (optional) ------------------------
# =====================================================================

def reciprocal_errors(survey, tol=1e-12, verbose=True):
    """Pair normal and reciprocal measurements, if any exist.

    (a,b,m,n) is reciprocal to (m,n,a,b). Returns {'pairs', 'errors'}; both
    empty if the protocol carries no reciprocals, which is not an error.
    """
    a, b, m, n = survey.abmn.T
    r = survey.r

    key = {}
    for i in range(len(a)):
        cur = tuple(sorted((int(a[i]), int(b[i]))))
        pot = tuple(sorted((int(m[i]), int(n[i]))))
        key.setdefault((cur, pot), []).append(i)

    pairs, errs = [], []
    for (cur, pot), idx in key.items():
        rec = key.get((pot, cur))
        if rec is None or cur >= pot:
            continue
        rn, rr = np.mean(r[idx]), np.mean(r[rec])
        denom = 0.5 * abs(rn + rr)
        if denom > tol:
            pairs.append((idx[0], rec[0]))
            errs.append(abs(rn - rr) / denom)

    errs = np.asarray(errs, float)
    if verbose:
        if len(errs):
            print(f"  [rec] {len(errs)} reciprocal pairs, median error "
                  f"{100*np.median(errs):.2f}%, 95th pct "
                  f"{100*np.percentile(errs, 95):.2f}%")
        else:
            print("  [rec] no reciprocal pairs in this dataset "
                  "(not a problem - errors will come from the noise model)")
    return {'pairs': pairs, 'errors': errs}


def fit_reciprocal_error_model(survey, rel=True, verbose=True):
    """Fit pyGIMLi's reciprocal error model and return a per-datum rel. error.

    This is a thin wrapper around pygimli.physics.ert.fitReciprocalErrorModel,
    which is pyGIMLi's own reciprocal error calculator. It pairs every normal
    reading with its reciprocal, bins the pairs by |R|, takes the standard
    deviation within each bin, and fits a straight line through those - the
    LaBrecque (1996) / Koestel (2008) model

        dR = a + b*|R|          (rel=False, the pyGIMLi default), or
        dR/R = a + b/|R|        (rel=True, fits the *relative* errors directly)

    The fitted straight line is a far better error estimate than any single
    pair's own disagreement, which is what the old per-pair path in
    reciprocal_errors() / stack_reciprocals() used and which tends to leave
    chi^2 too high because lucky-agreeing pairs get unrealistically tiny errors.

    Because the model is a smooth function of |R|, the relative error is defined
    for EVERY measurement, so the return array has length survey.n_data and can
    be dropped straight into survey.err.

        rel=False -> rel_err = b + a/|R|      (b is the high-signal % error)
        rel=True  -> rel_err = a + b/|R|      (a is the high-signal % error)

    Returns (a, b, rel_err, ok). If no reciprocal pairs are found, ok is False
    and rel_err is all-NaN so the caller can fall back to the noise model.
    """
    R = np.asarray(survey.r, float)

    # pyGIMLi works on a DataContainerERT; to_container() gives us one built
    # from the valid rows, with a/b/m/n indices registered and 'r' populated,
    # which is exactly what fitReciprocalErrorModel needs to find the pairs.
    data = survey.to_container()
    if not data.haveData('r') or not np.any(np.abs(np.asarray(data['r']))):
        data['r'] = np.asarray(survey.r[survey.valid], float)

    # how many N/R pairs does pyGIMLi actually see?
    try:
        iF, iB = ert.reciprocalIndices(data, True)
    except Exception as exc:
        if verbose:
            print(f"  [rec] fitReciprocalErrorModel: pairing failed ({exc})")
        return None, None, np.full(survey.n_data, np.nan), False

    if len(iF) == 0:
        if verbose:
            print("  [rec] fitReciprocalErrorModel: no reciprocal pairs found "
                  "(errors will come from the noise model)")
        return None, None, np.full(survey.n_data, np.nan), False

    # the fit itself. show=False returns just the [a, b] array on current
    # pyGIMLi, but older versions hand back (ab, axis); handle both.
    res = ert.fitReciprocalErrorModel(data, rel=rel, show=False)
    ab = res[0] if isinstance(res, tuple) else res
    a, b = float(ab[0]), float(ab[1])

    # evaluate the model for every datum in the survey
    absR = np.abs(R)
    absR[absR < 1e-12] = np.nan
    if rel:
        rel_err = a + b / absR          # dR/R = a + b/|R|
    else:
        rel_err = b + a / absR          # (dR = a + b|R|) / |R|

    if verbose:
        print(f"  [rec] pygimli fitReciprocalErrorModel: {len(iF)} pairs, "
              f"a={a:.4g}, b={b:.4g}  (rel={'yes' if rel else 'no'})")
        hi = a if rel else b            # the high-signal asymptote
        print(f"        high-signal relative error ~ {100*hi:.2f}%")
        e = rel_err[np.isfinite(rel_err)]
        if len(e):
            print(f"        per-datum relative error: median "
                  f"{100*np.median(e):.2f}%, 90th pct "
                  f"{100*np.percentile(e, 90):.2f}%, max {100*e.max():.2f}%")
    return a, b, rel_err, True


def stack_reciprocals(survey, keep_err=False, verbose=True):
    """Average each reciprocal pair into one datum carrying its own error.

    Silently does nothing if there are no reciprocals.

    keep_err=False : the stacked datum takes the pair's own normal-vs-reciprocal
                     disagreement as its error (the original behaviour).
    keep_err=True  : the stacked datum keeps whatever error is already on
                     survey.err[i]. Use this when the error has already been set
                     by a fitted model (fit_reciprocal_error_model), so stacking
                     does not clobber the smooth model with the noisy per-pair
                     value. The two pair members have near-identical |R|, so the
                     model error is effectively unchanged by the averaging.
    """
    rec = reciprocal_errors(survey, verbose=False)
    if not rec['pairs']:
        if verbose:
            print("  [rec] no reciprocal pairs to stack - data left as is")
        return survey

    survey.err = _writable(survey.err) if survey.err is not None \
        else np.full(survey.n_data, 0.03)
    survey.r = _writable(survey.r)
    drop = np.zeros(survey.n_data, bool)
    for (i, j), e in zip(rec['pairs'], rec['errors']):
        survey.r[i] = 0.5 * (survey.r[i] + survey.r[j])
        if not keep_err:
            survey.err[i] = e
        drop[j] = True
    if survey.has('k'):
        survey.rhoa = survey.k * survey.r
    survey.reject(drop, "reciprocal partner (stacked)", verbose=False)
    if verbose:
        kept = "kept fitted model error" if keep_err else "per-pair error"
        print(f"  [rec] stacked {len(rec['pairs'])} pairs ({kept}) -> "
              f"{survey.n_valid} measurements")
    return survey


# =====================================================================
# ------------------ 8. Error model -----------------------------------
# =====================================================================

def set_error_model(survey, relative=0.03, absolute_u=1e-4,
                    use_reciprocals='auto', use_instrument=False,
                    floor=0.01, reciprocal_rel_fit=True, verbose=True):
    """Set the relative error that weights the inversion.

    This error becomes data['err'] in the container, so it is exactly what the
    inversion uses to compute chi^2. Getting it right is the main lever on
    chi^2: too-small errors force the inversion to over-fit and leave chi^2 well
    above 1, which is the usual cause of a stubborn chi^2 ~ 10.

    use_reciprocals controls where the error comes from:

      'model' : fit pyGIMLi's reciprocal error model (a + b|R|) with
                fit_reciprocal_error_model() and USE THAT as the data error.
                This is the recommended setting when the protocol carries
                reciprocals - the smooth fitted model is a better error estimate
                than the flat `relative` value or any single pair's disagreement.
                Falls back to the noise model below only where the fit cannot
                produce a finite value, or if no reciprocals are found.

      'auto' / True : the older per-pair path - take the max of the noise model
                and each pair's own normal-vs-reciprocal disagreement.

      False   : noise model only.

    Noise model (used as the base, and as the fallback for 'model'):
        err = max( floor, sqrt(relative^2 + (absolute_u/U)^2),
                   [instrument Var%] )

    reciprocal_rel_fit chooses how the model is fitted: True fits the relative
    errors directly (dR/R = a + b/|R|, usually more realistic across a wide R
    range), False fits the absolute residuals (dR = a + b|R|).

    The LS 'Var %' column is the variance between stacks within one
    measurement. It reflects short-term noise only, so it is off by default.
    """
    n = survey.n_data
    err = np.full(n, float(relative))

    if survey.has('u') and absolute_u:
        u = np.abs(survey.u).astype(float)
        u[u <= 0] = np.nan
        e2 = np.sqrt(relative ** 2 + (absolute_u / u) ** 2)
        err = np.where(np.isfinite(e2), e2, relative)

    if use_reciprocals == 'model':
        # pyGIMLi's fitted reciprocal error model IS the data error here.
        a, b, rel_err, ok = fit_reciprocal_error_model(
            survey, rel=reciprocal_rel_fit, verbose=verbose)
        if ok:
            # use the fitted model everywhere it is finite; keep the noise
            # model only as a fallback for any datum the model could not score.
            err = np.where(np.isfinite(rel_err), rel_err, err)
        elif verbose:
            print("  [err] no reciprocals for the fitted model - falling back "
                  "to the noise model alone")
    elif use_reciprocals:
        rec = reciprocal_errors(survey, verbose=verbose)
        if rec['pairs']:
            for (i, j), e in zip(rec['pairs'], rec['errors']):
                err[i] = max(err[i], e)
                err[j] = max(err[j], e)
        elif use_reciprocals is True and verbose:
            print("  [err] reciprocals requested but none found - using the "
                  "noise model alone")

    if use_instrument and survey.has('err'):
        err = np.maximum(err, survey.err)

    survey.err = np.maximum(err, floor)
    if verbose:
        e = survey.err[survey.valid]
        if len(e):
            print(f"  [err] relative error: median {100*np.median(e):.2f}%, "
                  f"90th pct {100*np.percentile(e, 90):.2f}%, "
                  f"max {100*e.max():.2f}%")
        else:
            print("  [err] no valid measurements to summarise")
    return survey


# =====================================================================
# ------------------ 9. Automatic filters (all opt-in) ----------------
# =====================================================================

def apply_filters(survey, min_rhoa=None, max_rhoa=None, k_max=None,
                  max_instrument_err=None, max_reciprocal_err=None,
                  min_u=None, min_i=None, neighbour_thr=None,
                  n_neighbours=10, verbose=True):
    """Optional automatic rejection. Every criterion defaults to OFF.

    Only non-finite and non-positive apparent resistivities are always removed,
    because they cannot be inverted at all. Everything else is a judgement call
    and is left to you - either set the arguments here, or use interactive_qc().
    """
    n0 = survey.n_valid
    if verbose:
        print(f"\n  ---- filters ({n0} valid measurements) ----")

    if survey.has('rhoa'):
        survey.reject(~np.isfinite(survey.rhoa), "non-finite rhoa", verbose)
        survey.reject(survey.rhoa <= 0,
                      "rhoa <= 0 (cannot be inverted)", verbose)
        if min_rhoa is not None:
            survey.reject(survey.rhoa < min_rhoa, f"rhoa < {min_rhoa}", verbose)
        if max_rhoa is not None:
            survey.reject(survey.rhoa > max_rhoa, f"rhoa > {max_rhoa}", verbose)

    if k_max is not None and survey.has('k'):
        survey.reject(np.abs(survey.k) > k_max, f"|k| > {k_max} m", verbose)
    if min_u is not None and survey.has('u'):
        survey.reject(np.abs(survey.u) < min_u, f"|U| < {min_u} V", verbose)
    if min_i is not None and survey.has('i'):
        survey.reject(np.abs(survey.i) < min_i, f"|I| < {min_i} mA", verbose)
    if max_instrument_err is not None and survey.has('err'):
        survey.reject(survey.err > max_instrument_err,
                      f"instrument error > {100*max_instrument_err:.1f}%",
                      verbose)

    if max_reciprocal_err is not None:
        rec = reciprocal_errors(survey, verbose=False)
        if rec['pairs']:
            bad = np.zeros(survey.n_data, bool)
            for (i, j), e in zip(rec['pairs'], rec['errors']):
                if e > max_reciprocal_err:
                    bad[i] = bad[j] = True
            survey.reject(bad, f"reciprocal error > "
                               f"{100*max_reciprocal_err:.1f}%", verbose)

    if neighbour_thr is not None and survey.n_valid > n_neighbours + 1:
        survey.reject(neighbour_outliers(survey, thr=neighbour_thr,
                                         k=n_neighbours),
                      f"local outlier > {neighbour_thr} MAD", verbose)

    if verbose:
        print(f"  [qc] {survey.n_valid} / {survey.n_data} valid "
              f"({100.0 * survey.n_valid / max(survey.n_data, 1):.1f}%)")
        print("  ---------------------------------\n")
    return survey


def neighbour_outliers(survey, thr=3.0, k=10):
    """Flag measurements whose log10(rhoa) disagrees with their neighbours.

    Feature space = measurement centre plus scaled log10 of the configuration
    size, which plays the role that the a-spacing level plays in a 2D
    pseudosection. Only valid measurements take part.
    """
    bad = np.zeros(survey.n_data, bool)
    sel = np.where(survey.valid & np.isfinite(survey.rhoa) &
                   (survey.rhoa > 0))[0]
    if len(sel) < k + 2:
        return bad

    lr = np.log10(survey.rhoa[sel])
    centres = survey.centres()[sel]
    ls = np.log10(np.maximum(survey.spacings()[sel], 1e-9))

    span = centres.max(axis=0) - centres.min(axis=0)
    scale = np.median(span[span > 0]) if np.any(span > 0) else 1.0
    feat = np.column_stack([centres, ls * scale])

    if not _HAS_SCIPY:
        bins = np.quantile(ls, np.linspace(0, 1, 12))
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (ls >= lo) & (ls <= hi)
            if s.sum() < 5:
                continue
            med = np.median(lr[s])
            mad = 1.4826 * np.median(np.abs(lr[s] - med)) + 1e-12
            bad[sel[s]] = np.abs(lr[s] - med) > thr * mad
        return bad

    kq = min(k + 1, len(feat))
    _, idx = cKDTree(feat).query(feat, k=kq)
    neigh = lr[idx[:, 1:]]
    med = np.median(neigh, axis=1)
    mad = np.maximum(1.4826 * np.median(np.abs(neigh - med[:, None]), axis=1),
                     1e-3)
    bad[sel] = np.abs(lr - med) > thr * mad
    return bad


# =====================================================================
# ------------------ 10. Interactive QC -------------------------------
# =====================================================================

class InteractiveQC3D:
    """Click-to-toggle QC for a 3D dataset.

    Measurements are grouped into panels the way pseudosection levels group a
    2D line: by which line the configuration sits on, and by its size. Within a
    panel, apparent resistivity is plotted against distance along the survey
    axis, with a rolling median as a visual guide.

    Controls
        left-click a point   toggle accept/reject
        Next / Prev          move between groups
        Reject all / Accept all   act on everything in the current panel
        Auto-flag            run the neighbour filter on this panel only
        close the window     apply and continue
    """

    def __init__(self, survey, group_by='line+size', window=5, logy=True):
        self.s = survey
        self.window = window
        self.logy = logy
        self._build_groups(group_by)
        self.idx = 0
        self._make_figure()

    # ---- grouping ---------------------------------------------------
    def _build_groups(self, group_by):
        s = self.s
        sp = s.spacings()
        step = np.median(np.diff(np.unique(np.round(sp, 3)))) if \
            len(np.unique(np.round(sp, 3))) > 1 else 1.0
        size_key = np.round(sp / max(step, 1e-9)).astype(int)
        line_key = s.line_key()

        keys = []
        for i in range(s.n_data):
            parts = []
            if 'line' in group_by:
                parts.append(('line', line_key[i]))
            if 'size' in group_by:
                parts.append(('size', size_key[i]))
            if 'file' in group_by:
                parts.append(('file', s.src[i]))
            keys.append(tuple(parts) if parts else (('all', 0),))

        seen = {}
        for i, kk in enumerate(keys):
            seen.setdefault(kk, []).append(i)

        lines = sorted({v for kk in seen for n, v in kk if n == 'line'})

        def _fmt(n, v, first):
            if n == 'size':
                return f"size {sp[first]:.2f} m"
            if n == 'line':
                off = ', '.join(f"{float(t):+.2f}" for t in v)
                return f"line {lines.index(v) + 1} (offset {off} m)"
            return f"{n} {v}"

        order = []
        for kk in sorted(seen, key=lambda t: [str(v) for _, v in t]):
            first = seen[kk][0]
            label = ', '.join(_fmt(n, v, first) for n, v in kk)
            order.append((label, np.asarray(seen[kk])))
        self.groups = order
        print(f"  [qc] {len(self.groups)} groups: "
              f"{', '.join(l for l, _ in self.groups[:4])}"
              f"{' ...' if len(self.groups) > 4 else ''}")

    # ---- figure -----------------------------------------------------
    def _make_figure(self):
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        plt.subplots_adjust(bottom=0.22)
        from matplotlib.widgets import Button
        specs = [('Prev', 0.06, self._prev), ('Next', 0.17, self._next),
                 ('Reject all', 0.32, self._reject_all),
                 ('Accept all', 0.46, self._accept_all),
                 ('Auto-flag', 0.60, self._auto_flag)]
        self._buttons = []
        for label, x, cb in specs:
            b = Button(plt.axes([x, 0.06, 0.10, 0.06]), label)
            b.on_clicked(cb)
            self._buttons.append(b)
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self._draw()

    def _draw(self):
        self.ax.clear()
        label, ind = self.groups[self.idx]
        x = self.s.along()[ind]
        y = self.s.rhoa[ind]
        o = np.argsort(x)
        x, y, ind = x[o], y[o], ind[o]
        self._ind = ind

        if len(y) >= self.window:
            from scipy.ndimage import median_filter
            self.ax.plot(x, median_filter(y, size=self.window), color='grey',
                         ls='--', lw=1, zorder=1)

        colors = np.where(self.s.valid[ind], 'tab:blue', 'tab:red')
        self._sc = self.ax.scatter(x, y, c=colors, s=45, marker='o',
                                   edgecolors='k', linewidths=0.5, picker=5,
                                   zorder=2)
        if self.logy:
            self.ax.set_yscale('log')
        self.ax.set_xlabel('Distance along survey axis (m)')
        self.ax.set_ylabel('Apparent resistivity (Ωm)')
        n_ok = int(self.s.valid[ind].sum())
        self.ax.set_title(f"[{self.idx + 1}/{len(self.groups)}]  {label}   "
                          f"({n_ok}/{len(ind)} kept)   "
                          f"total {self.s.n_valid}/{self.s.n_data}")
        self.ax.grid(True, ls='--', alpha=0.5)
        self.fig.canvas.draw_idle()

    # ---- callbacks --------------------------------------------------
    def _on_pick(self, event):
        if event.artist is not self._sc:
            return
        for j in event.ind:
            g = self._ind[j]
            self.s.valid[g] = not self.s.valid[g]
        self._draw()

    def _next(self, _):
        self.idx = min(self.idx + 1, len(self.groups) - 1)
        self._draw()

    def _prev(self, _):
        self.idx = max(self.idx - 1, 0)
        self._draw()

    def _reject_all(self, _):
        self.s.valid[self._ind] = False
        self._draw()

    def _accept_all(self, _):
        self.s.valid[self._ind] = True
        self._draw()

    def _auto_flag(self, _):
        bad = neighbour_outliers(self.s, thr=3.0, k=min(10, len(self._ind) - 1))
        self.s.valid[self._ind] &= ~bad[self._ind]
        self._draw()

    # ---- run --------------------------------------------------------
    def run(self):
        print("  [qc] click points to toggle; close the window to continue")
        plt.show(block=True)
        return self.s

    def save_summary(self, path, ncols=3):
        """Static overview of every group with the final selection."""
        n = len(self.groups)
        nrows = int(np.ceil(n / ncols))
        fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows),
                                squeeze=False)
        axs = axs.flatten()
        for ax, (label, ind) in zip(axs, self.groups):
            x = self.s.along()[ind]
            y = self.s.rhoa[ind]
            o = np.argsort(x)
            c = np.where(self.s.valid[ind][o], 'tab:blue', 'tab:red')
            ax.scatter(x[o], y[o], c=c, s=25, edgecolors='k', linewidths=0.4)
            if self.logy:
                ax.set_yscale('log')
            ax.set_title(label, fontsize=10)
            ax.grid(True, ls='--', alpha=0.4)
        for j in range(n, len(axs)):
            fig.delaxes(axs[j])
        fig.supxlabel('Distance along survey axis (m)')
        fig.supylabel('Apparent resistivity (Ωm)')
        fig.suptitle(f'QC selection  (blue = kept, red = rejected)   '
                     f'{self.s.n_valid}/{self.s.n_data}', fontweight='bold')
        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  [qc] summary plot saved to {path}")


def interactive_qc(survey, group_by='line+size', window=5, summary_path=None):
    """Convenience wrapper: open the QC window, then optionally save a summary."""
    tool = InteractiveQC3D(survey, group_by=group_by, window=window)
    tool.run()
    if summary_path:
        tool.save_summary(summary_path)
    return survey


# =====================================================================
# ----------------------- 11. 3D mesh building -------------------------
# =====================================================================

def ensure_tetgen(extra_dirs=None, verbose=True):
    """Locate the TetGen executable and put its folder on PATH.

    pyGIMLi shells out to a `tetgen` binary for 3D meshing. If the shell cannot
    find it, createMesh fails deep inside readTetgen with a missing .node file,
    which does not obviously say "TetGen is not installed".

    Resolution lives here rather than in each main script so that every entry
    point behaves identically - two scripts setting PATH in their own headers
    is exactly how one ends up working and the other not.

    Note: the pip package called `tetgen` provides Python bindings, not the
    command-line binary pyGIMLi needs. The binary comes from the TetGen
    distribution itself.
    """
    import shutil

    found = shutil.which('tetgen')
    if found:
        if verbose:
            print(f"  [tetgen] found on PATH: {found}")
        return found

    here = os.path.dirname(os.path.abspath(__file__))
    script = os.path.dirname(os.path.abspath(sys.argv[0])) if sys.argv and \
        sys.argv[0] else os.getcwd()
    candidates = list(extra_dirs or [])
    candidates += [
        os.path.join(script, 'tetgen', 'build'),
        os.path.join(script, 'tetgen'), script,
        os.path.join(here, 'tetgen', 'build'),
        os.path.join(here, 'tetgen'), here,
        os.path.join(os.getcwd(), 'tetgen', 'build'),
        os.path.join(os.getcwd(), 'tetgen'), os.getcwd(),
        r'C:\tetgen', r'C:\Program Files\tetgen',
        os.path.join(sys.prefix, 'Scripts'),
        os.path.join(sys.prefix, 'Library', 'bin'),
        os.path.join(sys.prefix, 'bin'), sys.prefix,
    ]

    seen = []
    if os.name == 'nt':
        candidate_names = ('tetgen.exe', 'tetgen')
    else:
        candidate_names = ('tetgen',)

    for d in candidates:
        if not d or d in seen:
            continue
        seen.append(d)
        for name in candidate_names:
            exe = os.path.join(d, name)
            if not os.path.isfile(exe):
                continue
            if os.name != 'nt' and not os.access(exe, os.X_OK):
                continue
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
            if verbose:
                print(f"  [tetgen] found at {exe}")
                print(f"  [tetgen] added {d} to PATH for this process")
            return exe

    raise RuntimeError(
        "TetGen was not found, so no 3D mesh can be built.\n"
        "Searched:\n  " + "\n  ".join(seen) +
        "\n\nFix it in one of these ways:\n"
        "  1. put tetgen.exe in the folder holding your scripts, or in "
        "C:\\tetgen\n"
        "  2. add its folder to your Windows PATH permanently\n"
        "  3. pass the folder explicitly:  TETGEN_DIR = r'C:\\path\\to\\folder'\n"
        "     at the top of the script\n\n"
        "The pip package named 'tetgen' is NOT this - pyGIMLi needs the "
        "command-line binary.")


def build_3d_mesh(data, para_depth, para_max_cell_size, para_boundary=None,
                  para_dx=0.0, surface_quality=34, quality=1.3,
                  boundary=None, tetgen_dirs=None, verbose=True,
                  save_path=None):
    """Build the 3D PLC and tetrahedral mesh, and verify it really is 3D.

    para_max_cell_size is a VOLUME in m^3 here, not an area. A reasonable
    starting point is (0.5 to 1.0 x electrode spacing)^3.
    """
    data = data.to_container() if isinstance(data, Survey) else data
    ensure_tetgen(extra_dirs=tetgen_dirs, verbose=verbose)
    kwargs = dict(sensors=data,
                  paraDepth=para_depth,
                  paraMaxCellSize=para_max_cell_size,
                  surfaceMeshQuality=surface_quality)
    if para_boundary is not None:
        kwargs['paraBoundary'] = para_boundary
    if para_dx:
        kwargs['paraDX'] = para_dx
    if boundary is not None:
        kwargs['boundary'] = boundary

    plc = mt.createParaMeshPLC3D(**kwargs)
    if verbose:
        print(f"  [mesh] PLC: {plc.nodeCount()} nodes, "
              f"{plc.boundaryCount()} facets")
    try:
        mesh = mt.createMesh(plc, quality=quality)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"TetGen produced no output ({e}).\n"
            "TetGen was located but the meshing call still failed. The usual "
            "causes are a PLC TetGen cannot close (check the facet count "
            "above is non-zero), or a paraMaxCellSize so small that meshing "
            "runs out of memory. Try a larger paraMaxCellSize first.") from e

    verify_3d(mesh, data)
    if verbose:
        print(f"  [mesh] {mesh.cellCount()} cells, {mesh.nodeCount()} nodes, "
              f"dim {mesh.dim()}")
        bb = mesh.bb()
        print(f"  [mesh] bounds x {bb[0].x():.1f}..{bb[1].x():.1f}, "
              f"y {bb[0].y():.1f}..{bb[1].y():.1f}, "
              f"z {bb[0].z():.1f}..{bb[1].z():.1f}")
    if save_path:
        mesh.save(save_path)
        print(f"  [mesh] saved to {save_path}")
    return mesh


def _sensor_xyz(obj):
    """(E, 3) electrode positions from either a Survey or a DataContainerERT."""
    if isinstance(obj, Survey):
        return obj.pos
    return np.array([[p.x(), p.y(), p.z()] for p in obj.sensorPositions()])


def verify_3d(mesh, data=None):
    """Fail loudly rather than silently inverting in 2D.

    pyGIMLi will happily build and invert a 2D mesh if the PLC collapsed,
    and the only symptom is a suspiciously fast, suspiciously smooth result.
    """
    if mesh.dim() != 3:
        raise RuntimeError(
            f"Mesh is {mesh.dim()}D, not 3D. The PLC collapsed - usually "
            f"because the electrodes are collinear, or paraBoundary/paraDepth "
            f"shrank the domain past the electrode extent.")
    if mesh.cellCount() < 100:
        raise RuntimeError(f"Mesh has only {mesh.cellCount()} cells - TetGen "
                           f"almost certainly failed. Check it is on PATH.")
    if data is not None:
        p = _sensor_xyz(data)
        bb = mesh.bb()
        lo = np.array([bb[0].x(), bb[0].y(), bb[0].z()])
        hi = np.array([bb[1].x(), bb[1].y(), bb[1].z()])
        if np.any(p.min(axis=0) < lo - 1e-6) or np.any(p.max(axis=0) > hi + 1e-6):
            raise RuntimeError("Electrodes fall outside the mesh bounding box - "
                               "the parameter domain has been shrunk too far.")
    return True


# =====================================================================
# ----------------------- 12. Inversion --------------------------------
# =====================================================================

def run_inversion_3d(args):
    """One 3D inversion for one (cType, zWeight, lambda) combination.

    Return signature matches run_inversion() in ERTutilities_3D_v2 so the
    existing lamOpt() L-curve code works unchanged.
    """
    (ctype, zw, lam, max_iter, robust_data, blocky_model,
     data_file, mesh_file, vtk_dir, residual_dir, start_model_file) = args

    mgr = None
    try:
        data = ert.load(data_file)
        mesh = pg.load(mesh_file)
        verify_3d(mesh, data)
        print(f"[INVERSION] {data.sensorCount()} sensors, {data.size()} data, "
              f"{mesh.cellCount()} cells | cType={ctype} zW={zw} lam={lam}")

        mgr = ert.ERTManager(data)
        mgr.setMesh(mesh)

        start_model = None
        if start_model_file and os.path.exists(start_model_file):
            start_model = np.loadtxt(start_model_file)
            print(f"  using start model {start_model_file}")

        mgr.inv.inv.setDeltaPhiAbortPercent(0)
        mgr.invert(mesh=mesh, verbose=True, cType=ctype, zWeight=zw, lam=lam,
                   maxIter=max_iter, robustData=robust_data,
                   blockyModel=blocky_model, startModel=start_model)

        # the parameter domain must be a real 3D mesh after inversion. If the
        # forward-mesh refinement ran out of memory, paraDomain can come back as
        # a stub (e.g. a DataMap) with no .dim(); treat that as a failed
        # inversion rather than crashing with an obscure AttributeError.
        pd = getattr(mgr, 'paraDomain', None)
        if pd is None or not hasattr(pd, 'dim'):
            raise RuntimeError("inversion did not produce a parameter mesh "
                               "(usually an out-of-memory during forward-mesh "
                               "refinement - look for a MemoryError above).")
        if pd.dim() != 3:
            raise RuntimeError("paraDomain came back 2D - inversion silently "
                               "fell back to a 2D problem.")

        out_mesh = pg.Mesh(mgr.paraDomain)
        out_mesh['resistivity'] = mgr.model
        out_mesh['log10_resistivity'] = np.log10(mgr.model)
        cov = mgr.coverage()
        out_mesh['coverage'] = cov
        out_mesh['standardised_coverage'] = np.sign(np.abs(cov - np.median(cov)))
        grad = pg.solver.grad(out_mesh, mgr.model)
        out_mesh['model_gradient'] = np.linalg.norm(grad, axis=1)

        vtk_out = os.path.join(vtk_dir,
                               f"model_ctype{ctype}_zw{zw}_lam{lam:.3f}.vtk")
        out_mesh.exportVTK(vtk_out)

        model_norm = np.linalg.norm(mgr.model)
        C = mgr.fop.constraints()
        Cm = np.array(C * mgr.model)
        weighted_norm = np.sqrt(Cm.dot(Cm))

        _plot_residuals(mgr, ctype, zw, lam, residual_dir)

        return [ctype, zw, lam, mgr.inv.iter, mgr.inv.chi2(), mgr.inv.absrms(),
                mgr.inv.relrms(), mgr.inv.phi(), mgr.inv.phiModel(),
                mgr.inv.phiData(), model_norm, weighted_norm, vtk_out]

    except MemoryError:
        print(f"Inversion FAILED (cType={ctype}, zW={zw}, lam={lam}): out of "
              f"memory. The refined (H2) forward mesh for this geometry is "
              f"large (~10^5 cells); free RAM between runs (now done in the "
              f"finally below), coarsen PARA_MAX_CELL_SIZE / PARA_DEPTH, or run "
              f"fewer lambdas per process.")
        return [ctype, zw, lam, None, None, None, None, None, None, None,
                None, None, "FAILED"]

    except Exception as e:
        import traceback
        print(f"Inversion FAILED (cType={ctype}, zW={zw}, lam={lam}): {e}")
        traceback.print_exc()
        return [ctype, zw, lam, None, None, None, None, None, None, None,
                None, None, "FAILED"]

    finally:
        # release the manager and its large forward/refined meshes before the
        # next lambda, so a sweep does not accumulate meshes until it OOMs.
        try:
            del mgr
        except Exception:
            pass
        plt.close('all')
        gc.collect()


def _plot_residuals(mgr, ctype, zw, lam, residual_dir):
    obs = np.array(mgr.inv.dataVals)
    pred = np.array(mgr.inv.response)
    res = pred - obs

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].hist(res, bins=40, color='royalblue', edgecolor='navy', alpha=0.7)
    ax[0].axvline(0, color='red', ls='--', label='zero')
    ax[0].axvline(np.median(res), color='orange', ls='--',
                  label=f'median = {np.median(res):.3f}')
    ax[0].set_xlim(*np.percentile(res, [2, 98]))
    ax[0].set_xlabel('Residual (Ωm)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title(f'Residual distribution\ncType={ctype}, λ={lam}, zW={zw}')
    ax[0].legend()
    ax[0].grid(True, ls='--', alpha=0.5)

    sc = ax[1].scatter(obs, pred, c=res, cmap='viridis', alpha=0.6, s=10)
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax[1].plot(lims, lims, 'k--', lw=1, label='1:1')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('Observed ρₐ (Ωm)')
    ax[1].set_ylabel('Predicted ρₐ (Ωm)')
    ax[1].set_title('Observed vs predicted')
    ax[1].legend()
    ax[1].grid(True, which='both', ls='--', alpha=0.5)
    plt.colorbar(sc, ax=ax[1], label='Residual (Ωm)')

    plt.tight_layout()
    fig.savefig(os.path.join(residual_dir,
                             f'Residuals_ctype{ctype}_zw{zw}_lam{lam:.3f}.png'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# ----------------------- 13. Diagnostics ------------------------------
# =====================================================================

def plot_survey_3d(survey, outdir, base_name='survey'):
    """Electrode layout, plan view, and rhoa / |k| / error histograms."""
    os.makedirs(outdir, exist_ok=True)
    p = _sensor_xyz(survey)
    v = survey.valid if isinstance(survey, Survey) else slice(None)

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, height_ratios=[3, 2], wspace=0.3, hspace=0.35)

    ax0 = fig.add_subplot(gs[0, :2], projection='3d')
    sc = ax0.scatter(p[:, 0], p[:, 1], p[:, 2], c=p[:, 2], cmap='terrain', s=22)
    ax0.set_xlabel('X (m)')
    ax0.set_ylabel('Y (m)')
    ax0.set_zlabel('Z (m)')
    ax0.set_title(f'Electrode layout ({len(p)} electrodes)')
    ax0.view_init(elev=28, azim=-125)
    fig.colorbar(sc, ax=ax0, shrink=0.6, label='Elevation (m)')

    ax1 = fig.add_subplot(gs[0, 2])
    ax1.scatter(p[:, 0], p[:, 1], c='k', s=12)
    ax1.set_aspect('equal', 'box')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title('Plan view')
    ax1.grid(True, ls='--', alpha=0.5)

    def _hist(ax, vals, label, colour):
        vals = np.asarray(vals, float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            ax.hist(vals, bins=60, color=colour, edgecolor='k', alpha=0.4)
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(label)
        ax.grid(True, which='both', ls='--', alpha=0.4)

    ax2 = fig.add_subplot(gs[1, 0])
    _hist(ax2, survey.rhoa[v], 'Apparent resistivity (Ωm)', 'royalblue')
    ax2.set_ylabel('Frequency')
    ax3 = fig.add_subplot(gs[1, 1])
    _hist(ax3, np.abs(survey.k[v]) if survey.has('k') else [],
          '|Geometric factor| (m)', 'seagreen')
    ax4 = fig.add_subplot(gs[1, 2])
    _hist(ax4, 100 * survey.err[v] if survey.has('err') else [],
          'Relative error (%)', 'indianred')

    fig.suptitle(f'{base_name}: 3D survey geometry and data', fontsize=18)
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(outdir, f'{base_name}_survey3D.{ext}'),
                    dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [plot] survey diagnostics saved to {outdir}")


def plot_coverage_slices(vtk_or_mesh, outdir, base_name='model', z_levels=None):
    """Quick 2D slices through the 3D model, for when ParaView is overkill."""
    mesh = pg.load(vtk_or_mesh) if isinstance(vtk_or_mesh, str) else vtk_or_mesh
    res = np.asarray(mesh['resistivity'])
    cc = np.array([[c.center().x(), c.center().y(), c.center().z()]
                   for c in mesh.cells()])

    if z_levels is None:
        z_levels = np.percentile(cc[:, 2], [90, 70, 50, 30, 10])

    n = len(z_levels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    dz = 0.5 * np.diff(np.sort(np.unique(np.round(cc[:, 2], 2)))).mean()
    for ax, z in zip(axes[0], z_levels):
        sel = np.abs(cc[:, 2] - z) < max(dz, 0.05)
        if sel.sum() == 0:
            continue
        s = ax.scatter(cc[sel, 0], cc[sel, 1], c=np.log10(res[sel]),
                       cmap='Spectral_r', s=12)
        ax.set_aspect('equal', 'box')
        ax.set_title(f'z = {z:.2f} m')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        fig.colorbar(s, ax=ax, label='log₁₀ ρ (Ωm)')
    fig.suptitle(f'{base_name}: depth slices')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, f'{base_name}_slices.png'), dpi=300,
                bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# ------------------ 14. L-curve (self-contained) ----------------------
# =====================================================================
#
# Ported out of ERTutilities_3D_v2.lamOpt so the true-3D pipeline no longer
# has to import that module. The old module pulls in PyQt5, shapely and a
# forced 'Qt5Agg' backend at import time and therefore dies on a headless
# cluster node, and the failure was being swallowed by the try/except around
# the import - so no L-curve, no message. This version depends only on
# numpy/pandas/matplotlib (already imported at the top of this file) and
# writes with savefig, so it is headless-safe WITHOUT forcing a backend -
# which also means it leaves the interactive-QC backend alone.


def _lcurve_corner(phi_model, phi_data):
    """Index of the L-curve corner, robust to wide sweeps and swing-back.

    Two-stage method that survives an aggressive lambda range far better than a
    raw discrete-curvature estimate:

      1. Pareto front. Working in log-log, keep only the lambdas that are not
         dominated - i.e. drop any point that has another point with BOTH lower
         roughness and lower misfit. This deletes the low-lambda "hook" where an
         over-rough model starts fitting noise and the curve doubles back, which
         otherwise poisons any curvature estimate.

      2. Maximum distance to the chord. Normalise the surviving front to the unit
         square and take the point furthest (perpendicular) from the straight
         line joining its two ends. That is the classic robust L-corner / Kneedle
         criterion, and unlike three-point Menger curvature it depends on the
         global shape rather than local point spacing.

    Returns an index into the ORIGINAL passed arrays, or None if a corner cannot
    be defined (fewer than three usable points, or non-positive phi values).
    """
    pm = np.asarray(phi_model, float)
    pd_ = np.asarray(phi_data, float)
    if len(pm) < 3:
        return None
    lx, ly = np.log(pm), np.log(pd_)
    if not (np.all(np.isfinite(lx)) and np.all(np.isfinite(ly))):
        return None

    # --- 1. Pareto front (ascending roughness, strictly decreasing misfit) ---
    order = np.argsort(lx)
    front, best_y = [], np.inf
    for idx in order:
        if ly[idx] < best_y - 1e-12:
            front.append(idx)
            best_y = ly[idx]
    front = np.asarray(front)
    if len(front) < 3:
        return None

    # --- 2. Max perpendicular distance to the end-to-end chord ---------------
    fx, fy = lx[front], ly[front]
    sx = fx.max() - fx.min()
    sy = fy.max() - fy.min()
    fxn = (fx - fx.min()) / sx if sx > 0 else np.zeros_like(fx)
    fyn = (fy - fy.min()) / sy if sy > 0 else np.zeros_like(fy)
    x0, y0, x1, y1 = fxn[0], fyn[0], fxn[-1], fyn[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.hypot(dx, dy)
    if denom < 1e-12:
        return None
    dist = np.abs(dy * (fxn - x0) - dx * (fyn - y0)) / denom
    return int(front[np.argmax(dist)])


def plot_lcurve(misfit_file, meta_file=None, output_dir='.', base_name='survey',
                lam_idx=None, logplot=False, normplot=False, geom_avg=True,
                arith_avg=False, pct_start=False, mark_corner=True,
                verbose=True):
    """Draw an L-curve (phiModel vs phiData) per cType/zWeight and save a JPG.

    Self-contained replacement for ERTutilities_3D_v2.lamOpt, with the same
    plotting modes plus optional automatic corner detection. Reads the misfit
    CSV written by the main script, which already carries phiModel, phiData,
    Chi2, RRMS and lambda; meta_file is accepted for signature compatibility but
    is not needed for the plot (it is merged in only if present).

    Exactly one of logplot / normplot / geom_avg / arith_avg / pct_start may be
    True, or none for raw linear axes - the same rule as the original.

    Returns {(cType, zWeight): suggested_lambda} from the corner finder; empty
    if mark_corner is False or no corner could be found. That return value lets
    the caller print or reuse the suggested lambda.
    """
    active = sum([logplot, normplot, geom_avg, arith_avg, pct_start])
    if active > 1:
        raise ValueError("choose at most one of logplot / normplot / geom_avg "
                         "/ arith_avg / pct_start (or none for linear axes)")

    plt.rcParams['font.size'] = 10
    plt.rcParams['mathtext.fontset'] = 'cm'

    df = pd.read_csv(misfit_file)
    need = {'cType', 'zWeight', 'lambda', 'phiModel', 'phiData', 'Chi2', 'RRMS'}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"misfit file {misfit_file} is missing columns "
                       f"{sorted(missing)} - the L-curve needs them.")
    if meta_file and os.path.exists(meta_file):
        try:
            dm = pd.read_csv(meta_file)
            shared = [c for c in ('cType', 'zWeight', 'lambda') if c in dm.columns]
            if shared:
                df = pd.merge(df, dm, on=shared, how='left')
        except Exception as e:
            if verbose:
                print(f"  [lcurve] meta merge skipped ({e})")

    os.makedirs(output_dir, exist_ok=True)
    size_in = 75 / 25.4
    suggested = {}

    for ctype in sorted(df['cType'].unique()):
        cdf = df[df['cType'] == ctype]
        for zw in sorted(cdf['zWeight'].unique()):
            subset = cdf[cdf['zWeight'] == zw].sort_values('lambda')
            if len(subset) < 2:
                if verbose:
                    print(f"  [lcurve] cType={ctype} zW={zw}: only "
                          f"{len(subset)} point(s), need >=2 - skipped")
                continue

            x = subset['phiModel'].to_numpy(float)
            y = subset['phiData'].to_numpy(float)
            lam = subset['lambda'].to_numpy(float)
            chi2 = subset['Chi2'].to_numpy(float)
            rrms = subset['RRMS'].to_numpy(float)

            fig, ax = plt.subplots(figsize=(size_in, size_in))

            # ---- one of the axis transforms / label sets ----
            xlab = r'Model Roughness $\Phi_m$'
            ylab = r'Data Misfit $\Phi_d$'
            if logplot:
                xp, yp = x, y
                ax.set_xscale('log')
                ax.set_yscale('log')
            elif normplot:
                xp = (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
                yp = (y - y.min()) / (y.max() - y.min()) if y.max() > y.min() else y
                xlab = r'Normalised Model Roughness $\Phi_m$'
                ylab = r'Normalised Data Misfit $\Phi_d$'
            elif geom_avg and np.all(x > 0) and np.all(y > 0):
                xp = x / np.exp(np.log(x).mean())
                yp = y / np.exp(np.log(y).mean())
            elif arith_avg:
                xp = x / x.mean()
                yp = y / y.mean()
                xlab = r'Model Roughness / Arithmetic Mean'
                ylab = r'Data Misfit / Arithmetic Mean'
            elif pct_start:
                xp = (x / x[0]) * 100.0 if x[0] != 0 else x
                yp = (y / y[0]) * 100.0 if y[0] != 0 else y
                xlab = r'Model Roughness ($\%$ of Initial)'
                ylab = r'Data Misfit ($\%$ of Initial)'
            else:
                # raw linear, or geom_avg requested but a value was non-positive
                if geom_avg and verbose:
                    print(f"  [lcurve] cType={ctype} zW={zw}: geom_avg needs "
                          f"positive phiModel/phiData - falling back to raw axes")
                xp, yp = x, y
            ax.set_xlabel(xlab, fontsize=10)
            ax.set_ylabel(ylab, fontsize=10)

            ax.plot(xp, yp, marker='x', linestyle='--', linewidth=0.6,
                    color='black', markersize=4)

            if normplot or pct_start:
                ax.set_aspect('equal', adjustable='box')
                g0 = min(xp.min(), yp.min())
                g1 = max(xp.max(), yp.max())
                m = ((g1 - g0) or 0.1) * 0.05
                ax.set_xlim(g0 - m, g1 + m)
                ax.set_ylim(g0 - m, g1 + m)
                ax.set_yticks(ax.get_xticks())

            def _lbl(i):
                return (rf'$\lambda={lam[i]:.2f}$' + '\n' +
                        rf'$\chi^2={chi2[i]:.2f}$' + '\n' +
                        rf'$\mathrm{{RRMS}}={rrms[i]:.2f}\%$')

            bbox = dict(boxstyle='round,pad=0.2', fc='white', ec='none',
                        alpha=0.8)
            arrow = dict(arrowstyle='->', linewidth=0.6, color='black')

            ax.annotate(_lbl(0), xy=(xp[0], yp[0]), xycoords='data',
                        xytext=(0.95, 0.15), textcoords='axes fraction',
                        fontsize=8, ha='right', va='bottom',
                        multialignment='left', arrowprops=arrow, bbox=bbox)
            ax.annotate(_lbl(len(xp) - 1), xy=(xp[-1], yp[-1]), xycoords='data',
                        xytext=(0.15, 0.95), textcoords='axes fraction',
                        fontsize=8, ha='left', va='top',
                        multialignment='right', arrowprops=arrow, bbox=bbox)
            if lam_idx is not None and 0 <= lam_idx < len(xp):
                ax.annotate(_lbl(lam_idx), xy=(xp[lam_idx], yp[lam_idx]),
                            xycoords='data', xytext=(0.95, 0.95),
                            textcoords='axes fraction', fontsize=8, ha='right',
                            va='top', multialignment='right',
                            arrowprops=arrow, bbox=bbox)

            # ---- automatic corner (Hansen max-curvature) ----
            if mark_corner:
                ci = _lcurve_corner(x, y)
                if ci is not None:
                    suggested[(ctype, zw)] = float(lam[ci])
                    ax.plot(xp[ci], yp[ci], marker='o', mfc='none',
                            mec='crimson', mew=1.4, markersize=12, zorder=5)
                    if verbose:
                        note = " (coarse - few lambdas)" if len(xp) < 5 else ""
                        print(f"  [lcurve] cType={ctype} zW={zw}: corner near "
                              f"lambda={lam[ci]:.2f} (chi2={chi2[ci]:.2f}, "
                              f"RRMS={rrms[ci]:.2f}%){note}")

            ax.tick_params(axis='both', which='major', labelsize=9)
            ax.grid(True, which='both', linestyle='--', alpha=0.3)
            plt.tight_layout(pad=1.1)

            out = os.path.join(output_dir,
                               f'{base_name}_cType{ctype}_zW{zw}_Lcurve.jpg')
            fig.savefig(out, dpi=600)
            plt.close(fig)
            if verbose:
                print(f"  [lcurve] saved {out}")

    return suggested


# Backwards-compatible alias: anything still calling lamOpt() lands here.
lamOpt = plot_lcurve


# =====================================================================
# ------------------ 15. Timelapse ------------------------------------
# =====================================================================
#
# pyGIMLi's TimelapseERT requires every timestep to hold exactly the same
# measurements, in the same order, on the same electrodes. The 2D workflow
# achieved that with a positional mask, which is only safe when every file has
# identical row order and length. Here the alignment is done on the electrode
# quadruple itself, so timesteps may be acquired in any order, may differ in
# length, and may have had different points rejected during QC.


def times_from_filenames(files, verbose=True):
    """Pull acquisition times out of filenames.

    Recognises YYYYMMDD, YYMMDD and YYYY-MM-DD, optionally followed by HHMM.
    Returns None if any file cannot be parsed, so the caller can fall back to
    an explicit list.
    """
    import datetime as _dt
    pats = [(r'(20\d{2})[-_]?(\d{2})[-_]?(\d{2})', 4),
            (r'(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)', 2)]
    out = []
    for f in files:
        base = os.path.basename(f)
        got = None
        for pat, ylen in pats:
            m = re.search(pat, base)
            if m:
                y, mo, d = (int(g) for g in m.groups()[:3])
                if ylen == 2:
                    y += 2000
                hm = re.search(r'[_-](\d{2})(\d{2})(?!\d)', base[m.end():])
                try:
                    got = _dt.datetime(y, mo, d,
                                       int(hm.group(1)) if hm else 12,
                                       int(hm.group(2)) if hm else 0)
                except ValueError:
                    got = None
                if got:
                    break
        if got is None:
            if verbose:
                print(f"  [time] could not read a date from {base} - "
                      f"set TIMES explicitly")
            return None
        out.append(got)
    if verbose:
        for f, t in zip(files, out):
            print(f"  [time] {os.path.basename(f)}  ->  {t:%Y-%m-%d %H:%M}")
    return np.array(out)


def times_from_surveys(surveys, verbose=True):
    """Acquisition times taken from the instrument's own timestamps."""
    ts = [s.acquired for s in surveys]
    if any(t is None for t in ts):
        if verbose:
            print("  [time] not every file carries an acquisition timestamp")
        return None
    if verbose:
        for s, t in zip(surveys, ts):
            print(f"  [time] {s.src[0]}  ->  {t:%Y-%m-%d %H:%M}")
    return np.array(ts)


def resolve_times(n, explicit=None, files=None, surveys=None, verbose=True):
    """Work out one time per timestep, in order of preference.

    explicit list -> filenames -> the instrument's own Time column -> equidistant.

    Never returns None: pyGIMLi's TimelapseERT calls len() on whatever it is
    given, so handing it None fails inside its constructor.
    """
    import datetime as _dt

    if explicit is not None:
        if len(explicit) != n:
            raise ValueError(f"{len(explicit)} times given for {n} files")
        if verbose:
            print("  [time] using the times given in the config")
        return np.asarray(explicit)

    if files:
        t = times_from_filenames(files, verbose=verbose)
        if t is not None and len(t) == n:
            print("  [time] taken from the filenames")
            return t

    if surveys:
        t = times_from_surveys(surveys, verbose=verbose)
        if t is not None and len(t) == n:
            print("  [time] taken from the instrument's Time column")
            if len(set(t)) != len(t):
                print("  !  [time] two timesteps share an acquisition time")
            return t

    base = _dt.datetime(2000, 1, 1)
    t = np.array([base + _dt.timedelta(days=i) for i in range(n)])
    print("  !! [time] no real acquisition times found - using placeholders "
          "one day apart.\n"
          "     Set TIMES explicitly if the spacing between surveys matters "
          "for interpretation.")
    return t


def unify_surveys(surveys, tol=0.05, verbose=True):
    """Put every timestep on one shared electrode table.

    Electrode indices are local to each Survey, so before configurations can be
    compared across timesteps they must all refer to the same electrodes.
    """
    allpos = np.vstack([s.pos for s in surveys])
    pos, inv = build_electrode_table(allpos, tol=tol, sort=True)

    off = 0
    for s in surveys:
        n = len(s.pos)
        remap = inv[off:off + n]
        safe = np.where(s.abmn >= 0, s.abmn, 0)
        s.abmn = np.where(s.abmn >= 0, remap[safe], -1)
        s.pos = pos.copy()
        off += n

    if verbose:
        percounts = [len(np.unique(s.abmn[s.abmn >= 0])) for s in surveys]
        print(f"  [tl] shared electrode table: {len(pos)} electrodes "
              f"(per timestep in use: {percounts})")
        if len(set(percounts)) > 1:
            print("  !  [tl] timesteps use different electrode subsets - the "
                  "configuration intersection below will sort that out.")
    return surveys


def _config_keys(survey):
    """Canonical key per measurement: ((min a b, max a b), (min m n, max m n))."""
    a, b, m, n = survey.abmn.T
    cur = np.sort(np.column_stack([a, b]), axis=1)
    pot = np.sort(np.column_stack([m, n]), axis=1)
    return [((int(c[0]), int(c[1])), (int(p[0]), int(p[1])))
            for c, p in zip(cur, pot)]


def match_configurations(surveys, labels=None, verbose=True):
    """Reduce every timestep to the configurations common to all of them.

    Carries over the cross-timestep QC decision from the 2D workflow: a
    measurement rejected in ANY timestep is dropped from EVERY timestep, since
    otherwise a point present at one time and absent at another shows up as
    apparent change. The difference is that matching is done on the electrode
    quadruple rather than on row position, so acquisition order and length may
    differ between files.

    Returns the surveys, each trimmed and reordered to the same common set.
    """
    labels = labels or [f"t{i}" for i in range(len(surveys))]
    maps = []
    for s in surveys:
        keys = _config_keys(s)
        d = {}
        for i, k in enumerate(keys):
            if s.valid[i] and k not in d:
                d[k] = i
        maps.append(d)

    common = set(maps[0])
    for d in maps[1:]:
        common &= set(d)

    # keep the first timestep's ordering for reproducibility
    order = [k for k in _config_keys(surveys[0]) if k in common]
    seen, ordered = set(), []
    for k in order:
        if k not in seen:
            seen.add(k)
            ordered.append(k)

    if verbose:
        print(f"\n  ---- cross-timestep matching ----")
        for lab, s, d in zip(labels, surveys, maps):
            print(f"      {lab}: {len(d)} valid configurations of {s.n_data}")
        print(f"  [tl] common to all {len(surveys)} timesteps: {len(ordered)}")
        for lab, d in zip(labels, maps):
            lost = len(d) - len(ordered)
            if lost:
                print(f"      {lab} loses {lost} not present everywhere")
    if not ordered:
        raise RuntimeError(
            "No configurations are common to all timesteps. Either the "
            "protocols differ between acquisitions, or QC removed disjoint "
            "sets. Check the electrode layout is identical across files.")

    for s, d in zip(surveys, maps):
        s.take([d[k] for k in ordered])
        s.valid[:] = True
    if verbose:
        print(f"  [tl] every timestep now holds {surveys[0].n_data} "
              f"identical configurations")
        print("  ---------------------------------\n")
    return surveys


def to_timelapse(surveys, times, verbose=True):
    """Build the list of DataContainerERT that TimelapseERT expects."""
    alldata = [s.to_container() for s in surveys]
    ns = {d.sensorCount() for d in alldata}
    nd = {d.size() for d in alldata}
    if len(ns) != 1 or len(nd) != 1:
        raise RuntimeError(f"timesteps disagree: sensor counts {ns}, "
                           f"data counts {nd}. Run match_configurations first.")
    if times is not None and len(times) != len(alldata):
        raise ValueError(f"{len(times)} times given for {len(alldata)} files")
    if verbose:
        print(f"  [tl] {len(alldata)} timesteps x {alldata[0].size()} "
              f"measurements on {alldata[0].sensorCount()} electrodes")
    return alldata


def timelapse_coverage(data, mesh, lam=350, verbose=False):
    """Coverage for the timelapse parameter domain.

    fullInversion does not compute sensitivity internally, so this runs a
    single-iteration inversion of one timestep on the same mesh purely to build
    the Jacobian. Use it to mask cells no timestep actually constrained.
    """
    mgr = ert.ERTManager(data)
    mgr.setMesh(mesh)
    mgr.inv.inv.setDeltaPhiAbortPercent(0)
    mgr.invert(mesh=mesh, lam=lam, maxIter=1, verbose=verbose)
    return np.asarray(mgr.coverage()), mgr.paraDomain


def export_timelapse_vtks(models, para_domain, outdir, base_name,
                          all_diffs=True, coverage=None, times=None,
                          verbose=True):
    """One VTK per timestep, carrying the model and its changes.

    Change is expressed as a ratio minus one, as in the 2D workflow: 0 means no
    change, +0.2 means 20% more resistive.
    """
    os.makedirs(outdir, exist_ok=True)
    paths = []
    for i, model in enumerate(models):
        out = pg.Mesh(para_domain)
        out.clearData()

        grad = pg.solver.grad(out, model)
        norm_grad = np.linalg.norm(grad, axis=1)
        rng = norm_grad.max() - norm_grad.min()
        std_grad = (norm_grad - norm_grad.min()) / rng if rng > 0 \
            else np.zeros_like(norm_grad)

        if i > 0:
            if all_diffs:
                for j, other in enumerate(models):
                    if j != i:
                        out[f'diff_t{i}_vs_t{j}'] = (model / other) - 1
            else:
                out['baseline_diff'] = (model / models[0]) - 1
                out['preceding_diff'] = (model / models[i - 1]) - 1

        out['resistivity'] = model
        out['logResistivity'] = np.log10(model)
        out['normGradient'] = norm_grad
        out['stdGradient'] = std_grad
        if coverage is not None and len(coverage) == out.cellCount():
            out['coverage'] = coverage
            out['standardised_coverage'] = np.sign(
                np.abs(coverage - np.median(coverage)))

        stamp = f"{times[i]:%Y%m%d}" if times is not None else f"{i}"
        path = os.path.join(outdir, f"{base_name}_timestep-{i}_{stamp}.vtk")
        out.exportVTK(path)
        paths.append(path)
    if verbose:
        print(f"  [tl] {len(paths)} timestep VTKs written to {outdir}")
    return paths


def timelapse_change_summary(models, para_domain, outdir, base_name,
                             baseline_index=0, from_baseline=True,
                             coverage=None, verbose=True):
    """Average, median and spread of the change across timesteps.

    from_baseline=True compares every timestep against `baseline_index`.
    from_baseline=False uses every ordered pair, giving total change rather
    than change relative to one reference.
    """
    os.makedirs(outdir, exist_ok=True)
    models = np.asarray(models)

    if from_baseline:
        baseline = models[baseline_index]
        others = np.delete(np.arange(len(models)), baseline_index)
        deltas = models[others] / baseline
        tag = 'avgBaseChange'
    else:
        deltas = np.array([models[j] / models[i]
                           for i in range(len(models))
                           for j in range(i + 1, len(models))])
        tag = 'avgTotalChange'

    out = pg.Mesh(para_domain)
    out.clearData()
    out['deltaMean'] = deltas.mean(axis=0) - 1
    out['deltaMed'] = np.median(deltas, axis=0) - 1
    out['deltaStdDev'] = deltas.std(axis=0)
    if coverage is not None and len(coverage) == out.cellCount():
        out['coverage'] = coverage

    path = os.path.join(outdir, f"{base_name}_{tag}.vtk")
    out.exportVTK(path)
    if verbose:
        d = out['deltaMean']
        print(f"  [tl] {tag}: median change {100*np.median(d):+.2f}%, "
              f"5th-95th {100*np.percentile(d, 5):+.1f}% to "
              f"{100*np.percentile(d, 95):+.1f}%")
        print(f"  [tl] written to {path}")
    return path


def plot_timelapse_slices(models, para_domain, outdir, base_name,
                          times=None, z_levels=None, ratio=True,
                          baseline_index=0):
    """Depth slices through each timestep, and through the change."""
    os.makedirs(outdir, exist_ok=True)
    cc = np.array([[c.center().x(), c.center().y(), c.center().z()]
                   for c in para_domain.cells()])
    if z_levels is None:
        z_levels = np.percentile(cc[:, 2], [85, 60, 35])
    dz = max(0.5 * np.diff(np.sort(np.unique(np.round(cc[:, 2], 2)))).mean(),
             0.05)

    nrow, ncol = len(models), len(z_levels)
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.4 * nrow),
                            squeeze=False)
    for i, model in enumerate(models):
        vals = (model / models[baseline_index]) - 1 if (ratio and i > 0) \
            else model
        for j, z in enumerate(z_levels):
            ax = axs[i][j]
            sel = np.abs(cc[:, 2] - z) < dz
            if sel.sum() == 0:
                continue
            if ratio and i > 0:
                lim = np.percentile(np.abs(vals[sel]), 98)
                sc = ax.scatter(cc[sel, 0], cc[sel, 1], c=vals[sel],
                                cmap='RdBu_r', vmin=-lim, vmax=lim, s=10)
                lbl = 'Δρ/ρ₀'
            else:
                sc = ax.scatter(cc[sel, 0], cc[sel, 1], c=np.log10(vals[sel]),
                                cmap='Spectral_r', s=10)
                lbl = 'log₁₀ ρ (Ωm)'
            ax.set_aspect('equal', 'box')
            ax.grid(True, ls='--', alpha=0.3)
            if i == 0:
                ax.set_title(f'z = {z:.2f} m')
            if j == 0:
                t = f"{times[i]:%Y-%m-%d}" if times is not None else f"t{i}"
                ax.set_ylabel(f"{t}\nY (m)")
            fig.colorbar(sc, ax=ax, label=lbl, fraction=0.046)
    fig.supxlabel('X (m)')
    fig.suptitle(f'{base_name}: timelapse depth slices'
                 + (' (row 0 absolute, others relative to baseline)'
                    if ratio else ''), fontweight='bold')
    fig.tight_layout()
    path = os.path.join(outdir, f'{base_name}_timelapse_slices.png')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [tl] slice figure saved to {path}")
    return path