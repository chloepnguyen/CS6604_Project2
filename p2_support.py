"""Provided infrastructure for CS 6640 Project 2 (Spectral Filtering by Design).

Everything in this module is *provided scaffold*: deterministic data loading,
display helpers, metric helpers, the spatial reference convolution, and
correctness checks. None of it performs the project-specific work you are
asked to do in ``driver.ipynb``.

Conventions used throughout
---------------------------
* Images are ``float64`` arrays on the ``[0, 1]`` intensity scale with shape
  ``(M, N)``; row index ``m`` and column index ``n``.
* The two-dimensional Discrete Fourier Transform (DFT) follows the course
  convention: unnormalized forward transform, ``1/(MN)`` on the inverse. NumPy's
  ``np.fft.fft2`` / ``np.fft.ifft2`` implement exactly this convention.
* "Centered" frequency coordinates are the coordinates of the array produced by
  ``np.fft.fftshift``. For an even size ``M`` the centered row-frequency
  coordinate of array row ``i`` is ``u = i - M // 2`` (so DC sits at
  ``(M // 2, N // 2)``); likewise ``v = j - N // 2`` for columns.
* Numerical arrays are **never clipped** before transforms or metrics. Display
  limits are applied only inside plotting calls.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.metrics import peak_signal_noise_ratio

# ---------------------------------------------------------------------------
# Fixed experimental constants (not assessed choices)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256
"""Side length of the deterministic center crop used for every image."""

APPROVED_FAMILIES = ("hard", "gaussian")
"""Notch families in the approved parameter study."""

APPROVED_WIDTHS = (2, 4, 8)
"""Notch widths ``w`` (in frequency bins) in the approved parameter study."""

NTBSE_RADIUS = 10
"""Radius (in frequency bins) of each target neighborhood used by NTBSE."""

BOX_KERNEL_5x5 = np.full((5, 5), 1.0 / 25.0)
"""Normalized 5x5 box kernel used for the visible calibration experiment."""

DIAGNOSTIC_KERNEL = 0.25 * np.array(
    [[0.0, 0.0, 0.0],
     [0.0, 1.0, 2.0],
     [0.0, 1.0, 0.0]]
)
"""Asymmetric 3x3 diagnostic kernel; its origin is the center element."""

CONVOLUTION_TOLERANCE = 1e-9
"""Maximum absolute error allowed against the spatial reference (float64)."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def center_crop(image: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Return the deterministic ``size x size`` center crop of a 2-D array."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("center_crop expects a 2-D grayscale array")
    rows, cols = image.shape
    if rows < size or cols < size:
        raise ValueError(f"image {image.shape} is smaller than the {size}x{size} crop")
    top = (rows - size) // 2
    left = (cols - size) // 2
    return image[top:top + size, left:left + size].copy()


def load_image(name: str) -> np.ndarray:
    """Load a built-in scikit-image test image as a float64 [0, 1] 256x256 crop.

    Parameters
    ----------
    name:
        ``"camera"`` (design/development image) or ``"coins"`` (held-out image).
    """
    loaders = {"camera": data.camera, "coins": data.coins}
    if name not in loaders:
        raise ValueError(f"unknown image {name!r}; expected one of {sorted(loaders)}")
    image = loaders[name]()
    if image.ndim == 3:
        image = rgb2gray(image)
    image = img_as_float(image).astype(np.float64)
    return center_crop(image, IMAGE_SIZE)


# ---------------------------------------------------------------------------
# Frequency coordinates and display mappings
# ---------------------------------------------------------------------------

def centered_frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return centered frequency-bin coordinate arrays ``(U, V)`` for ``shape``.

    ``U[i, j]`` is the centered row-frequency coordinate ``u`` and ``V[i, j]``
    the centered column-frequency coordinate ``v`` of the element at ``(i, j)``
    of an ``np.fft.fftshift``-ed spectrum. Both are float64 so that distances
    to non-integer (off-grid) locations can be computed directly.
    """
    rows, cols = shape
    u = np.arange(rows, dtype=np.float64) - rows // 2
    v = np.arange(cols, dtype=np.float64) - cols // 2
    return np.meshgrid(u, v, indexing="ij")


def log_magnitude(spectrum: np.ndarray) -> np.ndarray:
    """Display mapping ``log(1 + |F|)``. Visualization only, not a new Fourier quantity."""
    return np.log1p(np.abs(spectrum))


def centered_to_array_index(center: tuple[float, float], shape: tuple[int, int]) -> tuple[float, float]:
    """Convert a centered ``(u, v)`` location to (row, column) plotting coordinates."""
    rows, cols = shape
    return center[0] + rows // 2, center[1] + cols // 2


# ---------------------------------------------------------------------------
# Metrics (metric implementation is not an assessed capability)
# ---------------------------------------------------------------------------

def mse(clean: np.ndarray, result: np.ndarray) -> float:
    """Mean squared error between two same-shape arrays (no clipping)."""
    clean = np.asarray(clean, dtype=np.float64)
    result = np.asarray(result, dtype=np.float64)
    _check_same_shape(clean, result)
    return float(np.mean((result - clean) ** 2))


def rmse(clean: np.ndarray, result: np.ndarray) -> float:
    """Root mean squared error; same units as the image intensities."""
    return float(np.sqrt(mse(clean, result)))


def psnr(clean: np.ndarray, result: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB with ``data_range=1.0`` (no clipping)."""
    clean = np.asarray(clean, dtype=np.float64)
    result = np.asarray(result, dtype=np.float64)
    _check_same_shape(clean, result)
    return float(peak_signal_noise_ratio(clean, result, data_range=1.0))


def target_band_mask(shape: tuple[int, int], centers: Sequence[tuple[float, float]],
                     radius: float = NTBSE_RADIUS) -> np.ndarray:
    """Boolean mask of centered bins within ``radius`` of any supplied center."""
    centers = _validate_centers(centers)
    U, V = centered_frequency_grid(shape)
    region = np.zeros(shape, dtype=bool)
    for cu, cv in centers:
        region |= (U - cu) ** 2 + (V - cv) ** 2 <= radius ** 2
    return region


def ntbse(clean: np.ndarray, result: np.ndarray, corrupted: np.ndarray,
          centers: Sequence[tuple[float, float]], radius: float = NTBSE_RADIUS) -> float:
    """Normalized target-band spectral error (NTBSE).

    ``centers`` are centered ``(u, v)`` frequency locations that **you** supply;
    this helper never fills them in. The target region is the union of circular
    neighborhoods of ``radius`` bins around those centers. The returned value is
    the target-band energy of the spectrum of ``result - clean`` divided by the
    target-band energy of the spectrum of ``corrupted - clean``.
    """
    clean = np.asarray(clean, dtype=np.float64)
    result = np.asarray(result, dtype=np.float64)
    corrupted = np.asarray(corrupted, dtype=np.float64)
    _check_same_shape(clean, result)
    _check_same_shape(clean, corrupted)
    region = target_band_mask(clean.shape, centers, radius)
    if not region.any():
        raise ValueError("the target region is empty; check the supplied centers")
    error_spectrum = np.fft.fftshift(np.fft.fft2(result - clean))
    baseline_spectrum = np.fft.fftshift(np.fft.fft2(corrupted - clean))
    denominator = float(np.sum(np.abs(baseline_spectrum[region]) ** 2))
    if denominator == 0.0:
        raise ValueError("the corrupted-image error has no energy in the target region")
    return float(np.sum(np.abs(error_spectrum[region]) ** 2) / denominator)


# ---------------------------------------------------------------------------
# Spatial reference convolution and correctness checks
# ---------------------------------------------------------------------------

def spatial_reference_convolution(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Zero-boundary, same-size linear convolution (SciPy reference).

    Uses ``scipy.signal.convolve2d(..., mode="same", boundary="fill", fillvalue=0)``.
    For an odd-sized kernel the kernel origin is its center element.
    """
    image = np.asarray(image, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    return convolve2d(image, kernel, mode="same", boundary="fill", fillvalue=0.0)


def check_linear_convolution(convolve_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             image: np.ndarray, kernel: np.ndarray,
                             tolerance: float = CONVOLUTION_TOLERANCE,
                             label: str = "") -> dict:
    """Compare ``convolve_fn(image, kernel)`` with the spatial reference.

    Returns only summary information (shape check, dtype, maximum absolute
    error, pass/fail). The reference array itself is not returned.
    """
    output = np.asarray(convolve_fn(image, kernel))
    if output.shape != image.shape:
        raise AssertionError(f"{label}: output shape {output.shape} != image shape {image.shape}")
    if np.iscomplexobj(output):
        raise AssertionError(f"{label}: return the real part of the result (the output is complex)")
    if not np.all(np.isfinite(output)):
        raise AssertionError(f"{label}: output contains non-finite values")
    error = float(np.max(np.abs(output.astype(np.float64) - spatial_reference_convolution(image, kernel))))
    summary = {"label": label, "max_abs_error": error, "passed": error <= tolerance,
               "tolerance": tolerance}
    status = "PASS" if summary["passed"] else "FAIL"
    print(f"[{status}] {label}: max |FFT result - reference| = {error:.3e} (tolerance {tolerance:.0e})")
    return summary


def check_real_output(complex_result: np.ndarray, relative_tolerance: float = 1e-10) -> float:
    """Report the largest imaginary magnitude of an inverse-transform result.

    A real-valued frequency response that respects the conjugate structure of
    a real image yields an inverse transform whose imaginary part is only
    floating-point round-off. Returns ``max|Im| / max|Re|`` and raises an
    ``AssertionError`` if that ratio exceeds ``relative_tolerance``.
    """
    complex_result = np.asarray(complex_result)
    if not np.iscomplexobj(complex_result):
        raise TypeError("pass the complex array returned by the inverse FFT, before taking .real")
    scale = max(float(np.max(np.abs(complex_result.real))), np.finfo(float).tiny)
    ratio = float(np.max(np.abs(complex_result.imag))) / scale
    if ratio > relative_tolerance:
        raise AssertionError(
            f"imaginary part is not negligible (max|Im|/max|Re| = {ratio:.2e}); "
            "check the frequency response and the transform sequence")
    return ratio


def assert_image_like(array: np.ndarray, shape: Optional[tuple[int, int]] = None, name: str = "array") -> None:
    """Assert that ``array`` is a finite, real, 2-D float array (optionally of ``shape``)."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise AssertionError(f"{name} must be 2-D, got shape {array.shape}")
    if shape is not None and array.shape != tuple(shape):
        raise AssertionError(f"{name} must have shape {tuple(shape)}, got {array.shape}")
    if np.iscomplexobj(array):
        raise AssertionError(f"{name} must be real-valued")
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains non-finite values")


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------

@dataclass
class FilterResult:
    """One filtering outcome. Fill every field from your own computations."""

    image: str
    family: str
    width: float
    psnr_db: float
    ntbse: float
    output_min: float
    output_max: float
    note: str = ""


def results_table(records: Iterable[FilterResult]) -> str:
    """Format result records as a Markdown table (for pasting into the report)."""
    records = list(records)
    header = "| Image | Family | Width w (bins) | PSNR (dB) | NTBSE | Output min | Output max | Note |"
    rule = "|---|---|---:|---:|---:|---:|---:|---|"
    rows = [
        f"| {r.image} | {r.family} | {r.width:g} | {r.psnr_db:.2f} | {r.ntbse:.4f} | "
        f"{r.output_min:.3f} | {r.output_max:.3f} | {r.note} |"
        for r in records
    ]
    return "\n".join([header, rule, *rows])


# ---------------------------------------------------------------------------
# Plotting helpers (display conventions from the project specification)
# ---------------------------------------------------------------------------

def show_images(images: Sequence[np.ndarray], titles: Sequence[str], suptitle: str = "",
                ncols: Optional[int] = None, figsize_per_panel: float = 3.2):
    """Show images on the common display range ``[0, 1]`` (display only)."""
    fig, axes = _panel_grid(len(images), ncols, figsize_per_panel)
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    _finish(fig, axes, len(images), suptitle)
    return fig


def show_error_images(errors: Sequence[np.ndarray], titles: Sequence[str], suptitle: str = "",
                      ncols: Optional[int] = None, figsize_per_panel: float = 3.2):
    """Show error images on one common symmetric scale ``[-e_max, +e_max]``."""
    e_max = max(float(np.max(np.abs(e))) for e in errors)
    e_max = e_max if e_max > 0 else 1.0
    fig, axes = _panel_grid(len(errors), ncols, figsize_per_panel)
    last = None
    for ax, error, title in zip(axes, errors, titles):
        last = ax.imshow(error, cmap="RdBu_r", vmin=-e_max, vmax=e_max, interpolation="nearest")
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    if last is not None:
        fig.colorbar(last, ax=list(axes[:len(errors)]), shrink=0.8, label="error (common symmetric scale)")
    _finish(fig, axes, len(errors), suptitle, tight=False)
    return fig


def show_spectra(spectra: Sequence[np.ndarray], titles: Sequence[str], suptitle: str = "",
                 centers: Optional[Sequence[tuple[float, float]]] = None,
                 ncols: Optional[int] = None, figsize_per_panel: float = 3.2):
    """Show centered spectra as ``log(1 + |F|)`` with one common display normalization.

    ``spectra`` must already be centered (``np.fft.fftshift``). If ``centers``
    is supplied (centered ``(u, v)`` locations you provide), they are overlaid
    as open circles on every panel.
    """
    shown = [log_magnitude(s) for s in spectra]
    vmax = max(float(np.max(s)) for s in shown)
    fig, axes = _panel_grid(len(shown), ncols, figsize_per_panel)
    for ax, image, title, spectrum in zip(axes, shown, titles, spectra):
        ax.imshow(image, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
        if centers:
            for center in centers:
                row, col = centered_to_array_index(center, spectrum.shape)
                ax.plot(col, row, marker="o", markersize=11, markerfacecolor="none",
                        markeredgecolor="cyan", markeredgewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("v (centered column-frequency bin)", fontsize=7)
        ax.set_ylabel("u (centered row-frequency bin)", fontsize=7)
        _centered_ticks(ax, spectrum.shape)
    _finish(fig, axes, len(shown), suptitle)
    return fig


def show_mask(mask: np.ndarray, title: str = "Frequency response H[u, v] (centered)"):
    """Show a centered real-valued frequency response on the fixed range [0, 1]."""
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(mask, cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("v (centered column-frequency bin)", fontsize=7)
    ax.set_ylabel("u (centered row-frequency bin)", fontsize=7)
    _centered_ticks(ax, mask.shape)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_parameter_study(records: Sequence[FilterResult], suptitle: str = "Parameter study"):
    """Two-panel summary of PSNR and NTBSE across completed result records.

    Consumes records you have already computed; it does not run any filtering.
    """
    records = list(records)
    if not records:
        raise ValueError("no result records supplied")
    families = sorted({r.family for r in records})
    markers = {"hard": "s", "gaussian": "o"}
    linestyles = {"hard": "--", "gaussian": "-"}
    fig, (ax_psnr, ax_ntbse) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for family in families:
        rows = sorted((r for r in records if r.family == family), key=lambda r: r.width)
        widths = [r.width for r in rows]
        ax_psnr.plot(widths, [r.psnr_db for r in rows], marker=markers.get(family, "^"),
                     linestyle=linestyles.get(family, ":"), label=family)
        ax_ntbse.plot(widths, [r.ntbse for r in rows], marker=markers.get(family, "^"),
                      linestyle=linestyles.get(family, ":"), label=family)
    for ax, label in [(ax_psnr, "PSNR (dB), higher is better"), (ax_ntbse, "NTBSE, lower is better")]:
        ax.set_xlabel("width w (frequency bins)")
        ax.set_ylabel(label)
        ax.set_xticks(sorted({r.width for r in records}))
        ax.grid(alpha=0.3)
        ax.legend(title="family")
    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_same_shape(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")


def _validate_centers(centers: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if centers is None:
        raise ValueError("supply the centered (u, v) interference centers you identified")
    out = [(float(c[0]), float(c[1])) for c in centers]
    if not out:
        raise ValueError("supply at least one centered (u, v) center")
    return out


def _panel_grid(count: int, ncols: Optional[int], size: float):
    ncols = ncols or min(count, 4)
    nrows = int(np.ceil(count / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(size * ncols, size * nrows), squeeze=False)
    return fig, axes.ravel()


def _finish(fig, axes, used: int, suptitle: str, tight: bool = True) -> None:
    for ax in axes[used:]:
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle)
    if tight:
        fig.tight_layout()


def _centered_ticks(ax, shape: tuple[int, int]) -> None:
    rows, cols = shape
    tick_u = [-rows // 2, -rows // 4, 0, rows // 4, rows // 2 - 1]
    tick_v = [-cols // 2, -cols // 4, 0, cols // 4, cols // 2 - 1]
    ax.set_yticks([t + rows // 2 for t in tick_u])
    ax.set_yticklabels([str(t) for t in tick_u], fontsize=6)
    ax.set_xticks([t + cols // 2 for t in tick_v])
    ax.set_xticklabels([str(t) for t in tick_v], fontsize=6)


__all__ = [
    "IMAGE_SIZE", "APPROVED_FAMILIES", "APPROVED_WIDTHS", "NTBSE_RADIUS",
    "BOX_KERNEL_5x5", "DIAGNOSTIC_KERNEL", "CONVOLUTION_TOLERANCE",
    "center_crop", "load_image", "centered_frequency_grid", "log_magnitude",
    "centered_to_array_index", "mse", "rmse", "psnr", "target_band_mask", "ntbse",
    "spatial_reference_convolution", "check_linear_convolution", "check_real_output",
    "assert_image_like", "FilterResult", "results_table", "show_images",
    "show_error_images", "show_spectra", "show_mask", "plot_parameter_study",
]
