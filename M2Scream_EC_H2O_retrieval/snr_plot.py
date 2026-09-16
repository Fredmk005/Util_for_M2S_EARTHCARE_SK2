"""
Reads storage.pkl and plots, for each stored track:

    1. Distribution of good viewing altitudes  (in-band only)
    2. Distribution of good wavelengths        (in-band only)
    3. The 2D SNR mask                         (in-band only)
    4. Noise vs radiance signal                (in-band only)

Out-of-band bins (transmission <= threshold) are excluded from every
statistic and every plot, because they are not real measurements.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


STORAGE_FILE   = "storage.pkl"
IN_BAND_THRESH = 0.01     # transmission threshold below which a bin is "off"


def get_in_band(data, n_wav):
    """Return boolean (n_wav,) mask of bins the aperture actually transmits."""
    if "transmission" in data:
        T = np.asarray(data["transmission"])
        return T > IN_BAND_THRESH
    fr = np.abs(np.asarray(data["filtered_rad"]))
    col_max = fr.max(axis=1, keepdims=True)
    return (fr > 1e-6 * col_max).any(axis=1)



def plot_track(track_id, data, save=True):
    required = ("mask", "snr", "wavelengths_nm", "viewing_altitude_grid")
    missing = [k for k in required if k not in data]
    if missing:
        print(f"[skip] track {track_id}: missing fields {missing}")
        return

    mask       = np.asarray(data["mask"])
    snr        = np.asarray(data["snr"])
    wav_nm     = np.asarray(data["wavelengths_nm"])
    alt_km     = np.asarray(data["viewing_altitude_grid"])
    lat        = data.get("lat", np.nan)
    lon        = data.get("lon", np.nan)
    min_snr    = data.get("min_snr", np.nan)

    filtered_rad = data.get("filtered_rad", None)
    err_var      = data.get("err_var", None)

    n_wav, n_view = snr.shape
    mask_2d = mask.reshape(n_wav, n_view)

    in_band = get_in_band(data, n_wav)
    n_in_band = int(in_band.sum())

    mask_2d_in    = mask_2d[in_band, :]
    snr_in        = snr[in_band, :]
    wav_in        = wav_nm[in_band]
    n_wav_in      = n_in_band

    good_per_wav   = mask_2d_in.sum(axis=1)
    good_per_alt   = mask_2d_in.sum(axis=0)
    good_altitudes = good_per_alt > 0
    good_wavs      = good_per_wav > 0

    dw = (wav_in[1] - wav_in[0]) if len(wav_in) > 1 else 1.0
    da = (alt_km[1] - alt_km[0]) if len(alt_km) > 1 else 1.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(alt_km, good_per_alt, width=da * 0.9,
           color="steelblue", edgecolor="k")
    ax.set_xlabel("Tangent altitude (km)")
    ax.set_ylabel(f"Good in-band wavelengths per altitude (max {n_wav_in})")
    ax.set_title(f"Good-altitude distribution  "
                 f"({good_altitudes.sum()}/{n_view} altitudes usable)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.imshow(
        mask_2d_in.T, aspect="auto", origin="lower",
        extent=[wav_in[0], wav_in[-1], alt_km[0], alt_km[-1]],
        cmap="Greys", vmin=0, vmax=1
    )
    ax.set_xlabel("Wavelength (nm)  [in-band only]")
    ax.set_ylabel("Tangent altitude (km)")
    ax.set_title("SNR mask (white = good)")

    fig.suptitle(f"Track {track_id}   lat={lat:.3f}, lon={lon:.3f}   "
                 f"in-band: {n_in_band}/{n_wav} wavelengths")
    plt.tight_layout()
    if save:
        plt.savefig(f"good_altitudes_track{track_id}.png", dpi=150)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.bar(wav_in, good_per_wav, width=dw * 0.9,
           color="darkorange", edgecolor="k")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(f"Good altitudes per wavelength (max {n_view})")
    ax.set_title(f"Good-wavelength distribution  "
                 f"({good_wavs.sum()}/{n_wav_in} in-band wavelengths usable)")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    ax.plot(wav_in, good_per_wav / n_view, "k-", lw=1.5)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Fraction of altitudes usable")
    ax.set_ylim(0, 1.05)
    ax.set_title("Usable-altitude fraction per wavelength (in-band only)")
    ax.grid(alpha=0.3)

    fig.suptitle(f"Track {track_id}   lat={lat:.3f}, lon={lon:.3f}")
    plt.tight_layout()
    if save:
        plt.savefig(f"good_wavelengths_track{track_id}.png", dpi=150)
    plt.show()

    if filtered_rad is None or err_var is None:
        print(f"[skip] track {track_id}: missing filtered_rad or err_var.")
    else:
        filtered_rad = np.asarray(filtered_rad)
        err_var      = np.asarray(err_var)
        sigma_rad    = np.sqrt(np.maximum(err_var, 0.0))

        fr_in      = filtered_rad[in_band, :]
        sig_in     = sigma_rad[in_band, :]
        n_in       = fr_in.shape[0]

        iw_local = int(np.argmax(np.nanmedian(np.abs(fr_in), axis=1)))
        iw_global = int(np.where(in_band)[0][iw_local])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.errorbar(
            alt_km, fr_in[iw_local, :],
            yerr=sig_in[iw_local, :],
            fmt="o-", ms=3, lw=1, capsize=2,
            label=f"Signal ± 1σ  ({wav_nm[iw_global]:.2f} nm)"
        )
        ax.set_xlabel("Tangent altitude (km)")
        ax.set_ylabel("Filtered radiance (per bin)")
        ax.set_title(f"Noise vs radiance signal, track {track_id}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        if save:
            plt.savefig(f"noise_vs_radiance_track{track_id}.png", dpi=150)
        plt.show()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        signal_abs = np.abs(fr_in)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_unc = np.where(signal_abs > 0, sig_in / signal_abs, np.nan)

        ax = axes[0]
        finite = rel_unc[np.isfinite(rel_unc)]
        vmax = np.nanpercentile(finite, 95) if finite.size else 1.0
        im = ax.imshow(
            rel_unc.T, aspect="auto", origin="lower",
            extent=[wav_in[0], wav_in[-1], alt_km[0], alt_km[-1]],
            vmin=0, vmax=vmax
        )
        ax.set_xlabel("Wavelength (nm)  [in-band only]")
        ax.set_ylabel("Tangent altitude (km)")
        ax.set_title("Relative uncertainty  σ(L)/|L|")
        fig.colorbar(im, ax=ax, label="σ / |L|")

        ax = axes[1]
        wav_picks = np.unique(np.linspace(0, n_in - 1,
                                          min(5, n_in)).astype(int))
        for iw in wav_picks:
            ax.plot(rel_unc[iw, :], alt_km, lw=1.5,
                    label=f"{wav_in[iw]:.2f} nm")
        ax.axvline(1.0, color="r", ls="--", lw=1, label="σ = |L|")
        ax.set_xlabel("σ(L) / |L|")
        ax.set_ylabel("Tangent altitude (km)")
        ax.set_xscale("log")
        ax.set_title("Relative uncertainty vs altitude (in-band only)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

        fig.suptitle(f"Track {track_id}   lat={lat:.3f}, lon={lon:.3f}")
        plt.tight_layout()
        if save:
            plt.savefig(f"rel_uncertainty_track{track_id}.png", dpi=150)
        plt.show()

        # ---- 3d. Signal vs noise magnitude, brightest bin ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(alt_km, np.abs(fr_in[iw_local, :]),
                    "k-", lw=2, label=f"|L| ({wav_nm[iw_global]:.2f} nm)")
        ax.semilogy(alt_km, sig_in[iw_local, :],
                    "r-", lw=2, label=f"σ(L) ({wav_nm[iw_global]:.2f} nm)")
        ax.set_xlabel("Tangent altitude (km)")
        ax.set_ylabel("Radiance (log scale)")
        ax.set_title(f"Signal vs noise magnitude, track {track_id}")
        ax.legend()
        ax.grid(alpha=0.3, which="both")
        plt.tight_layout()
        if save:
            plt.savefig(f"signal_vs_noise_magnitude_track{track_id}.png", dpi=150)
        plt.show()


    n_total_in_band = n_in_band * n_view
    n_good_in_band  = int(mask_2d_in.sum())

    print(f"\n{'-'*60}")
    print(f"Track {track_id}  lat={lat:.3f}, lon={lon:.3f}")
    print(f"min_snr threshold:              {min_snr}")
    print(f"In-band wavelengths:            {n_in_band}/{n_wav}")
    print(f"In-band measurements:           {n_total_in_band}")
    print(f"In-band good measurements:      {n_good_in_band}  "
          f"({100 * n_good_in_band / max(n_total_in_band, 1):.1f}%)")
    print(f"Usable altitudes:               {good_altitudes.sum()}/{n_view}  "
          f"({100 * good_altitudes.sum() / n_view:.1f}%)")
    print(f"Usable in-band wavelengths:     {good_wavs.sum()}/{n_wav_in}  "
          f"({100 * good_wavs.sum() / max(n_wav_in, 1):.1f}%)")
    if err_var is not None:
        fr_in_abs = np.abs(filtered_rad[in_band, :])
        sig_in    = np.sqrt(np.maximum(err_var[in_band, :], 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(fr_in_abs > 0, sig_in / fr_in_abs, np.nan)
        finite = rel[np.isfinite(rel)]
        if finite.size:
            print(f"σ(L)/|L| median:{np.nanmedian(finite):.3e}")
            print(f"σ(L)/|L| 5th/95th pc:"
                  f"{np.nanpercentile(finite, 5):.3e} / "
                  f"{np.nanpercentile(finite, 95):.3e}")
    print(f"{'-'*60}\n")



def plot_combined(storage):
    fig, ax = plt.subplots(figsize=(8, 6))
    any_plotted = False

    for track_id, data in storage.items():
        if "mask" not in data or "viewing_altitude_grid" not in data:
            continue
        mask   = np.asarray(data["mask"])
        alt_km = np.asarray(data["viewing_altitude_grid"])
        n_wav, n_view = data["snr"].shape

        in_band = get_in_band(data, n_wav)
        n_in = int(in_band.sum())
        if n_in == 0:
            continue

        mask_2d = mask.reshape(n_wav, n_view)[in_band, :]
        frac = mask_2d.sum(axis=0) / n_in

        ax.plot(frac, alt_km, lw=1, label=f"{track_id}")
        any_plotted = True

    if not any_plotted:
        print("No tracks with mask data for combined plot.")
        plt.close(fig)
        return

    ax.set_xlabel("Fraction of in-band wavelengths usable per altitude")
    ax.set_ylabel("Tangent altitude (km)")
    ax.set_xlim(0, 1.05)
    ax.set_title("Usable-altitude fraction, all stored tracks (in-band only)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("good_altitudes_combined.png", dpi=150)
    plt.show()

def main():
    with open(STORAGE_FILE, "rb") as f:
        storage = pickle.load(f)

    print(f"Loaded {len(storage)} tracks from {STORAGE_FILE}")

    for track_id, data in storage.items():
        plot_track(track_id, data)

    plot_combined(storage)


if __name__ == "__main__":
    main()