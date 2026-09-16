import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from config import MERRA_CONFIG

#TODO: Clean up redundant variables

plot_min = 15
plot_max = 25
def load_results(filename="storage.pkl"):
    with open(filename, "rb") as f:
        return pickle.load(f)

def plot_single_retrieval(data, along_track, save=True):
    DOFS = data["DOFS"]
    altitude_km = data["altitude_grid"]
    h2o_true = data["h2o_truth"] * 1e6
    h2o_apriori = data["h2o_apriori"] * 1e6
    h2o_retrieved = data["h2o_retrieved"] * 1e6
    lat = data["lat"]
    lon = data["lon"]
    tropo_height_km = data.get("tropopause_height_km", 12.0)
    time = MERRA_CONFIG["date"]

    plot_mask = (altitude_km>=plot_min) & (altitude_km<=plot_max)
        
    A_full = data["averaging_kernel"]
    row_sums_full = data["row_sums"]

    n_alt = len(altitude_km)
    n_ak = A_full.shape[0]

    print(f"Track {along_track}: altitude={n_alt}, AK={A_full.shape}, row_sums={len(row_sums_full)}")

    if n_alt == n_ak:
        A = A_full
        row_sums = row_sums_full
        altitude_ak = altitude_km

    else:
        # Use the AK grid size
        n = min(n_alt, n_ak, len(row_sums_full))

        indices = np.linspace(0, n_ak - 1, n, dtype=int)

        A = A_full[np.ix_(indices, indices)]
        row_sums = row_sums_full[indices]

        # Select matching altitude points
        altitude_ak = np.linspace(
            altitude_km[0],
            altitude_km[-1],
            n
        )
    
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = np.clip(A, 0, 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(h2o_true[plot_mask], altitude_km[plot_mask], color="black", linewidth=2, label="Truth")
    ax.plot(h2o_apriori[plot_mask], altitude_km[plot_mask], "--", color="gray", linewidth=2, label="A priori")
    ax.plot(h2o_retrieved[plot_mask], altitude_km[plot_mask], color="tab:blue", linewidth=2, label="Retrieved")
    ax.axhline(y=tropo_height_km, color='red', linestyle=':', linewidth=2, alpha=0.8, label='Tropopause')
    ax.set_xlabel("H$_2$O ppmv")
    ax.set_ylabel("Altitude (km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[0, 1]
    extent = [altitude_ak[0], altitude_ak[-1], altitude_ak[0], altitude_ak[-1]]
    im = ax.imshow(A, origin="lower", aspect="auto", extent=extent, cmap="viridis")
    ax.axhline(y=tropo_height_km, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax.axvline(x=tropo_height_km, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax.set_xlabel("True altitude (km)")
    ax.set_ylabel("Retrieved altitude (km)")
    plt.colorbar(im, ax=ax, label="Averaging kernel")
    
    ax = axes[1, 0]
    ax.plot(altitude_ak, row_sums, color="tab:red", linewidth=2)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Ideal (sum=1)')
    ax.axvline(x=tropo_height_km, color='red', linestyle=':', linewidth=2, alpha=0.8)
    ax.set_xlabel("Altitude (km)")
    ax.set_ylabel("Row Sum")
    ax.set_title("Averaging Kernel Row Sums")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1, 1]
    ax.text(0.5, 0.5, f'DOFS = {DOFS:.2f}\nLat = {lat:.2f}°\nLon = {lon:.2f}°\ndate={time}', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    ax.axis('off')
    
    fig.suptitle(f"H$_2$O Retrieval, Track = {along_track} | DOFS = {DOFS:.2f} | Tropopause = {tropo_height_km:.1f} km", fontsize=14)
    plt.tight_layout()
    
    if save:
        filename = f"track={along_track}_lat={lat:.2f}_lon={lon:.2f}.jpeg"
        plt.savefig(filename, dpi=300, transparent=True, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_kernel_rows(data, along_track, save=True):
    A_full = np.asarray(data["averaging_kernel"])
    altitude_km = np.asarray(data["altitude_grid"])
    row_sums_full = np.asarray(data["row_sums"])

    tropo_height_km = data.get("tropopause_height_km", 12.0)
    lat = data["lat"]
    lon = data["lon"]
    DOFS = data["DOFS"]

    plot_mask = (altitude_km >= plot_min) & (altitude_km <= plot_max)

    n_alt = len(altitude_km)
    n_ak = A_full.shape[0]
    n_rows = len(row_sums_full)

    print(
        f"Track {along_track}: "
        f"altitude={n_alt}, "
        f"AK={A_full.shape}, "
        f"row_sums={n_rows}"
    )

    if n_alt == n_ak and n_rows == n_alt:

        A = A_full
        row_sums = row_sums_full
        altitude_ak = altitude_km

    else:

        n = min(n_alt, n_ak, n_rows)

        # Select evenly spaced AK rows/columns
        indices = np.linspace(
            0,
            n_ak - 1,
            n,
            dtype=int
        )

        A = A_full[np.ix_(indices, indices)]
        row_sums = row_sums_full[
            np.linspace(
                0,
                n_rows - 1,
                n,
                dtype=int
            )
        ]

        # Approximate altitude grid for AK
        altitude_ak = np.linspace(
            altitude_km[0],
            altitude_km[-1],
            n
        )

    A = np.nan_to_num(
        A,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    A = np.clip(A, 0, 1)

    fig_kernel_rows, ax_kernel = plt.subplots(
        figsize=(12, 8)
    )

    # Find AK indices corresponding to 10–25 km
    ak_plot_mask = (
        (altitude_ak >= plot_min)
        & (altitude_ak <= plot_max)
    )

    ak_indices = np.where(ak_plot_mask)[0]

    for idx in ak_indices:

        ax_kernel.plot(
            altitude_ak,
            A[idx, :],
            label=f"{altitude_ak[idx]:.1f} km",
            linewidth=1.5
        )

    # Tropopause
    ax_kernel.axvline(
        x=tropo_height_km,
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.8,
        label=f"Tropopause = {tropo_height_km:.1f} km"
    )

    ax_kernel.set_xlabel("Altitude (km)")
    ax_kernel.set_ylabel("Averaging Kernel Value")

    ax_kernel.set_title(
        f"Averaging Kernel Rows - "
        f"Track {along_track}, "
        f"DOFS = {DOFS:.2f}"
    )

    ax_kernel.grid(True, alpha=0.3)

    ax_kernel.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=8
    )

    if save:

        filename_kernel_rows = (
            f"kernel_rows_track={along_track}"
            f"_lat={lat:.2f}"
            f"_lon={lon:.2f}.jpeg"
        )

        plt.savefig(
            filename_kernel_rows,
            dpi=300,
            transparent=True,
            bbox_inches="tight"
        )

        plt.close()

    else:
        plt.show()

def plot_heatmap(storage, save=True):
    atmo_track = sorted(storage.keys())
    
    if len(storage) < 2:
        print(f"Need at least 2 tracks to create heatmap. Only have {len(storage)}.")
        return
    
    first_track = storage[atmo_track[0]]
    altitude_km = first_track["altitude_grid"]
    plot_mask = (altitude_km>=plot_min) & (altitude_km<=plot_max)
    n_alt = len(altitude_km[plot_mask])
    n_total = len(atmo_track)
    
    h2o_2d = np.full((n_total, n_alt), np.nan)
    lats = np.full(n_total, np.nan)
    dofs = np.full(n_total, np.nan)
    tropo_heights = np.full(n_total, np.nan)
    
    for i, track_idx in enumerate(atmo_track):
        data = storage[track_idx]
        
        h2o_2d[i, :] = np.interp(
            altitude_km[plot_mask],
            data["altitude_grid"],
            data["h2o_retrieved"] * 1e6,
        )

        
        lats[i] = data["lat"]
        dofs[i] = data["DOFS"]
        tropo_heights[i] = data.get("tropopause_height_km", 12.0)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.pcolormesh(
        np.arange(n_total),
        altitude_km[plot_mask],
        h2o_2d.T,
        shading="nearest",
        cmap="viridis",
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=200)
    )
    
    for i, tropo in enumerate(tropo_heights):
        ax.plot([i-0.5, i+0.5], [tropo, tropo], color='red', linestyle=':', linewidth=2, alpha=0.8)
    
    ax.set_xlabel("Track Index")
    ax.set_ylabel("Altitude (km)")
    ax.set_title("Retrieved H$_2$O VMR (ppmv)")
    ax.set_xticks(np.arange(n_total))
    ax.set_xticklabels(atmo_track)
    
    ax2 = ax.twiny()
    ax2.set_xticks(np.arange(n_total))
    ax2.set_xticklabels([f"{lat:.1f}°" for lat in lats], rotation=45, fontsize=8)
    ax2.set_xlabel("Latitude (°)")
    
    plt.colorbar(im, ax=ax, label="H$_2$O (ppmv)")
    
    for i, dof in enumerate(dofs):
        ax.text(
            i,
            altitude_km[plot_mask][-1] + 0.3,
            f"{dof:.1f}",
            ha="center",
            fontsize=7
        )
    
    plt.tight_layout()
    
    if save:
        plt.savefig("h2o_heatmap.jpeg", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

if __name__ == "__main__":
    storage = load_results("storage.pkl")
    
    for track_idx, data in storage.items():
        plot_single_retrieval(data, track_idx)
        plot_kernel_rows(data, track_idx)
    
    plot_heatmap(storage)