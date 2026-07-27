import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# --------------------------------------------------
# Paths
# --------------------------------------------------
NC_PATH = "../dataset/california_monthly.nc"
OUT_DIR = Path("./Supplementary")
OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
ds = xr.open_dataset(NC_PATH)

years = ds["year"].values
months = ds["month"].values

time = pd.date_range(
    start=f"{years.min()}-01-01",
    periods=len(years) * len(months),
    freq="MS"
)

scdhi = ds["scdhi"].values.flatten()
sdhi = ds["sdhi2"].values.flatten()
scei = ds["scei"].values.flatten()

df = pd.DataFrame({
    "Time": time,
    "SCDHI": ds["scdhi"].values,
    "SDHI": ds["sdhi2"].values,
    "SCEI": ds["scei"].values
})

# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.8))

ax.plot(
    df["Time"],
    df["SCDHI"],
    lw=2.2,
    label="SCDHI"
)

ax.plot(
    df["Time"],
    df["SDHI"],
    lw=1.8,
    alpha=0.85,
    label="SDHI"
)

ax.plot(
    df["Time"],
    df["SCEI"],
    lw=1.8,
    alpha=0.85,
    label="SCEI"
)

# --------------------------------------------------
# Formatting
# --------------------------------------------------
ax.set_ylabel("Standardized Index")

ax.set_xlabel("Time (Year)")

ax.xaxis.set_major_locator(
    mdates.YearLocator(10)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%Y")
)

ax.grid(
    True,
    linestyle="--",
    linewidth=0.5,
    alpha=0.4
)

ax.legend(
    frameon=False,
    ncol=3,
    loc="upper right"
)

ax.margins(x=0)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "Figure_S2_Compound_Indices.png",
    dpi=600,
    bbox_inches="tight"
)

plt.savefig(
    OUT_DIR / "Figure_S2_Compound_Indices.pdf",
    bbox_inches="tight"
)

plt.show()