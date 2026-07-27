import xarray as xr

ds_sdhi = xr.open_dataset("../dataset/ca_sdhi_warmest_1951_2025.nc")
ds_scei = xr.open_dataset("../dataset/ca_scei_warmest_1951_2025.nc")

ds_sdhi_renamed = ds_sdhi.rename({
    "prcp": "prcp_sdhi",
    "tmean": "tmean_sdhi",
    "tmax": "tmax_sdhi",
    "tmin": "tmin_sdhi",
    "pet": "pet_sdhi",
    "spi": "spi_sdhi",
    "sti": "sti_sdhi",
    "spei": "spei_sdhi",
    "x1": "x1_sdhi",
    "sdhi1": "sdhi1",
    "x2": "x2_sdhi",
    "sdhi2": "sdhi2"
})

ds_scei_renamed = ds_scei.rename({
    "SPI": "spi_scei",
    "STI": "sti_scei",
    "SCEI": "scei"
})

ds_merged = xr.merge([ds_sdhi_renamed, ds_scei_renamed])

ds_merged.attrs["title"] = "California Compound Climate Extremes — SDHI + SCEI (1951–2025)"
ds_merged.attrs["author"] = "Farzad Asgari"
ds_merged.attrs["source"] = "NOAA nClimGrid Gridded Dataset"

out = "../dataset/california_warmest.nc"
ds_merged.to_netcdf(out)

ds_sdhi = xr.open_dataset("../dataset/ca_sdhi_monthly_1951_2025.nc")
ds_scei = xr.open_dataset("../dataset/ca_scei_monthly_1951_2025.nc")
ds_scdhi = xr.open_dataset("../dataset/ca_scdhi_monthly_1951_2025.nc")

ds_sdhi_renamed = ds_sdhi.rename({
    "pet": "pet_sdhi",
    "spi": "spi_sdhi",
    "sti": "sti_sdhi",
    "spei": "spei_sdhi",
})

ds_scei_renamed = ds_scei.rename({
    "spi": "spi_scei",
    "sti": "sti_scei",
    "scei": "scei"
})

ds_scdhi_renamed = ds_scdhi.rename({
    "sapei": "sapei_scdhi",
    "sti": "sti_scdhi",
    "scdhi": "scdhi"
})

ds_sdhi_selected = ds_sdhi_renamed[[
    "pet_sdhi", "sdhi1", "sdhi2", "spei_sdhi", "spi_sdhi", "sti_sdhi",
    "prcp", "tmax", "tmean", "tmin", "month", "year", "lat", "lon"
]]

ds_scei_selected = ds_scei_renamed[[
    "spi_scei", "sti_scei", "scei"
]]

ds_scdhi_selected = ds_scdhi_renamed[[
    "sapei_scdhi", "sti_scdhi", "scdhi"
]]

ds_merged = xr.merge([ds_sdhi_selected, ds_scei_selected, ds_scdhi_selected])

ds_merged.attrs["title"] = "California Compound Climate Extremes — SDHI + SCEI + SCDHI (1951–2025)"
ds_merged.attrs["author"] = "Farzad Asgari"
ds_merged.attrs["source"] = "NOAA nClimGrid Gridded Dataset"

out = "../dataset/california_monthly.nc"
ds_merged.to_netcdf(out)
