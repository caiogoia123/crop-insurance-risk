"""FastAPI service for the pre-season (contract-time) claim model.

Run locally:  uvicorn croprisk.serving.app:app --port 8001
In production it sits behind Nginx at /crop-risk/ (ROOT_PATH=/crop-risk).
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from croprisk import calendar, config
from croprisk.serving.bundle import Bundle, current_dir

log = logging.getLogger("croprisk.api")
STATIC = Path(__file__).parent / "static"
MODELS_DIR = Path(os.environ.get("CROPRISK_MODELS", config.MODELS))


@lru_cache(maxsize=1)
def get_bundle() -> Bundle:
    path = current_dir(MODELS_DIR)
    log.info("loading model bundle %s", path)
    return Bundle.load(path)


app = FastAPI(
    title="Crop insurance claim risk",
    description=(
        "Probability that a Brazilian subsidized crop insurance policy (PSR) is indemnified, "
        "estimated at contract time from contract fields, municipality loss history, "
        "1981-2005 climatology (NASA POWER) and ENSO. Research project, not an underwriting tool. "
        "Source: github.com/caiogoia123/crop-insurance-risk"
    ),
    version="1.0",
    root_path=os.environ.get("ROOT_PATH", ""),
)


class PredictRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "crop": "Soja",
                "ibge_code": 4104808,
                "contract_date": "2025-09-15",
                "coverage_level": 0.7,
                "area_ha": 120,
                "sum_insured": 350000,
                "yield_expected": 3600,
                "yield_insured": 2520,
                "insurer": "BRASILSEG COMPANHIA DE SEGUROS",
                "product_class": "PRODUTIVIDADE",
                "rate": 0.065,
            }
        }
    )

    crop: str = Field(description="Crop name as in the PSR data, e.g. 'Soja', 'Milho 2ª safra'")
    ibge_code: int = Field(ge=1_100_000, le=5_399_999, description="IBGE 7-digit municipality code")
    contract_date: date = Field(description="Coverage start / contract date")
    coverage_level: float | None = Field(
        None, ge=0.05, le=1.0, description="Share of expected yield covered"
    )
    area_ha: float = Field(gt=0, le=100_000)
    sum_insured: float = Field(gt=0, le=1e9, description="Sum insured (R$)")
    yield_expected: float | None = Field(
        None, gt=0, le=200_000, description="Expected yield (kg/ha)"
    )
    yield_insured: float | None = Field(None, gt=0, le=200_000, description="Insured yield (kg/ha)")
    insurer: str | None = Field(
        None, description="Insurer name; unknown names are treated as 'other'"
    )
    product_class: Literal["PRODUTIVIDADE", "CUSTEIO"] | None = None
    rate: float | None = Field(
        None,
        gt=0,
        lt=1,
        description="Optional premium rate charged; returns the rate-implied probability",
    )

    @field_validator("crop")
    @classmethod
    def known_crop(cls, v: str) -> str:
        if v not in calendar.CROP_GROUP:
            raise ValueError(f"unknown crop '{v}'; see /meta/options")
        return v

    @field_validator("contract_date")
    @classmethod
    def plausible_date(cls, v: date) -> date:
        if not date(2006, 1, 1) <= v <= date.today() + timedelta(days=400):
            raise ValueError("contract_date must be between 2006-01-01 and about a year from today")
        return v


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/health")
def health() -> dict:
    b = get_bundle()
    return {"status": "ok", "model_version": b.meta["version"]}


@app.get("/model-info")
def model_info() -> dict:
    m = get_bundle().meta
    return {
        "version": m["version"],
        "created_at": m["created_at"],
        "model": m["model"],
        "train_safras": m["train_safras"],
        "train_rows": m["train_rows"],
        "holdout": m["holdout"],
        "features": m["features"],
        "data_snapshot": m["data_snapshot"],
        "limitations": [
            "Only federally subsidized policies (PSR) are in the data: selection bias.",
            "Claims registered after the public snapshot are missing (claims cutoff in data_snapshot).",
            "Pre-season risk cannot see the season's weather: the ranking can fail in drought years.",
            "Climate is per 0.5 x 0.625 degree grid cell, not per farm.",
        ],
    }


@app.get("/meta/options")
def options() -> dict:
    m = get_bundle().meta
    return {
        "crops": m["crops"],
        "insurers": [i for i in m["categories"]["insurer"] if i != "other"],
        "product_classes": ["PRODUTIVIDADE", "CUSTEIO"],
        "ufs": sorted(get_bundle().munis["uf"].dropna().unique().tolist()),
    }


@app.get("/meta/municipalities")
def municipalities(uf: str = Query(min_length=2, max_length=2)) -> list[dict]:
    mun = get_bundle().munis
    sel = mun[mun["uf"] == uf.upper()].sort_values("muni_name")
    if sel.empty:
        raise HTTPException(404, f"unknown UF '{uf}'")
    return [
        {"ibge_code": int(i), "name": n} for i, n in zip(sel.index, sel["muni_name"], strict=True)
    ]


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    b = get_bundle()
    if req.ibge_code not in b.munis.index:
        raise HTTPException(422, f"unknown IBGE municipality code {req.ibge_code}")
    return b.predict(req.model_dump())
