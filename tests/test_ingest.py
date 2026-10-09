"""Ingestion: personal data never reaches the parquet output (SYNTHETIC fixture)."""

import io
from pathlib import Path

import pandas as pd
import pytest

from croprisk.data import psr

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_sisser.csv"


def _ingest(tmp_path, text: str) -> pd.DataFrame:
    dest = tmp_path / "out.parquet"
    psr.ingest_stream(io.StringIO(text), "synthetic", dest, chunksize=7)
    return pd.read_parquet(dest)


def test_personal_and_location_columns_are_dropped(tmp_path):
    df = _ingest(tmp_path, FIXTURE.read_bytes().decode("latin1"))
    assert len(df) == 40
    banned = {"NM_SEGURADO", "NR_DOCUMENTO_SEGURADO", "NR_DECIMAL_LATITUDE", "NR_APOLICE"}
    assert not banned & set(df.columns)
    assert not {"nm_segurado", "nr_documento_segurado"} & {c.lower() for c in df.columns}
    text = df.astype(str).to_csv()
    assert "SINTETICO" not in text  # fake names
    assert "000.000.000" not in text  # fake CPFs


def test_types_and_missing_codes(tmp_path):
    df = _ingest(tmp_path, FIXTURE.read_bytes().decode("latin1"))
    assert df["rate"].between(0.03, 0.15).all()
    assert df["coverage_level"].eq(0.7).all()
    assert df["dt_start"].dt.year.between(2016, 2023).all()
    claims = df["event"].notna()
    assert (df.loc[claims, "indemnity"] > 0).all()
    assert df.loc[~claims, "indemnity"].isna().all()  # "-" -> missing
    assert df["ibge_code"].isin([4104808, 5107925, 5211909]).all()


def test_unknown_column_fails_closed(tmp_path):
    raw = FIXTURE.read_bytes().decode("latin1").splitlines()
    raw[0] += ";NM_CONJUGE"
    raw[1:] = [r + ";FULANO" for r in raw[1:]]
    with pytest.raises(ValueError, match="Unexpected columns"):
        _ingest(tmp_path, "\n".join(raw))


def test_out_of_range_dates_become_missing():
    chunk = pd.read_csv(FIXTURE, sep=";", dtype=str, encoding="latin1", keep_default_na=False).head(
        2
    )
    chunk.loc[0, "DT_PROPOSTA"] = "09/05/5207"
    out = psr.parse_chunk(chunk, "synthetic")
    assert pd.isna(out.loc[0, "dt_proposal"])
    assert not pd.isna(out.loc[1, "dt_proposal"])
