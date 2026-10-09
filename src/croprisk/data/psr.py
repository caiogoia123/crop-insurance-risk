"""Ingestion of PSR/SISSER policies (MAPA open data).

Privacy by design (LGPD): each CSV is streamed from the server and parsed in chunks.
Personal columns (name, CPF/CNPJ), insurer-side identifiers and exact property
coordinates are dropped from every chunk *before* anything is written to disk.
The raw CSV with personal data never touches the disk.
"""

from __future__ import annotations

import io
import json
import logging
import unicodedata
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from croprisk import config

log = logging.getLogger(__name__)

# Columns removed at ingestion. Personal data first, then data minimization.
DROP_COLUMNS = {
    "NM_SEGURADO",  # policyholder name (personal data)
    "NR_DOCUMENTO_SEGURADO",  # CPF/CNPJ (personal data)
    "NR_PROPOSTA",  # insurer-side identifiers: linkable back to the person
    "NR_APOLICE",
    "LATITUDE",  # exact property location: not needed (climate is per municipality)
    "NR_GRAU_LAT",
    "NR_MIN_LAT",
    "NR_SEG_LAT",
    "LONGITUDE",
    "NR_GRAU_LONG",
    "NR_MIN_LONG",
    "NR_SEG_LONG",
    "NR_DECIMAL_LATITUDE",
    "NR_DECIMAL_LONGITUDE",
}

RENAME = {
    "NM_RAZAO_SOCIAL": "insurer",
    "CD_PROCESSO_SUSEP": "susep_product",
    "ID_PROPOSTA": "proposal_id",
    "DT_PROPOSTA": "dt_proposal",
    "DT_INICIO_VIGENCIA": "dt_start",
    "DT_FIM_VIGENCIA": "dt_end",
    "NM_MUNICIPIO_PROPRIEDADE": "municipality",
    "SG_UF_PROPRIEDADE": "uf",
    "NM_CLASSIF_PRODUTO": "product_class",
    "NM_CULTURA_GLOBAL": "crop",
    "NR_AREA_TOTAL": "area_ha",
    "NR_ANIMAL": "n_animals",
    "NR_PRODUTIVIDADE_ESTIMADA": "yield_expected",
    "NR_PRODUTIVIDADE_SEGURADA": "yield_insured",
    "NIVELDECOBERTURA": "coverage_level",
    "VL_LIMITE_GARANTIA": "sum_insured",
    "VL_PREMIO_LIQUIDO": "premium",
    "PE_TAXA": "rate",
    "VL_SUBVENCAO_FEDERAL": "subsidy",
    "DT_APOLICE": "dt_policy",
    "ANO_APOLICE": "policy_year",
    "CD_GEOCMU": "ibge_code",
    "VALOR_INDENIZACAO": "indemnity",
    "EVENTO_PREPONDERANTE": "event",
}

DATE_COLS = ["dt_proposal", "dt_start", "dt_end", "dt_policy"]
FLOAT_COLS = [
    "area_ha",
    "n_animals",
    "yield_expected",
    "yield_insured",
    "coverage_level",
    "sum_insured",
    "premium",
    "rate",
    "subsidy",
    "indemnity",
]
INT_COLS = ["policy_year", "ibge_code", "proposal_id"]
STR_COLS = [
    "insurer",
    "susep_product",
    "municipality",
    "uf",
    "product_class",
    "crop",
    "event",
]

SCHEMA = pa.schema(
    [(c, pa.string()) for c in STR_COLS]
    + [(c, pa.timestamp("ns")) for c in DATE_COLS]
    + [(c, pa.float64()) for c in FLOAT_COLS]
    + [(c, pa.int64()) for c in INT_COLS]
    + [("source_file", pa.string())]
)


def _norm(name: str) -> str:
    """Upper-case column name without accents ('VALOR_INDENIZAÇÃO' -> 'VALOR_INDENIZACAO')."""
    s = unicodedata.normalize("NFKD", name.strip()).encode("ascii", "ignore").decode()
    return s.upper()


def list_resources() -> list[dict]:
    """CSV resources of the sisser3 package, with server metadata."""
    r = requests.get(config.CKAN_PACKAGE_URL, headers={"User-Agent": config.USER_AGENT}, timeout=60)
    r.raise_for_status()
    res = r.json()["result"]["resources"]
    return [
        {
            "name": x["name"],
            "url": x["url"],
            "last_modified": x.get("last_modified"),
            "id": x["id"],
        }
        for x in res
        if (x.get("format") or "").upper() == "CSV"
    ]


class IterStream(io.RawIOBase):
    """Read-only binary stream over an iterator of byte chunks (returns b"" at EOF)."""

    def __init__(self, chunks):
        self._it = iter(chunks)
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:
        while not self._buf:
            try:
                self._buf = next(self._it)
            except StopIteration:
                return 0
        n = min(len(b), len(self._buf))
        b[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


def parse_chunk(chunk: pd.DataFrame, source: str) -> pd.DataFrame:
    """Drop personal/unneeded columns, rename and type a raw chunk (all-string input)."""
    chunk = chunk.rename(columns=_norm)
    chunk = chunk.drop(columns=[c for c in chunk.columns if c in DROP_COLUMNS])
    unknown = set(chunk.columns) - set(RENAME)
    if unknown:
        # Fail closed: a new column could be personal data.
        raise ValueError(f"Unexpected columns in {source}: {sorted(unknown)}")
    chunk = chunk.rename(columns=RENAME)
    out = pd.DataFrame(index=chunk.index)
    for c in STR_COLS:
        s = chunk[c].astype("string").str.strip()
        out[c] = s.where(~s.isin(["-", ""]), None)
    for c in DATE_COLS:
        d = pd.to_datetime(chunk[c], format="%d/%m/%Y", errors="coerce")
        # Typos such as year 5207 exist; anything outside 1990-2035 becomes missing.
        d = d.where((d.dt.year >= 1990) & (d.dt.year <= 2035))
        out[c] = d.astype("datetime64[ns]")
    for c in FLOAT_COLS:
        s = chunk[c].astype("string").str.strip().str.replace(",", ".", regex=False)
        out[c] = pd.to_numeric(s.where(~s.isin(["-", ""]), None), errors="coerce").astype("float64")
    for c in INT_COLS:
        out[c] = pd.to_numeric(chunk[c], errors="coerce").astype("Int64")
    out["source_file"] = source
    return out


def ingest_stream(lines: io.TextIOBase, source: str, dest: Path, chunksize: int = 100_000) -> int:
    """Parse a CSV text stream chunk by chunk into a parquet file. Returns row count."""
    tmp = dest.with_suffix(".parquet.tmp")
    n = 0
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        reader = pd.read_csv(
            lines,
            sep=";",
            dtype=str,
            keep_default_na=False,
            chunksize=chunksize,
            on_bad_lines="error",
        )
        for raw in reader:
            df = parse_chunk(raw, source)
            del raw
            writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
            n += len(df)
    tmp.replace(dest)
    return n


def download_all(force: bool = False) -> list[Path]:
    """Stream every CSV resource into data/interim/psr_<id>.parquet (skips unchanged files)."""
    config.ensure_dirs()
    manifest_path = config.RAW / "psr_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    paths = []
    for res in list_resources():
        dest = config.INTERIM / f"psr_{res['id'][:8]}.parquet"
        prev = manifest.get(res["id"], {})
        if dest.exists() and not force and prev.get("last_modified") == res["last_modified"]:
            log.info("unchanged, skipping: %s", res["name"])
            paths.append(dest)
            continue
        log.info("streaming %s", res["name"])
        with requests.get(
            res["url"], headers={"User-Agent": config.USER_AGENT}, stream=True, timeout=120
        ) as r:
            r.raise_for_status()
            raw = io.BufferedReader(IterStream(r.iter_content(chunk_size=1 << 20)))
            text = io.TextIOWrapper(raw, encoding="latin1", newline="")
            n = ingest_stream(text, res["name"], dest)
        manifest[res["id"]] = {**res, "rows": n}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log.info("%s: %d rows -> %s", res["name"], n, dest.name)
        paths.append(dest)
    return paths


def load_policies() -> pd.DataFrame:
    """All ingested policies in one frame (deduplicated by proposal id)."""
    files = sorted(config.INTERIM.glob("psr_*.parquet"))
    if not files:
        raise FileNotFoundError("No PSR parquet files. Run `make data` first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_all()
