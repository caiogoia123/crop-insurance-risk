"""Generate the SYNTHETIC SISSER-like CSV used by the tests (fixed seed)."""

from pathlib import Path

import numpy as np

HEADER = (
    "NM_RAZAO_SOCIAL;CD_PROCESSO_SUSEP;NR_PROPOSTA;ID_PROPOSTA;DT_PROPOSTA;DT_INICIO_VIGENCIA;"
    "DT_FIM_VIGENCIA;NM_SEGURADO;NR_DOCUMENTO_SEGURADO;NM_MUNICIPIO_PROPRIEDADE;SG_UF_PROPRIEDADE;"
    "LATITUDE;NR_GRAU_LAT;NR_MIN_LAT;NR_SEG_LAT;LONGITUDE;NR_GRAU_LONG;NR_MIN_LONG;NR_SEG_LONG;"
    "NR_DECIMAL_LATITUDE;NR_DECIMAL_LONGITUDE;NM_CLASSIF_PRODUTO;NM_CULTURA_GLOBAL;NR_AREA_TOTAL;"
    "NR_ANIMAL;NR_PRODUTIVIDADE_ESTIMADA;NR_PRODUTIVIDADE_SEGURADA;NivelDeCobertura;"
    "VL_LIMITE_GARANTIA;VL_PREMIO_LIQUIDO;PE_TAXA;VL_SUBVENCAO_FEDERAL;NR_APOLICE;DT_APOLICE;"
    "ANO_APOLICE;CD_GEOCMU;VALOR_INDENIZAÇÃO;EVENTO_PREPONDERANTE"
)
MUNIS = [("Cascavel", "PR", 4104808), ("Sorriso", "MT", 5107925), ("Jataí", "GO", 5211909)]
CROPS = ["Soja", "Milho 2ª safra", "Trigo"]


def fmt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def main() -> None:
    rng = np.random.default_rng(7)
    lines = [HEADER]
    for i in range(40):
        name, uf, code = MUNIS[i % 3]
        crop = CROPS[i % 3]
        area = rng.uniform(10, 500)
        si = area * rng.uniform(2000, 4000)
        rate = rng.uniform(0.03, 0.15)
        claim = rng.random() < 0.2
        ind = fmt(si * rng.uniform(0.1, 0.6)) if claim else "-"
        event = "SECA" if claim else "-"
        year = 2016 + i % 8
        lines.append(
            ";".join(
                [
                    "SEGURADORA SINTETICA S.A.",
                    "15414000000000000",
                    str(9000 + i),
                    str(100000 + i),
                    f"15/08/{year}",
                    f"20/08/{year}",
                    f"20/04/{year + 1}",
                    f"SINTETICO {i:03d}",
                    f"000.000.000-{i:02d}",
                    name,
                    uf,
                    "S",
                    "25",
                    "0",
                    "0",
                    "O",
                    "53",
                    "0",
                    "0",
                    "-25,0",
                    "-53,0",
                    "PRODUTIVIDADE",
                    crop,
                    fmt(area),
                    "-",
                    "3500",
                    "2450",
                    "0,7",
                    fmt(si),
                    fmt(si * rate),
                    f"{rate:.6f}".replace(".", ","),
                    fmt(si * rate * 0.4),
                    str(70000 + i),
                    f"25/08/{year}",
                    str(year),
                    str(code),
                    ind,
                    event,
                ]
            )
        )
    out = Path(__file__).with_name("synthetic_sisser.csv")
    out.write_bytes(("\r\n".join(lines) + "\r\n").encode("latin1"))


if __name__ == "__main__":
    main()
