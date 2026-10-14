"""
check_gee.py — Verificación de un comando de la ingesta real de Sentinel-1.

Uso (tras registrar el proyecto en https://code.earthengine.google.com/register):

    cd hidroalert
    .venv/Scripts/python.exe scripts/check_gee.py          # Distrito de La Yarada-Los Palos (Tacna)
    .venv/Scripts/python.exe scripts/check_gee.py --zone tacna-yarada
    .venv/Scripts/python.exe scripts/check_gee.py --bbox -70.5853,-18.3501,-70.3161,-18.0776

Pasos que ejecuta:
    1. Inicializa Earth Engine con HIDROALERT_GEE_PROJECT_ID.
    2. Busca imágenes Sentinel-1 GRD recientes para el AOI.
    3. Corre el pipeline Otsu + HAND y muestra área de agua observada.
    4. Descarga edificios (Open Buildings) del AOI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Raíz del repo (D:\<algo>\) para importar el paquete hidroalert
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hidroalert.data_ingest.aoi import DEFAULT_AOI, parse_bbox_text
from hidroalert.data_ingest.gee_client import get_ee
from hidroalert.data_ingest.sar_flood import detect_observed_flood
from hidroalert.data_ingest.buildings_roads import fetch_buildings


def main() -> int:
    ap = argparse.ArgumentParser(description="Validación de ingesta GEE de HidroAlert")
    ap.add_argument(
        "--zone",
        default=None,
        help="Id de zona del registro (ej. tacna-yarada, tacna-los-palos)",
    )
    ap.add_argument(
        "--bbox",
        default=None,
        help="min_lon,min_lat,max_lon,max_lat (por defecto: Dto. de La Yarada-Los Palos, Tacna)",
    )
    ap.add_argument("--days", type=int, default=12, help="ventana de búsqueda en días")
    args = ap.parse_args()

    # 1) Inicialización
    from hidroalert.config import CONFIG
    proj = CONFIG.gee.project_id or "(no configurado)"
    try:
        ee = get_ee()
        print(f"[1/4] Earth Engine inicializado (proyecto {proj}).")
    except RuntimeError as exc:
        print(f"[1/4] FALLO de inicialización con proyecto {proj}:")
        print("   ", str(exc)[:300])
        print("\n    Causa habitual: el proyecto existe en GCP pero aún no está REGISTRADO")
        print("    para Earth Engine, o el .env contiene un número en vez del ID string")
        print("    (ej. 'ee-mi-proyecto').")
        print("    Acción: https://code.earthengine.google.com/register")
        return 1

    from hidroalert.data_ingest.aoi import get_zone

    if args.zone:
        aoi = get_zone(args.zone)
        if aoi is None:
            print(f"[2/4] Zona desconocida: {args.zone}. Use /zones o --bbox.")
            return 2
    elif args.bbox:
        aoi = parse_bbox_text(args.bbox)
    else:
        aoi = DEFAULT_AOI
    print(f"[2/4] AOI: {aoi.name} {aoi.bbox} (escala {aoi.scale_m:.0f} m)")

    # 2) Ingesta SAR
    try:
        sar = detect_observed_flood(aoi, lookback_days=args.days)
    except RuntimeError as exc:
        print(f"[2/4] FALLO de ingesta SAR: {str(exc)[:250]}")
        return 2

    th = sar["threshold_db"]
    ha = sar["water_area_m2"] / 1e4
    img = sar["image_id"].split("/")[-1]
    print(f"[2/4] Sentinel-1: {img} ({sar['date'][:10]}) | umbral Otsu {th:.2f} dB")
    print(f"[3/4] Agua observada: {ha:.1f} ha | píxeles: {sar['stats']['water_pixels']} de {sar['stats']['valid_pixels']}")

    # 4) Edificios
    try:
        blds = fetch_buildings(aoi)
        n = len(blds.get("features", []))
        print(f"[4/4] Open Buildings: {n} huellas descargadas")
    except Exception as exc:
        print(f"[4/4] Open Buildings no disponible: {str(exc)[:150]}")

    print("\n✅ Ingesta GEE operativa. El dashboard/API usarán datos reales automáticamente.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
