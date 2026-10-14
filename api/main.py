"""
main.py â€” API REST de HidroAlert (FastAPI).

Endpoints:
    GET  /health               â€” estado del servicio y de las integraciones.
    GET  /scenarios            â€” presets de escenarios (Ligera/Moderada/Torrencial).
    POST /simulate             â€” ejecuta la simulaciÃ³n y devuelve mÃ©tricas JSON.
    POST /simulate-and-alert   â€” simulaciÃ³n + evaluaciÃ³n de alerta + notificaciÃ³n
                                 (Twilio/Webhook) + informe PDF/JSON en disco.
    GET  /api/geo/simulate     â€” payload 3D para el Centro de Mando (CesiumJS).
    POST /api/alert/trigger    â€” dispara alerta (Twilio/Webhook) con captura 3D.
    POST /api/report/pdf       â€” boletÃ­n PDF con captura 3D incrustada.
    GET  /mando3d              â€” interfaz 3D (Centro de Mando).

Ejecutar en desarrollo:
    uvicorn hidroalert.api.main:app --reload --port 8000
    â†’ abrir http://localhost:8000/mando3d
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from hidroalert import __version__
from hidroalert.config import CONFIG, bankfull_for_zone
from hidroalert.data_ingest.aoi import (
    AreaOfInterest,
    DEFAULT_AOI,
    list_zones,
    resolve_aoi,
)
from hidroalert.data_ingest.ingest_service import build_area_snapshot, build_demo_snapshot
from hidroalert.hydrology.simulation_engine import simulate_flood, scenario_presets
from hidroalert.alerts.engine import evaluate_alert
from hidroalert.alerts.notifier import send_emergency_notifications
from hidroalert.alerts.report import generate_alert_report
from hidroalert.data_ingest.forecast import forecast_precipitation
from hidroalert.hydrology.timeline import build_alert_timeline
from hidroalert.api.geo import simulate_geo, get_snapshot  # noqa: F401 (get_snapshot p/ trigger)
from hidroalert.models import AreaSnapshot

logger = logging.getLogger(__name__)

app = FastAPI(
    title="HidroAlert API",
    description=(
        "Monitoreo, simulaciÃ³n y alerta temprana de inundaciones, desbordes de rÃ­os "
        "y fallas de quebradas (workspace jhons-workspace-amus9)."
    ),
    version=__version__,
)

# CompresiÃ³n HTTP transparente: los payloads GeoJSON del mando 3D bajan
# de ~3.5 MB a ~0.2-0.4 MB por interacciÃ³n (el navegador negocia gzip solo).
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def _no_cache_dynamic(request, call_next):
    """
    Sin cachÃ© para respuestas dinÃ¡micas: sin esto el navegador reutiliza el
    HTML del mando 3D y los payloads de simulaciÃ³n viejos (con esquemas
    anteriores) y "desaparecen" las funciones nuevas al desplegar cambios.
    """
    resp = await call_next(request)
    if request.method == "GET":
        resp.headers["Cache-Control"] = "no-store"
    return resp

# ---------------------------------------------------------------------------
# Interfaz 3D (Centro de Mando)
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).resolve().parents[1] / "static"


@app.get("/mando3d", tags=["3D â€” Centro de Mando"], include_in_schema=False)
def mando3d() -> FileResponse:
    """Sirve la interfaz 3D inmersiva (CesiumJS + glassmorphism)."""
    return FileResponse(_STATIC_DIR / "mando3d.html", media_type="text/html")


@app.get("/api/geo/config", tags=["3D â€” Centro de Mando"])
def api_geo_config() -> dict:
    """
    ConfiguraciÃ³n del visor 3D: token de Cesium Ion (si estÃ¡ configurado en
    .env) para activar Cesium World Terrain + OSM Buildings, y banderas de
    quÃ© capas puede usar el frontend.
    """
    return {
        "cesium_ion_token": CONFIG.cesium.ion_token or None,
        "terrain": "ion-world" if CONFIG.cesium.is_configured else "ellipsoid",
        "osm_buildings": CONFIG.cesium.is_configured,
    }


class TriggerRequest(BaseModel):
    """Cuerpo del disparo de alerta desde la interfaz 3D."""

    precipitation_mm_h: float = Field(default=120, ge=0, le=200)
    flow_increase_pct: float = Field(default=250, ge=0, le=300)
    screenshot_png: str | None = Field(
        default=None, description="Captura base64 (sin prefijo data:) del foco 3D"
    )
    zone: str | None = Field(default=None, description="Id de zona (/zones) o bbox")


@app.post("/api/alert/trigger", tags=["3D â€” Centro de Mando"])
def api_alert_trigger(req: TriggerRequest) -> dict:
    """
    Simula con los parÃ¡metros actuales de la escena 3D, evalÃºa el nivel y
    envÃ­a las notificaciones (Twilio + Webhook COER). Si se incluye la
    captura PNG del globo, se guarda junto al informe.
    """
    from hidroalert.data_ingest.aoi import parse_bbox_text

    try:
        snapshot, demo = get_snapshot(req.zone)
        result = simulate_flood(
            snapshot,
            precipitation_mm_h=req.precipitation_mm_h,
            flow_increase_pct=req.flow_increase_pct,
            bankfull_depth_m=bankfull_for_zone(req.zone, snapshot.aoi_name),
        )
        decision = evaluate_alert(result)

        # Captura 3D adjunta (si llegÃ³ del frontend)
        shot_path = None
        if req.screenshot_png:
            import base64

            from hidroalert.config import REPORTS_DIR

            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = result.timestamp[:19].replace(":", "").replace("-", "")
            shot_path = REPORTS_DIR / f"captura3d_{decision.level.lower()}_{stamp}.png"
            shot_path.write_bytes(base64.b64decode(req.screenshot_png))

        out = send_emergency_notifications(result, decision, force_all_levels=True)
        return {
            "level": decision.level,
            "reasons": decision.reasons,
            "sent": not out.get("skipped", False),
            "twilio_count": len(out.get("twilio", [])),
            "webhook_ok": out.get("webhook_ok", False),
            "screenshot": str(shot_path) if shot_path else None,
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/report/pdf", tags=["3D â€” Centro de Mando"])
def api_report_pdf(req: TriggerRequest) -> dict:
    """BoletÃ­n PDF con captura 3D del foco (si se enviÃ³)."""
    try:
        snapshot, _ = get_snapshot(req.zone)
        result = simulate_flood(
            snapshot,
            precipitation_mm_h=req.precipitation_mm_h,
            flow_increase_pct=req.flow_increase_pct,
            bankfull_depth_m=bankfull_for_zone(req.zone, snapshot.aoi_name),
        )
        decision = evaluate_alert(result)
        paths = generate_alert_report(result, decision)
        if req.screenshot_png and paths.get("pdf"):
            # Incrusta la captura 3D como pÃ¡gina final del PDF
            import base64

            from hidroalert.config import REPORTS_DIR

            shot = REPORTS_DIR / f"foco3d_{decision.level.lower()}.png"
            shot.write_bytes(base64.b64decode(req.screenshot_png))
            from hidroalert.alerts.report import _append_image_page

            _append_image_page(Path(paths["pdf"]), shot, "Captura 3D del foco (Cesium)")
        return {
            "level": decision.level,
            "pdf": str(paths["pdf"]) if paths.get("pdf") else None,
            "json": str(paths["json"]) if paths.get("json") else None,
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Cache de snapshots por AOI: evita re-descargar GEE en cada simulaciÃ³n.
# ---------------------------------------------------------------------------
_SNAPSHOT_CACHE: dict[str, AreaSnapshot] = {}


class SimulateRequest(BaseModel):
    """Cuerpo del request para /simulate, /simulate-and-alert y /forecast-timeline."""

    aoi_name: str = Field(
        default="",
        description="Etiqueta personalizada del AOI (opcional). VacÃ­o usa el "
                    "nombre de la zona del registro o 'AOI personalizado'.",
    )
    zone: str | None = Field(
        default=None,
        description="Id de zona del registro (/zones). Tiene prioridad sobre bbox.",
    )
    bbox: str | None = Field(
        default=None,
        description="min_lon,min_lat,max_lon,max_lat (WGS84). None = AOI por defecto.",
    )
    precipitation_mm_h: float = Field(default=20.0, ge=0, le=200)
    flow_increase_pct: float = Field(default=50.0, ge=0, le=300)
    scenario_label: str = Field(default="")
    include_overland: bool | None = Field(
        default=None,
        description=(
            "Simular tambiÃ©n la concentraciÃ³n de agua sobre terreno "
            "(escorrentÃ­a superficial ruteada con D8: pozas en depresiones). "
            "None = default del endpoint: False en /simulate y "
            "/simulate-and-alert; True en /forecast-timeline."
        ),
    )
    use_demo_data: bool = Field(
        default=False,
        description="Usa un Ã¡rea sintÃ©tica (sin GEE) para pruebas offline.",
    )
    lookback_days: int = Field(default=12, ge=1, le=60)


def _resolve_aoi(req: SimulateRequest) -> AreaOfInterest:
    """Resuelve el AOI del request: id de zona > bbox personalizado > default."""
    if req.zone:
        aoi = resolve_aoi(req.zone)
        if req.aoi_name:
            aoi = AreaOfInterest(
                name=req.aoi_name,
                min_lon=aoi.min_lon, min_lat=aoi.min_lat,
                max_lon=aoi.max_lon, max_lat=aoi.max_lat,
            )
        return aoi
    if req.bbox:
        aoi = parse_bbox_text(req.bbox)
        aoi = AreaOfInterest(
            name=req.aoi_name or aoi.name,
            min_lon=aoi.min_lon,
            min_lat=aoi.min_lat,
            max_lon=aoi.max_lon,
            max_lat=aoi.max_lat,
        )
        return aoi
    return DEFAULT_AOI


def _get_snapshot(req: SimulateRequest) -> AreaSnapshot:
    """Obtiene (y cachea) el snapshot del Ã¡rea: real (GEE) o demo."""
    aoi = _resolve_aoi(req)
    cache_key = "demo" if req.use_demo_data else f"{aoi.bbox}@{aoi.scale_m}"
    if cache_key in _SNAPSHOT_CACHE:
        return _SNAPSHOT_CACHE[cache_key]

    if req.use_demo_data or not CONFIG.gee.is_configured:
        if not req.use_demo_data:
            logger.warning("GEE sin configurar; usando datos demo.")
        snapshot = build_demo_snapshot()
    else:
        snapshot = build_area_snapshot(aoi, lookback_days=req.lookback_days)
    _SNAPSHOT_CACHE[cache_key] = snapshot
    return snapshot


@app.get("/health", tags=["sistema"])
def health() -> dict:
    """Estado del servicio y de las integraciones configuradas."""
    return {
        "status": "ok",
        "version": __version__,
        "gee_configured": CONFIG.gee.is_configured,
        "gmaps_configured": CONFIG.gmaps.is_configured,
        "twilio_configured": CONFIG.twilio.is_configured,
        "webhook_configured": CONFIG.webhook.is_configured,
    }


@app.get("/zones", tags=["sistema"])
def zones() -> dict:
    """
    Catálogo de zonas de análisis: distrito de La Yarada-Los Palos (Tacna)
    y sus subsectores, más coordenadas personalizadas vía bbox. Para ambas UIs.
    """
    return {"zones": list_zones(), "default": "tacna-yarada-los-palos"}


@app.get("/scenarios", tags=["simulaciÃ³n"])
def scenarios() -> dict:
    """Presets de escenarios para el dashboard."""
    return {"presets": scenario_presets()}


@app.post("/simulate", tags=["simulaciÃ³n"])
def simulate(req: SimulateRequest) -> dict:
    """Ejecuta la simulaciÃ³n hidrolÃ³gica y devuelve las mÃ©tricas JSON."""
    try:
        snapshot = _get_snapshot(req)
        result = simulate_flood(
            snapshot,
            precipitation_mm_h=req.precipitation_mm_h,
            flow_increase_pct=req.flow_increase_pct,
            scenario_label=req.scenario_label,
            bankfull_depth_m=bankfull_for_zone(req.zone, snapshot.aoi_name),
            include_overland=bool(req.include_overland),
        )
        return result.to_dict()
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/geo/simulate", tags=["3D â€” Centro de Mando"])
def api_geo_simulate(
    zone: str | None = Query(default=None, description="Id de zona (/zones) o bbox personalizado"),
    precipitation_mm_h: float = Query(default=35.0, ge=0, le=200),
    flow_increase_pct: float = Query(default=80.0, ge=0, le=300),
    overland: bool = Query(
        default=False,
        description="Incluir concentraciÃ³n de agua sobre terreno (escorrentÃ­a D8)",
    ),
    sar: bool = Query(
        default=False,
        description="Incluir capa de agua observada Sentinel-1 (bajo demanda)",
    ),
) -> dict:
    """
    Payload 3D para CesiumJS: agua volumÃ©trica, viviendas en puntos, vÃ­as,
    radar SAR, focos crÃ­ticos y KPIs â€” una llamada por cambio de slider o zona.
    Con `overland=true` la lÃ¡mina incluye las pozas de escorrentÃ­a sobre el
    terreno, no solo el desborde del cauce; con `sar=true` viaja tambiÃ©n el
    agua observada por Sentinel-1 (capa bajo demanda: solo se pide cuando el
    usuario activa "Radar SAR").
    """
    try:
        return simulate_geo(
            zone, precipitation_mm_h, flow_increase_pct,
            include_overland=overland, include_sar=sar,
        )
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/forecast", tags=["pronÃ³stico"])
def get_forecast(
    bbox: str | None = Query(default=None, description="min_lon,min_lat,max_lon,max_lat"),
    hours: int = Query(default=48, ge=1, le=96),
) -> dict:
    """PronÃ³stico horario de precipitaciÃ³n (WeatherNext 3 si hay key; si no, Open-Meteo)."""
    try:
        aoi = parse_bbox_text(bbox) if bbox else DEFAULT_AOI
        fc = forecast_precipitation(aoi, hours=hours)
        return {
            "source": fc["source"],
            "peak": {"mm_h": fc["peak_mm_h"], "time": fc["peak_time"]},
            "series": [{"time": t, "mm_h": v} for t, v in fc["series"]],
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/forecast-timeline", tags=["pronÃ³stico"])
def forecast_timeline(req: SimulateRequest, max_hours: int = Query(default=48, ge=1, le=96)) -> dict:
    """
    LÃ­nea de tiempo de alertas: propaga el pronÃ³stico horario por el motor
    hidrolÃ³gico â€” con la CONCENTRACIÃ“N DE AGUA SOBRE TERRENO incorporada
    por defecto (cada hora reporta `overland_volume_m3`) â€” y devuelve
    CUÃNDO se cruza AMARILLO/ROJO.
    """
    try:
        snapshot = _get_snapshot(req)
        aoi = _resolve_aoi(req)
        fc = forecast_precipitation(aoi, hours=max_hours)
        # Por defecto la predicciÃ³n de lluvias incluye la escorrentÃ­a sobre
        # terreno (None â†’ True); el request puede desactivarla explÃ­citamente.
        include_overland = True if req.include_overland is None else req.include_overland
        tl = build_alert_timeline(
            snapshot, fc, max_hours=max_hours,
            bankfull_depth_m=bankfull_for_zone(req.zone, snapshot.aoi_name),
            include_overland=include_overland,
        )
        return {
            "aoi": snapshot.aoi_name,
            "source": fc["source"],
            "peak": tl["peak"],
            "rise_to": tl["rise_to"],
            "timeline": tl["timeline"],
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/simulate-and-alert", tags=["alerta"])
def simulate_and_alert(
    req: SimulateRequest,
    notify: bool = Query(default=True, description="Enviar Twilio/Webhook"),
    generate_report: bool = Query(default=True, description="Generar PDF/JSON"),
) -> dict:
    """
    Pipeline completo: simulaciÃ³n â†’ alerta condicional â†’ notificaciones â†’ informe.
    """
    try:
        snapshot = _get_snapshot(req)
        result = simulate_flood(
            snapshot,
            precipitation_mm_h=req.precipitation_mm_h,
            flow_increase_pct=req.flow_increase_pct,
            scenario_label=req.scenario_label,
            bankfull_depth_m=bankfull_for_zone(req.zone, snapshot.aoi_name),
            include_overland=bool(req.include_overland),
        )
        decision = evaluate_alert(result)
        notified = (
            send_emergency_notifications(result, decision) if notify else {"skipped": True}
        )
        report_paths = generate_alert_report(result, decision) if generate_report else {}
        return {
            **result.to_dict(),
            "alerta": decision.to_dict(),
            "notificacion": {
                "sent": not notified.get("skipped", False),
                "webhook_ok": notified.get("webhook_ok", False),
                "twilio_count": len(notified.get("twilio", [])),
            },
            "informe": {k: str(v) if v else None for k, v in report_paths.items()},
        }
    except (RuntimeError, ValueError, ImportError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
