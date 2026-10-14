"""
geo.py — Capa de datos geográficos para el Centro de Mando 3D (CesiumJS).

Convierte el motor hidrológico (Módulo 2) y la ingesta GEE (Módulo 1) en
un payload único listo para renderizar en 3D:

    * water:     celdas inundadas como polígonos con clase de profundidad
                 (el frontend las extruye en Z = lámina × exageración).
    * buildings: huellas con altura estimada + flag "affected" + lámina (cm).
    * roads:     líneas con flag "blocked".
    * sar_water: agua observada por Sentinel-1 (capa radar).
    * hotspots:  top N focos críticos para animaciones flyTo de cámara.

Todo se sirve por `GET /api/geo/simulate` con parámetros de escenario, de
modo que cada movimiento de slider re-simula y re-pinta la ciudad 3D.
"""

from __future__ import annotations

import base64
import logging
import math
import os
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import shape

from hidroalert.config import CONFIG, BANKFULL_DEPTH_M, bankfull_for_zone
from hidroalert.data_ingest.aoi import (
    AreaOfInterest,
    DEFAULT_AOI,
    parse_bbox_text,
    resolve_aoi,
)
from hidroalert.data_ingest.gee_client import is_available
from hidroalert.data_ingest.ingest_service import build_area_snapshot, build_demo_snapshot
from hidroalert.hydrology.impact import _lonlat_to_indices
from hidroalert.hydrology.simulation_engine import simulate_flood, ROAD_BLOCK_DEPTH_M
from hidroalert.alerts.engine import evaluate_alert
from hidroalert.models import AreaSnapshot

logger = logging.getLogger(__name__)

# Límites de payload para el render 3D (rendimiento WebGL)
MAX_WATER_CELLS = 4000
MAX_BUILDINGS = 1200
MAX_SAR_CELLS = 1500

# Red de drenaje: umbral de área upstream (km²) para considerar 'cauce'
DRAINAGE_UPA_MIN_KM2 = 25.0
MAX_DRAINAGE_POINTS = 800

# Propagación temporal de la inundación (horas simuladas del evento)
PROPAGATION_HOURS = 6

# Cache de snapshots por bbox (evita re-descargar GEE en cada slider)
_SNAP_CACHE: dict[str, AreaSnapshot] = {}

# Cache PERSISTENTE en serverless (/tmp): las instancias que se reutilizan
# no vuelven a descargar GEE tras un arranque frío. Inactiva en local
# (el directorio no es escribible o no existe la variable).
_SNAP_DISK_DIR = os.environ.get("HIDROALERT_SNAPSHOT_CACHE_DIR", "")


def _snapshot_disk_path(key: str, demo: bool) -> Path:
    """Ruta del snapshot serializado; None si el caché en disco está inactivo."""
    if not _SNAP_DISK_DIR:
        return None
    try:
        d = Path(_SNAP_DISK_DIR)
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{'demo' if demo else 'real'}__{key.replace('/', '_')}.pkl"
    except OSError:
        return None


def _snapshot_load_disk(key: str, demo: bool):
    """Carga el snapshot desde /tmp si existe; None si no."""
    import pickle

    path = _snapshot_disk_path(key, demo)
    if path is None or not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:  # pkl corrupto/versión distinta: se ignora
        return None


def _snapshot_save_disk(key: str, demo: bool, snap: AreaSnapshot) -> None:
    import pickle

    path = _snapshot_disk_path(key, demo)
    if path is None:
        return
    try:
        with open(path, "wb") as fh:
            pickle.dump(snap, fh, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # /tmp lleno u otro fallo: el caché es best-effort
        pass


def get_snapshot(zone_or_bbox: str | None) -> tuple[AreaSnapshot, bool]:
    """
    Snapshot cacheado para una zona (id del registro) o bbox personalizado.
    Usa GEE real si es posible; degrada a demo sin credenciales.

    Orden de resolución: memoria → /tmp (serverless) → GEE → demo.

    Returns:
        (snapshot, es_demo)
    """
    key = (zone_or_bbox or "").strip()
    if not key:
        key = "tacna-yarada-los-palos"  # zona por defecto: Dto. La Yarada-Los Palos (Tacna)
    if key in _SNAP_CACHE:
        return _SNAP_CACHE[key], _SNAP_CACHE[key].aoi_name.startswith("Valle Demo")

    # 1) Caché persistente en /tmp (instancia reutilizada tras arranque frío)
    for demo_flag in (False, True):
        cached = _snapshot_load_disk(key, demo_flag)
        if cached is not None:
            _SNAP_CACHE[key] = cached
            return cached, cached.aoi_name.startswith("Valle Demo")

    if not is_available():
        snap = build_demo_snapshot()
        _SNAP_CACHE[key] = snap
        _snapshot_save_disk(key, True, snap)
        return snap, True
    try:
        aoi = resolve_aoi(key)
        snap = build_area_snapshot(aoi, lookback_days=12)
    except Exception as exc:
        logger.warning("GEE falló (%s); usando área demo", exc)
        snap = build_demo_snapshot()
        _SNAP_CACHE[key] = snap
        _snapshot_save_disk(key, True, snap)
        return snap, True
    _SNAP_CACHE[key] = snap
    _snapshot_save_disk(key, False, snap)
    return snap, False


def _make_elevation_lookup(
    dem: np.ndarray | None,
    bbox: tuple[float, float, float, float],
):
    """
    Devuelve función (lon, lat) → elevación base (m, nivel del mar).

    Usa la matriz DEM del snapshot (MERIT `elv`); si no hay (demo) devuelve 0.
    Es CRÍTICA para Cesium: las entidades deben apoyarse sobre el terreno
    real (La Yarada-Los Palos: 0–200 m snm), no sobre el elipsoide, o
    quedarían enterradas.
    """
    if dem is None or dem.size == 0:
        return lambda lon, lat: 0.0

    nrows, ncols = dem.shape

    def _base_h(lon: float, lat: float) -> float:
        try:
            col, row = _lonlat_to_indices(lon, lat, bbox, ncols, nrows)
            v = float(dem[row, col])
            return v if np.isfinite(v) and v > -100 else 0.0
        except Exception:
            return 0.0

    return _base_h


def _cell_geojson(
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    depth: np.ndarray | None = None,
    base_h=None,
    cap: int = MAX_WATER_CELLS,
) -> dict[str, Any]:
    """
    Máscara booleana → GeoJSON de RECTÁNGULOS FUSIONADOS.

    Las celdas mojadas contiguas se fusionan por fila (runs) y verticalmente
    (mismos extremos en filas adyacentes) en el menor número de polígonos:
    ~3-5× menos features que celda-a-celda, cobertura visual completa (sin
    damero) y payload menor. `__row/__col` queda como el centro del rectángulo
    para que la animación de propagación siga funcionando sin cambios.
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    nrows, ncols = mask.shape
    dlon = (max_lon - min_lon) / ncols
    dlat = (max_lat - min_lat) / nrows
    if not mask.any():
        return {"type": "FeatureCollection", "features": []}

    # --- Fusión: runs horizontales + extensión vertical de runs idénticos ---
    rects: list[dict[str, int]] = []
    open_rects: dict[tuple[int, int], int] = {}  # (c0, c1) → índice con fila abierta
    for r in range(nrows):
        row = mask[r]
        if not row.any():
            open_rects.clear()
            continue
        new_open: dict[tuple[int, int], int] = {}
        c = 0
        while c < ncols:
            if not row[c]:
                c += 1
                continue
            c0 = c
            while c < ncols and row[c]:
                c += 1
            c1 = c - 1
            key = (c0, c1)
            idx = open_rects.get(key)
            if idx is not None and rects[idx]["r1"] == r - 1:
                rects[idx]["r1"] = r          # extiende el rectángulo hacia abajo
            else:
                idx = len(rects)
                rects.append({"r0": r, "r1": r, "c0": c0, "c1": c1})
            new_open[key] = idx
        open_rects = new_open

    # Salvaguarda: aún demasiados rectángulos → filas alternas
    if len(rects) > cap:
        rects = [rc for rc in rects if rc["r0"] % 2 == 0]

    feats: list[dict[str, Any]] = []
    for rc in rects:
        r0, r1, c0, c1 = rc["r0"], rc["r1"], rc["c0"], rc["c1"]
        # 6 decimales ≈ 0.1 m: suficiente para render y ~50 % más liviano
        lat_n = round(max_lat - r0 * dlat, 6)
        lat_s = round(max_lat - (r1 + 1) * dlat, 6)
        lon_w = round(min_lon + c0 * dlon, 6)
        lon_e = round(min_lon + (c1 + 1) * dlon, 6)
        props: dict[str, Any] = {
            "kind": "water",
            "__r0": r0, "__r1": r1, "__c0": c0, "__c1": c1,
            "__row": (r0 + r1) // 2, "__col": (c0 + c1) // 2,  # centro (animación)
        }
        if base_h is not None:
            props["base"] = round(base_h((lon_w + lon_e) / 2, (lat_n + lat_s) / 2), 1)
        if depth is not None:
            dmax = float(depth[r0:r1 + 1, c0:c1 + 1].max())
            props["dclass"] = 2 if dmax > 0.8 else (1 if dmax > 0.3 else 0)
            props["depth_cm"] = round(dmax * 100)
        feats.append({
            "type": "Feature",
            "properties": props,
            "geometry": {"type": "Polygon", "coordinates": [[
                [lon_w, lat_s], [lon_e, lat_s], [lon_e, lat_n], [lon_w, lat_n],
                [lon_w, lat_s],
            ]]},
        })
    return {"type": "FeatureCollection", "features": feats}


def _round_coords(obj: Any) -> Any:
    """Redondea recursivamente los [lon, lat] de cualquier geometría GeoJSON."""
    if isinstance(obj, (list, tuple)):
        if (
            len(obj) >= 2
            and isinstance(obj[0], (int, float))
            and isinstance(obj[1], (int, float))
        ):
            return [round(float(obj[0]), 6), round(float(obj[1]), 6)]
        return [_round_coords(o) for o in obj]
    return obj


def _simplify_line(coords: list, max_pts: int = 80) -> list:
    """Reduce una polilínea a ≤ max_pts vértices (primero y último siempre)."""
    n = len(coords)
    if n <= max_pts:
        return coords
    step = (n - 1) / (max_pts - 1)
    keep = {round(i * step) for i in range(max_pts - 1)}
    keep.add(n - 1)
    return [coords[i] for i in sorted(keep)]


def _simplify_geom(obj: Any, max_pts: int = 80) -> Any:
    """Simplifica recursivamente las líneas de una geometría GeoJSON."""
    if not isinstance(obj, list) or not obj:
        return obj
    is_point = (
        len(obj) >= 2
        and isinstance(obj[0], (int, float))
        and isinstance(obj[1], (int, float))
    )
    if is_point:
        return obj
    is_line = all(
        isinstance(o, (list, tuple))
        and len(o) >= 2
        and isinstance(o[0], (int, float))
        and isinstance(o[1], (int, float))
        for o in obj
    )
    if is_line:
        return _simplify_line(obj, max_pts)
    return [_simplify_geom(o, max_pts) for o in obj]


def _roads_geojson(
    snapshot: AreaSnapshot,
    result,
    bbox: tuple[float, float, float, float],
) -> tuple[dict[str, Any], int, int]:
    """
    Vías con flag de bloqueo (muestreo de vértices sobre la lámina).

    Returns:
        (geojson, bloqueadas, total)
    """
    feats_out: list[dict[str, Any]] = []
    blocked = 0
    total = 0
    if not snapshot.roads:
        return {"type": "FeatureCollection", "features": []}, 0, 0
    nrows, ncols = result.flood_mask.shape

    def _flatten_coords(obj):
        """Aplina recursivamente coordenadas GeoJSON hasta pares [lon, lat]."""
        if (
            isinstance(obj, (list, tuple))
            and len(obj) >= 2
            and isinstance(obj[0], (int, float))
            and isinstance(obj[1], (int, float))
        ):
            return [obj]
        out = []
        for item in obj or []:
            out.extend(_flatten_coords(item))
        return out

    for feat in snapshot.roads["features"]:
        try:
            geom = feat["geometry"]
        except (KeyError, TypeError):
            continue
        coords = _flatten_coords(geom.get("coordinates", []))
        max_d = 0.0
        for lon, lat in coords[:: max(1, len(coords) // 60) or None]:
            col, row = _lonlat_to_indices(lon, lat, bbox, ncols, nrows)
            max_d = max(max_d, float(result.depth_m[row, col]))
        is_blocked = max_d >= 0.20
        blocked += int(is_blocked)
        total += 1
        # Simplificar + redondear las COORDENATES del dict de geometría
        # (los helpers esperan listas; antes recibían el dict completo y no
        # hacían nada — de ahí que las vías pesaran siempre 446 KB).
        coords_ligeras = _round_coords(_simplify_geom(geom.get("coordinates", [])))
        feats_out.append({
            "type": "Feature",
            "properties": {
                "road_id": feat.get("properties", {}).get("road_id", feat.get("id")),
                "blocked": is_blocked,
                "depth_cm": round(max_d * 100),
            },
            "geometry": {**geom, "coordinates": coords_ligeras},
        })
    return {"type": "FeatureCollection", "features": feats_out}, blocked, total


def _component_cells(
    river: np.ndarray, visited: np.ndarray, r0: int, c0: int,
) -> tuple[list[int], list[int]]:
    """
    Celdas de la componente conexa (4-conexa) que contiene (r0, c0), sin
    marcarlas como visitadas (solo lectura). Limita el tamaño por rendimiento.
    """
    from collections import deque

    nrows, ncols = river.shape
    comp_r, comp_c = [], []
    q = deque([(r0, c0)])
    seen = {(r0, c0)}
    while q and len(comp_r) < 20_000:
        r, c = q.popleft()
        comp_r.append(r)
        comp_c.append(c)
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < nrows and 0 <= cc < ncols
                and river[rr, cc] and (rr, cc) not in seen
            ):
                seen.add((rr, cc))
                q.append((rr, cc))
    return comp_r, comp_c


def _drainage_geojson(
    snapshot: AreaSnapshot,
    bbox: tuple[float, float, float, float],
    base_h=None,
) -> dict[str, Any]:
    """
    Red de drenaje 3D: polilíneas del cauce principal (celdas con área
    upstream > DRAINAGE_UPA_MIN_KM2 de MERIT Hydro 'upa'), ordenadas
    río abajo por conexión con la celda vecina de mayor 'upa'.

    Cada tramo lleva 'order' (1=cauce primario) y 'flow' (dirección de la
    corriente) para que el frontend anime las partículas de flujo.
    """
    upa = snapshot.upa
    if upa is None or upa.size == 0:
        return {"type": "FeatureCollection", "features": []}

    nrows, ncols = upa.shape
    min_lon, min_lat, max_lon, max_lat = bbox
    dlon = (max_lon - min_lon) / ncols
    dlat = (max_lat - min_lat) / nrows

    river = upa >= DRAINAGE_UPA_MIN_KM2
    if not river.any():
        return {"type": "FeatureCollection", "features": []}

    # Tramos del cauce: camino ORDENADO río abajo. En hidrología el área
    # upstream crece hacia la desembocadura, así que se camina desde la celda
    # de menor 'upa' (naciente) al vecino de mayor 'upa' (downstream): la
    # polilínea resultante es continua y físicamente direccional.
    #
    # RENDIMIENTO: `processed` marca la componente entera tras analizarla UNA
    # vez; los tramos adicionales salen de sus celdas ordenadas por 'upa' sin
    # re-recorrer la componente (antes era O(celdas²): ~58 s en modo demo).
    visited = np.zeros_like(river, dtype=bool)      # celdas ya caminadas
    processed = np.zeros_like(river, dtype=bool)     # componentes ya analizadas
    feats: list[dict[str, Any]] = []
    order = 0
    for r0, c0 in zip(*np.nonzero(river)):
        if processed[r0, c0]:
            continue
        comp_r, comp_c = _component_cells(river, processed, int(r0), int(c0))
        processed[comp_r, comp_c] = True  # esta componente no vuelve a analizarse
        # Celdas de la componente ordenadas por 'upa' ascendente: cada tramo
        # parte de la naciente (menor upa) aún no caminada.
        comp_upa = upa[comp_r, comp_c]
        for k in np.argsort(comp_upa):
            r, c = int(comp_r[k]), int(comp_c[k])
            if visited[r, c]:
                continue
            order += 1
            visited[r, c] = True

            pts: list[tuple[float, float, float]] = []
            while len(pts) < MAX_DRAINAGE_POINTS:
                lon = min_lon + (c + 0.5) * dlon
                lat = max_lat - (r + 0.5) * dlat
                elev = float(base_h(lon, lat)) if base_h else 0.0
                pts.append((round(lon, 6), round(lat, 6), round(elev + 6.0, 1)))
                # Avanza al vecino de cauce NO visitado con MAYOR upa (río abajo)
                best = None
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):  # 8-conexo
                    rr, cc = r + dr, c + dc
                    if (
                        0 <= rr < nrows and 0 <= cc < ncols
                        and river[rr, cc] and not visited[rr, cc]
                    ):
                        if best is None or upa[rr, cc] > upa[best[1], best[2]]:
                            best = (float(upa[rr, cc]), rr, cc)
                if best is None:
                    break
                _, r, c = best
                visited[r, c] = True
            if len(pts) >= 3:
                feats.append({
                    "type": "Feature",
                    "properties": {
                        "order": order,
                        "upa_max_km2": round(float(np.nanmax(upa[river])), 1),
                        "flow": "downstream",
                    },
                    "geometry": {"type": "LineString", "coordinates": pts},
                })
            if len(feats) >= 12:  # límite de tramos renderizables
                break
        if len(feats) >= 12:
            break
    return {"type": "FeatureCollection", "features": feats}


def _propagation_series(
    snapshot: AreaSnapshot,
    precipitation_mm_h: float,
    flow_increase_pct: float,
    bankfull_m: float | None = None,
    include_overland: bool = False,
) -> list[dict[str, Any]]:
    """
    Serie temporal de propagación de la inundación (0 → PROPAGATION_HOURS h).

    Modelo de concentración: la lámina del cauce crece desde el caudal base
    hasta el pico del escenario con una curva en S (hide-and-decay simple);
    cada paso re-simula el ráster y guarda el área/volumen acumulado para
    que el frontend anime la expansión progresiva de la mancha.
    """
    steps: list[dict[str, Any]] = []
    for h in range(PROPAGATION_HOURS + 1):
        frac = h / PROPAGATION_HOURS
        # Curva de concentración en S (crecimiento suave, satura al final)
        factor = frac * frac * (3.0 - 2.0 * frac)
        p_h = precipitation_mm_h * (0.25 + 0.75 * factor)
        q_h = flow_increase_pct * (0.20 + 0.80 * factor)
        result = simulate_flood(
            snapshot,
            precipitation_mm_h=float(p_h),
            flow_increase_pct=float(q_h),
            fast_mode=True,
            bankfull_depth_m=bankfull_m,
            include_overland=include_overland,
        )
        hid = result.to_dict()["hydrology"]
        steps.append({
            "hour": h,
            "precipitation_mm_h": round(p_h, 1),
            "flow_increase_pct": round(q_h, 1),
            "flooded_ha": hid["flooded_area_ha"],
            "volume_m3": hid["water_volume_m3"],
            "max_depth_m": hid["max_depth_m"],
            # Máscara comprimida zlib+base64: ~97 % más liviana que el hex
            # (ráster mayormente ceros). El navegador la descomprime con
            # DecompressionStream; sin soporte, la animación se omite.
            "mask": base64.b64encode(
                zlib.compress(result.flood_mask.astype(np.uint8).tobytes(), 6)
            ).decode("ascii"),
            "mask_encoding": "zlib",
            "shape": list(result.flood_mask.shape),
        })
    return steps


def _buildings_points(
    snapshot: AreaSnapshot,
    result,
    bbox: tuple[float, float, float, float],
    base_h=None,
    cap_unaffected: int = MAX_BUILDINGS,
) -> tuple[dict[str, Any], int, int]:
    """
    Viviendas como PUNTOS (centroide) para el mando 3D.

    Más visibles a cualquier zoom que las huellas extruidas y ~8× más
    livianas en el payload (sin anillos de polígono). Se envían TODAS las
    afectadas (son el objeto de la alerta) + una muestra de las ilesas.
    """
    if not snapshot.buildings:
        return {"type": "FeatureCollection", "features": []}, 0, 0
    nrows, ncols = result.flood_mask.shape
    evaluated: list[tuple[float, float, float, bool, float, float]] = []
    for feat in snapshot.buildings["features"]:
        try:
            ring = feat["geometry"]["coordinates"][0]
            pts = [(p[0], p[1]) for p in ring if len(p) >= 2]
        except (KeyError, IndexError, TypeError):
            continue
        if len(pts) < 3:
            continue
        clon = round(sum(p[0] for p in pts) / len(pts), 6)
        clat = round(sum(p[1] for p in pts) / len(pts), 6)
        col, row = _lonlat_to_indices(clon, clat, bbox, ncols, nrows)
        d = float(result.depth_m[row, col])
        props = feat.get("properties", {})
        h = props.get("height") or props.get("Height")
        floors = props.get("num_floors") or props.get("NumFloors")
        height_m = float(h) if h else (float(floors) * 3.0 if floors else 6.0)
        elev = round(base_h(clon, clat), 1) if base_h else 0.0
        evaluated.append((clon, clat, d, d >= 0.15, max(3.0, height_m), elev))

    aff = [e for e in evaluated if e[3]]
    unaff = [e for e in evaluated if not e[3]]
    step = max(1, -(-len(unaff) // max(1, cap_unaffected - len(aff))))

    out: list[dict[str, Any]] = []
    affected = 0
    for lon, lat, d, is_aff, h, elev in aff + unaff[::step]:
        affected += int(is_aff)
        out.append({
            "type": "Feature",
            "properties": {
                "affected": is_aff,
                "depth_cm": round(d * 100),
                "height": h,
                "base": elev,
            },
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
        })
    return {"type": "FeatureCollection", "features": out}, affected, len(out)


def _dist_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Distancia aproximada en metros (equirectangular; exacta a escala local)."""
    mid_lat = math.radians((lat1 + lat2) / 2)
    dx = (lon2 - lon1) * 111_320.0 * math.cos(mid_lat)
    dy = (lat2 - lat1) * 110_540.0
    return math.hypot(dx, dy)


def _hotspot_reasons(
    hotspots: list[dict[str, Any]],
    result,
    roads_gj: dict[str, Any],
) -> None:
    """
    Anota cada foco crítico con `level` y `why`: la enumeración de motivos
    por los que el punto es una zona de peligro.

    Criterios por foco:
      1. Lámina local contra los umbrales del motor de alertas
         (0.80 m ROJO / 0.30 m AMARILLO).
      2. Viviendas afectadas en un radio de 150 m (concentración de daños).
      3. Vías bloqueadas a menos de 250 m (riesgo de aislamiento).
      4. Cercanía al punto de lámina máxima (epicentro del desborde).
    """
    t = CONFIG.thresholds

    # --- Contexto global del evento (una vez, no por foco) ---
    aff = [(b.centroid[0], b.centroid[1]) for b in result.impact.buildings_detail]
    aff_lons = np.array([p[0] for p in aff]) if aff else np.empty(0)
    aff_lats = np.array([p[1] for p in aff]) if aff else np.empty(0)

    blocked_rds: list[list[tuple[float, float]]] = []  # vértices por vía bloqueada
    for f in roads_gj.get("features", []):
        if not f.get("properties", {}).get("blocked"):
            continue
        pts: list[tuple[float, float]] = []

        def _flat(o) -> None:
            if (
                isinstance(o, (list, tuple)) and len(o) >= 2
                and isinstance(o[0], (int, float)) and isinstance(o[1], (int, float))
            ):
                pts.append((o[0], o[1]))
            else:
                for it in o or []:
                    _flat(it)

        _flat(f["geometry"].get("coordinates", []))
        if pts:
            blocked_rds.append(pts)

    max_pt = next((h for h in hotspots if h.get("kind") == "max_depth"), None)

    for h in hotspots:
        d = float(h["depth"])
        why: list[str] = []

        # 1) Lámina local contra umbrales del motor de alertas
        if d >= t.depth_red_m:
            h["level"] = "ROJO"
            why.append(
                f"Lámina de {d:.2f} m: peligro para personas y vehículos "
                f"(umbral ROJO {t.depth_red_m:.2f} m)."
            )
        elif d >= t.depth_yellow_m:
            h["level"] = "AMARILLO"
            why.append(
                f"Lámina de {d:.2f} m de agua sobre el terreno "
                f"(umbral AMARILLO {t.depth_yellow_m:.2f} m)."
            )
        else:
            h["level"] = "AMARILLO"
            why.append(f"Lámina de {d:.2f} m: inundación incipiente en el punto.")

        if h.get("kind") == "building":
            why.append(f"Vivienda con {d * 100:.0f} cm de agua en su centroide.")
        elif h.get("kind") == "max_depth":
            why.append("Punto de lámina máxima del evento: epicentro del desborde.")

        # 2) Concentración de viviendas afectadas en 150 m
        if aff_lons.size:
            ds = np.hypot(
                (aff_lons - h["lon"]) * 111_320.0 * math.cos(math.radians(h["lat"])),
                (aff_lats - h["lat"]) * 110_540.0,
            )
            n_near = int(np.count_nonzero(ds <= 150.0))
            if h.get("kind") == "building":
                n_near -= 1  # la propia vivienda del foco no se cuenta
            if n_near >= 3:
                why.append(
                    f"{n_near} viviendas afectadas en un radio de 150 m: "
                    "concentración de daños."
                )
            elif n_near >= 1:
                why.append(f"{n_near} vivienda(s) afectada(s) en un radio de 150 m.")

        # 3) Vías bloqueadas en 250 m (aislamiento / evacuación)
        n_roads = 0
        for rd in blocked_rds:
            for lon, lat in rd[:: max(1, len(rd) // 12)]:
                if _dist_m(h["lon"], h["lat"], lon, lat) <= 250.0:
                    n_roads += 1
                    break
        if n_roads:
            why.append(
                f"{n_roads} vía(s) bloqueada(s) a menos de 250 m: riesgo de "
                "aislamiento y corte de rutas de evacuación."
            )

        # 4) Cercanía al epicentro del desborde
        if max_pt is not None and h is not max_pt:
            dist = _dist_m(h["lon"], h["lat"], max_pt["lon"], max_pt["lat"])
            if dist <= 400.0:
                why.append(
                    f"A {dist:.0f} m del punto de máxima lámina "
                    f"({float(max_pt['depth']):.2f} m)."
                )

        h["why"] = why


def _roads_reasons(
    roads_gj: dict[str, Any],
    result,
    huayco_risk: bool,
    max_pt: dict[str, Any] | None,
) -> None:
    """
    Anota cada VÍA BLOQUEADA con `why` (motivos de peligro), `level` y su
    punto medio `mid` — para la ficha "¿por qué es zona de peligro?" que se
    abre al hacer clic sobre la vía o su marcador 🚧 en el mapa 3D.
    """
    t = CONFIG.thresholds
    aff = [(b.centroid[0], b.centroid[1]) for b in result.impact.buildings_detail]
    aff_lons = np.array([p[0] for p in aff]) if aff else np.empty(0)
    aff_lats = np.array([p[1] for p in aff]) if aff else np.empty(0)

    for f in roads_gj.get("features", []):
        p = f.get("properties", {})
        if not p.get("blocked"):
            continue
        # Punto medio de la vía: referencia para el marcador 🚧 y distancias
        pts: list[tuple[float, float]] = []

        def _flat(o) -> None:
            if (
                isinstance(o, (list, tuple)) and len(o) >= 2
                and isinstance(o[0], (int, float)) and isinstance(o[1], (int, float))
            ):
                pts.append((o[0], o[1]))
            else:
                for it in o or []:
                    _flat(it)

        _flat(f["geometry"].get("coordinates", []))
        if not pts:
            continue
        lon = sum(x[0] for x in pts) / len(pts)
        lat = sum(x[1] for x in pts) / len(pts)
        p["mid"] = [round(lon, 6), round(lat, 6)]

        d = (p.get("depth_cm") or 0) / 100.0
        why: list[str] = []
        if d >= t.depth_red_m:
            p["level"] = "ROJO"
            why.append(
                f"Lámina de {d:.2f} m sobre la calzada: corriente capaz de "
                f"arrastrar personas y vehículos (umbral ROJO {t.depth_red_m:.2f} m)."
            )
        else:
            p["level"] = "AMARILLO"
            why.append(
                f"Lámina de {d:.2f} m sobre la calzada (umbral de bloqueo "
                f"{ROAD_BLOCK_DEPTH_M:.2f} m): vía intransitable."
            )

        # Viviendas damnificadas que sirve esta vía
        if aff_lons.size:
            ds = np.hypot(
                (aff_lons - lon) * 111_320.0 * math.cos(math.radians(lat)),
                (aff_lats - lat) * 110_540.0,
            )
            n_near = int(np.count_nonzero(ds <= 150.0))
            if n_near:
                why.append(
                    f"{n_near} vivienda(s) afectada(s) en un radio de 150 m: "
                    "la vía da servicio a población damnificada."
                )

        why.append(
            "Corte potencial de rutas de evacuación y acceso de ayuda: "
            "riesgo de aislamiento."
        )

        if max_pt is not None:
            dist = _dist_m(lon, lat, max_pt["lon"], max_pt["lat"])
            if dist <= 400.0:
                why.append(
                    f"A {dist:.0f} m del punto de máxima lámina "
                    f"({float(max_pt['depth']):.2f} m)."
                )
        if huayco_risk:
            why.append(
                "Escenario torrencial: posible flujo de detritos (huayco) "
                "sobre la vía."
            )
        p["why"] = why


def _farmland_flood(
    snapshot: AreaSnapshot,
    result,
    bbox: tuple[float, float, float, float],
    huayco_risk: bool,
    max_pt: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], float]:
    """
    Terrenos agrícolas que SERÁN INUNDADOS por el escenario, por fundo.

    Para cada polígono de suelo agrícola (ESA WorldCover 'Cropland', clase 40)
    intersecta la máscara de inundación —vectorizado con shapely.contains_xy,
    solo las celdas del bbox del fundo— y produce un marcador con hectáreas
    sumergidas, lámina media y motivos de peligro para la ficha al hacer clic.

    Returns:
        (marcadores: top-8 fundos por hectáreas, hectáreas totales inundadas)
    """
    fl = snapshot.farmland_polygon
    if not fl or not result.flood_mask.any():
        return [], 0.0
    feats = fl.get("features") or [fl]
    mask = result.flood_mask
    nrows, ncols = mask.shape
    min_lon, min_lat, max_lon, max_lat = bbox
    # Resolución real de celda (m) y área por grado² para el % del fundo
    cell_m = (
        (max_lon - min_lon) / ncols * 111_320.0
        * math.cos(math.radians((min_lat + max_lat) / 2))
    )
    cell_area_m2 = cell_m * cell_m
    deg_area_ha = 111_320.0 * 110_540.0 / 10_000.0

    out: list[dict[str, Any]] = []
    for f in feats:
        try:
            g = shape(f["geometry"])
        except (KeyError, TypeError, ValueError):
            continue
        if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        w, s, e, n = g.bounds
        c0 = max(0, int((w - min_lon) / (max_lon - min_lon) * ncols))
        c1 = min(ncols - 1, int((e - min_lon) / (max_lon - min_lon) * ncols))
        r0 = max(0, int((max_lat - n) / (max_lat - min_lat) * nrows))
        r1 = min(nrows - 1, int((max_lat - s) / (max_lat - min_lat) * nrows))
        if c0 > c1 or r0 > r1:
            continue
        rr, cc = np.meshgrid(
            np.arange(r0, r1 + 1), np.arange(c0, c1 + 1), indexing="ij")
        lons = min_lon + (cc + 0.5) / ncols * (max_lon - min_lon)
        lats = max_lat - (rr + 0.5) / nrows * (max_lat - min_lat)
        inside = shapely.contains_xy(g, lons, lats)
        if not inside.any():
            continue
        wet = inside & mask[rr, cc]
        n_wet = int(wet.sum())
        if n_wet == 0:
            continue
        ha = n_wet * cell_area_m2 / 10_000.0
        poly_ha = g.area * deg_area_ha
        pct = min(100.0, ha / poly_ha * 100.0) if poly_ha > 0 else 100.0
        dmean = float(result.depth_m[rr[wet], cc[wet]].mean())
        clon = float(lons[wet].mean())
        clat = float(lats[wet].mean())

        why = [
            f"{ha:,.1f} ha de terreno agrícola bajo agua ({pct:.0f} % del fundo)."
        ]
        if dmean >= 0.30:
            why.append(
                f"Lámina media de {dmean * 100:.0f} cm sobre el cultivo: "
                "pérdida de la campaña en curso."
            )
        else:
            why.append(
                f"Lámina media de {dmean * 100:.0f} cm: encharcamiento con "
                "daño parcial del cultivo."
            )
        why.append(
            "Impacto económico: pérdida de cosecha; suelos requieren "
            "rehabilitación después del evento."
        )
        if max_pt is not None:
            dist = _dist_m(clon, clat, max_pt["lon"], max_pt["lat"])
            if dist <= 1000.0:
                why.append(
                    f"A {dist:.0f} m del punto de máxima lámina "
                    f"({float(max_pt['depth']):.2f} m)."
                )
        if huayco_risk:
            why.append(
                "Escenario torrencial: posible depósito de sedimentos y "
                "piedras (huayco) sobre los suelos agrícolas."
            )
        out.append({
            "lon": clon, "lat": clat,
            "ha": round(ha, 1),
            "pct": round(pct),
            "depth_cm": round(dmean * 100),
            "level": "ROJO" if dmean >= CONFIG.thresholds.depth_red_m else "AMARILLO",
            "why": why,
        })
    out.sort(key=lambda m: -m["ha"])
    total = float(sum(m["ha"] for m in out))
    return out[:8], round(total, 1)


def simulate_geo(
    zone_or_bbox: str | None,
    precipitation_mm_h: float,
    flow_increase_pct: float,
    include_overland: bool = False,
    include_sar: bool = False,
) -> dict[str, Any]:
    """
    Simulación completa + payload 3D (una llamada por cambio de slider).

    Args:
        zone_or_bbox: id de zona del registro (`/zones`) o bbox personalizado
            "min_lon,min_lat,max_lon,max_lat". None = zona por defecto.
        include_overland: añade la concentración de agua sobre terreno
            (escorrentía D8) a la lámina y al KPI de escorrentía.
        include_sar: incluye la capa de agua observada por Sentinel-1.
            Bajo demanda (por defecto NO viaja): el frontend la pide solo
            cuando la capa "Radar SAR" está activada.
    """
    snapshot, demo = get_snapshot(zone_or_bbox)
    # Bankfull calibrado por río (Caplina 1.2 m)
    bankfull = bankfull_for_zone(zone_or_bbox, snapshot.aoi_name)
    result = simulate_flood(
        snapshot,
        precipitation_mm_h=precipitation_mm_h,
        flow_increase_pct=flow_increase_pct,
        fast_mode=True,  # series interactivas: omite intersección agrícola pesada
        bankfull_depth_m=bankfull,
        include_overland=include_overland,
    )

    bbox = result.bbox
    # Elevación DEM real por celda: SIN esto las entidades 3D de Cesium
    # quedarían enterradas bajo el terreno (La Yarada-Los Palos: 0–200 m snm).
    base_h = _make_elevation_lookup(snapshot.dem, bbox)
    roads_gj, roads_blocked, roads_total = _roads_geojson(snapshot, result, bbox)
    drainage_gj = _drainage_geojson(snapshot, bbox, base_h=base_h)
    propagation = _propagation_series(
        snapshot, precipitation_mm_h, flow_increase_pct,
        bankfull_m=bankfull, include_overland=include_overland,
    )
    # Inyecta el conteo de vías para que el motor de alertas lo considere
    result.impact.roads_blocked = roads_blocked
    result.impact.roads_total = roads_total

    decision = evaluate_alert(result)
    hid = result.to_dict()["hydrology"]

    water_gj = _cell_geojson(result.flood_mask, bbox, depth=result.depth_m, base_h=base_h)
    sar_gj = (
        _cell_geojson(snapshot.flood_mask_observed, bbox, base_h=base_h, cap=MAX_SAR_CELLS)
        if include_sar and snapshot.flood_mask_observed is not None
        else {"type": "FeatureCollection", "features": []}
    )
    bld_gj, bld_aff, bld_total = _buildings_points(snapshot, result, bbox, base_h=base_h)

    # Hotspots: TODAS las viviendas afectadas (lista completa en scroll en el
    # panel) ordenadas por profundidad + la celda de lámina máxima.
    hotspots: list[dict[str, Any]] = []
    for b in result.impact.buildings_detail:
        hotspots.append({
            "lon": b.centroid[0], "lat": b.centroid[1],
            "label": f"Vivienda · {b.depth_m * 100:.0f} cm",
            "depth": b.depth_m,
            "kind": "building",
        })
    hotspots.sort(key=lambda h: -h["depth"])
    if result.flood_mask.any():
        r, c = np.unravel_index(np.argmax(result.depth_m), result.depth_m.shape)
        min_lon, min_lat, max_lon, max_lat = bbox
        nrows, ncols = result.flood_mask.shape
        hotspots.append({
            "lon": min_lon + (c + 0.5) / ncols * (max_lon - min_lon),
            "lat": max_lat - (r + 0.5) / nrows * (max_lat - min_lat),
            "label": f"Lámina máxima · {result.depth_m[r, c]:.2f} m",
            "depth": float(result.depth_m[r, c]),
            "kind": "max_depth",
        })
    # Enumeración de motivos de peligro por foco (para la ficha al hacer clic)
    _hotspot_reasons(hotspots, result, roads_gj)

    # Fichas de vías bloqueadas (igual que las viviendas: clic → por qué)
    huayco_risk = bool(
        result.precipitation_mm_h >= 80 and result.flow_increase_pct >= 200
    )
    max_pt = next((h for h in hotspots if h.get("kind") == "max_depth"), None)
    _roads_reasons(roads_gj, result, huayco_risk, max_pt)

    # Terrenos agrícolas que serán inundados + marcadores 🌾 por fundo
    farm_markers, farmland_ha = _farmland_flood(
        snapshot, result, bbox, huayco_risk, max_pt
    )

    center = [(bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2]

    # --- Riesgo de HUAYCO (falla de quebrada / flujo de detritos) ---
    # Criterio 5 del motor de alertas: escenario torrencial
    # (lluvia >= 80 mm/h Y caudal >= 200 %).
    huayco = {
        "risk": huayco_risk,
        "precip_mm_h": precipitation_mm_h,
        "flow_pct": flow_increase_pct,
        "trigger": "lluvia ≥ 80 mm/h y caudal ≥ 200 %",
        "message": (
            "⚠️ RIESGO DE HUAYCO: escenario torrencial — posible falla de "
            "quebrada y flujo de detritos (huayco). Activar protocolo de "
            "evacuación inmediata en quebradas activas."
            if huayco_risk else
            "Sin condiciones de huayco: se requiere lluvia ≥ 80 mm/h y "
            "caudal ≥ 200 % simultáneos."
        ),
    }

    # --- Modelos de simulación utilizados (ficha técnica) ---
    nrows, ncols = result.flood_mask.shape
    cell_m = round(
        (bbox[2] - bbox[0]) / ncols * 111_320.0
        * math.cos(math.radians((bbox[1] + bbox[3]) / 2)),
        1,
    )
    models = {
        "hydraulic": (
            "Llenado hidráulico HAND (Height Above Nearest Drainage, MERIT "
            "Hydro ~90 m): una celda se inunda cuando la lámina del cauce "
            "supera su altura sobre el drenaje."
        ),
        "bankfull": (
            f"Desborde bankfull calibrado por río ({bankfull:.2f} m en esta "
            "zona): solo el agua que excede la capacidad del cauce inunda "
            "la llanura."
        ),
        "runoff": (
            "Escorrentía superficial ruteada con D8 celda a celda cuesta "
            "abajo (capa opcional: pozas en depresiones del terreno)."
        ),
        "sar": (
            "Detección de agua observada Sentinel-1 GRD: ratio VH/VV, umbral "
            "adaptativo de Otsu y máscara HAND (elimina sombras radar)."
        ),
        "forecast": (
            "Pronóstico de precipitación 48 h: WeatherNext 3 (Google) u "
            "Open-Meteo (ICON·GFS·ECMWF), propagado por el motor hidrológico."
        ),
        "exposure": (
            "Exposición: huellas Open Buildings (Google/Microsoft), vías OSM "
            "y áreas agrícolas intersectadas con la lámina simulada."
        ),
    }

    # --- Precisión: métricas honestas del modelo ---
    # Concordancia simulado vs. agua observada por Sentinel-1 (IoU y CSI,
    # estándar en detección de inundaciones). La observación es la última
    # imagen disponible (condición actual), no el escenario simulado.
    iou_pct = pod_pct = far_pct = None
    obs = snapshot.flood_mask_observed
    if obs is not None and obs.size and (obs.any() or result.flood_mask.any()):
        hits = int(np.logical_and(result.flood_mask, obs).sum())
        misses = int(np.logical_and(~result.flood_mask, obs).sum())
        false_alarms = int(np.logical_and(result.flood_mask, ~obs).sum())
        union = hits + misses + false_alarms
        if union:
            # IoU = CSI = aciertos / unión; POD y FAR complementan el análisis
            iou_pct = round(100.0 * hits / union, 1)
            if hits + misses:
                pod_pct = round(100.0 * hits / (hits + misses), 1)  # detectado/observado
            if hits + false_alarms:
                far_pct = round(100.0 * false_alarms / (hits + false_alarms), 1)
    precision = {
        "cell_m": cell_m,
        "hand_base_m": 90.0,
        "deterministic": True,
        "bankfull_m": bankfull,
        "sar_iou_pct": iou_pct,
        "sar_pod_pct": pod_pct,
        "sar_far_pct": far_pct,
        "sar_area_ha": round(snapshot.observed_water_area_m2 / 1e4, 1),
        "calibration": "Bankfull calibrado por río (oct 2026)",
        "note": (
            "Modelo conceptual determinista de alerta temprana: el resultado "
            "depende del escenario elegido, no de una probabilidad de "
            "ocurrencia. No sustituye modelación hidráulica 2D (HEC-RAS)."
        ),
    }

    return {
        "aoi": {
            "name": snapshot.aoi_name,
            "bbox": list(bbox),
            "center": [center[1], center[0]],  # [lon, lat] para Cesium
            "demo": demo,
            "n_buildings": len(snapshot.buildings["features"]) if snapshot.buildings else 0,
        },
        "scenario": {
            "precipitation_mm_h": precipitation_mm_h,
            "flow_increase_pct": flow_increase_pct,
            "bankfull_m": bankfull,
        },
        "kpi": {
            "level": decision.level,
            "reasons": decision.reasons,
            "volume_m3": hid["water_volume_m3"],
            "flooded_ha": hid["flooded_area_ha"],
            "max_depth_m": hid["max_depth_m"],
            "buildings_affected": bld_aff,
            "buildings_total": bld_total,
            "roads_blocked": roads_blocked,
            "roads_total": roads_total,
            "sar_water_ha": round(snapshot.observed_water_area_m2 / 1e4, 1),
            # Escorrentía concentrada sobre terreno (0 si no se simuló)
            "overland_volume_m3": hid["overland_volume_m3"],
            # Terrenos agrícolas que serán inundados (ha totales)
            "farmland_flooded_ha": farmland_ha,
        },
        "water": water_gj,
        "sar_water": sar_gj,
        "buildings": bld_gj,
        "roads": roads_gj,
        "drainage": drainage_gj,
        "propagation": propagation,
        "hotspots": hotspots,
        "farmland": farm_markers,
        "huayco": huayco,
        "models": models,
        "precision": precision,
    }
