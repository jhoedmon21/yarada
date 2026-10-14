"""
aoi.py — Definición del Área de Interés (AOI).

Un `AreaOfInterest` encapsula el recorte geográfico que el usuario analiza:
 bounding box WGS84, geometría `ee.Geometry` equivalente y resolución de trabajo.
El registro de zonas cubre el distrito de La Yarada-Los Palos (provincia de
Tacna, ubigeo 230111) —distrito completo y subsectores clave—; cualquier otra
zona puede analizarse mediante coordenadas personalizadas (parse_bbox_text).
"""

from __future__ import annotations

from dataclasses import dataclass

from hidroalert.data_ingest.gee_client import get_ee


@dataclass(frozen=True)
class AreaOfInterest:
    """
    Área de interés rectangular en WGS84.

    Atributos:
        name:     nombre descriptivo (distrito, cuenca, quebrada).
        min_lon, min_lat, max_lon, max_lat: límites del rectángulo.
        scale_m:  resolución de trabajo en metros (30 m por defecto).
    """

    name: str
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    scale_m: float = 30.0

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Tupla (min_lon, min_lat, max_lon, max_lat)."""
        return (self.min_lon, self.min_lat, self.max_lon, self.max_lat)

    def to_ee_geometry(self):
        """Convierte el AOI a `ee.Geometry.Rectangle` (requiere GEE inicializado)."""
        ee = get_ee()
        return ee.Geometry.Rectangle([self.min_lon, self.min_lat, self.max_lon, self.max_lat])

    @property
    def diagonal_km(self) -> float:
        """Longitud aproximada de la diagonal del AOI en km (aprox. esférica)."""
        import math

        dlon = (self.max_lon - self.min_lon) * 111.32 * math.cos(
            math.radians((self.min_lat + self.max_lat) / 2)
        )
        dlat = (self.max_lat - self.min_lat) * 110.57
        return math.hypot(dlon, dlat)


# ---------------------------------------------------------------------------
# Registro de zonas de análisis (análisis multi-zona)
#
# FOCO: el distrito de LA YARADA-LOS PALOS (provincia de Tacna, ubigeo
# 230111, Ley N.º 30358 del 07/11/2015): distrito completo (2 320 km²,
# capital Los Palos, fronterizo con Chile) y sus subsectores clave.
# Sistema hidrográfico: río Caplina (curso inferior y acuífero que alimenta
# la irrigación de La Yarada) y quebradas del norte (Cauñani, El Hospicio,
# Agua Dulce). Los bboxes son aproximados por zona; la escala de trabajo
# es de 100 m (análisis distrital).
#
# El usuario también puede analizar cualquier otra zona mediante coordenadas
# personalizadas (parse_bbox_text).
# ---------------------------------------------------------------------------
_ZONE_SCALE_M = 100.0  # resolución de trabajo para distritos completos

ZONE_PRESETS: dict[str, AreaOfInterest] = {
    # ---------- Distrito de LA YARADA-LOS PALOS (Tacna) ----------
    # Ubigeo 230111 · Ley N.º 30358 (07/11/2015) · capital: Los Palos ·
    # 2 320 km² · frontera sur con Chile (punto Concordia). Sistema
    # fluvial: río Caplina (esporádico, crecidas excepcionales FEN) y
    # quebradas del norte. Bboxes referenciales (litoral, valle agrícola
    # y eje del río).
    "tacna-yarada-los-palos": AreaOfInterest(
        name="La Yarada-Los Palos — Distrito completo (provincia de Tacna)",
        min_lon=-70.5853, min_lat=-18.3501, max_lon=-70.3161, max_lat=-18.0776,
        scale_m=_ZONE_SCALE_M,
    ),
    "tacna-yarada": AreaOfInterest(
        name="La Yarada-Los Palos — Valle agrícola La Yarada (irrigación Caplina)",
        min_lon=-70.5650, min_lat=-18.3100, max_lon=-70.4200, max_lat=-18.1500,
        scale_m=_ZONE_SCALE_M,
    ),
    "tacna-los-palos": AreaOfInterest(
        name="La Yarada-Los Palos — Balneario Los Palos y litoral sur (frontera)",
        min_lon=-70.4900, min_lat=-18.3501, max_lon=-70.3800, max_lat=-18.2450,
        scale_m=_ZONE_SCALE_M,
    ),
    "tacna-caplina": AreaOfInterest(
        name="La Yarada-Los Palos — Eje río Caplina (curso inferior y quebradas)",
        min_lon=-70.5853, min_lat=-18.2450, max_lon=-70.3500, max_lat=-18.0776,
        scale_m=_ZONE_SCALE_M,
    ),
}

# AOI por defecto: distrito completo de La Yarada-Los Palos (pampa costera
# fronteriza; crecidas excepcionales del Caplina y huaicos en quebradas).
DEFAULT_AOI = ZONE_PRESETS["tacna-yarada-los-palos"]


def list_zones() -> list[dict]:
    """Catálogo de zonas para los selectores de la UI (dashboard y mando 3D)."""
    return [
        {"id": zid, "name": aoi.name, "bbox": list(aoi.bbox)}
        for zid, aoi in ZONE_PRESETS.items()
    ]


def get_zone(zone_id: str | None) -> AreaOfInterest | None:
    """Resuelve un id de zona del registro; None si no existe o es vacío."""
    if not zone_id:
        return None
    return ZONE_PRESETS.get(zone_id.strip().lower())


def resolve_aoi(key: str | None) -> AreaOfInterest:
    """
    Resuelve un AreaOfInterest a partir de un id de zona del registro o de
    un bbox "min_lon,min_lat,max_lon,max_lat". Cae en DEFAULT_AOI si key es
    vacío/None (comportamiento histórico intacto).
    """
    if not key:
        return DEFAULT_AOI
    zone = get_zone(key)
    if zone is not None:
        return zone
    return parse_bbox_text(key)


def parse_bbox_text(text: str) -> AreaOfInterest:
    """
    Construye un AOI desde texto "min_lon,min_lat,max_lon,max_lat" (WGS84).

    Lanza ValueError si el formato o el orden de coordenadas es inválido.
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        raise ValueError("Formato esperado: min_lon,min_lat,max_lon,max_lat")
    try:
        min_lon, min_lat, max_lon, max_lat = (float(p) for p in parts)
    except ValueError as exc:
        raise ValueError("Las coordenadas deben ser números decimales") from exc
    if min_lon >= max_lon or min_lat >= max_lat:
        raise ValueError("Las coordenadas deben cumplir min < max en ambos ejes")
    if not (-180 <= min_lon < max_lon <= 180 and -90 <= min_lat < max_lat <= 90):
        raise ValueError("Coordenadas fuera de rango WGS84")
    return AreaOfInterest(
        name="AOI personalizado", min_lon=min_lon, min_lat=min_lat, max_lon=max_lon, max_lat=max_lat
    )
