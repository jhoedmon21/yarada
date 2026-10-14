"""
config.py — Configuración central de HidroAlert.

Carga credenciales y parámetros desde variables de entorno (archivo `.env`)
y define los umbrales del motor de alertas (VERDE / AMARILLO / ROJO).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

from dotenv import load_dotenv

# Raíz del paquete: hidroalert/
BASE_DIR = Path(__file__).resolve().parent
# Directorio donde se guardan los informes generados. Sobrescribible con
# HIDROALERT_REPORTS_DIR (en serverless —p. ej. Vercel— el FS es de solo
# lectura salvo /tmp).
REPORTS_DIR = Path(os.environ.get("HIDROALERT_REPORTS_DIR") or (BASE_DIR / "reports"))

# Carga `.env` si existe (no sobreescribe variables ya definidas)
load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str = "") -> str:
    """Lee una variable de entorno con prefijo HIDROALERT_."""
    return os.getenv(f"HIDROALERT_{key}", default)


@dataclass(frozen=True)
class GEEConfig:
    """Credenciales de Google Earth Engine (Módulo 1)."""

    project_id: str = field(default_factory=lambda: _env("GEE_PROJECT_ID"))
    service_account_json: str = field(
        default_factory=lambda: _env("GEE_SERVICE_ACCOUNT_JSON")
    )

    # Colecciones estándar de GEE usadas por el motor
    sentinel1_collection: str = "COPERNICUS/S1_GRD"
    dem_dataset: str = "MERIT/DEM/v1_0_3"          # DEM global ~92 m (celda de HAND)
    hand_dataset: str = "MERIT/Hydro/v1_0_1"       # HAND pre-calculado (banda 'hand')
    # Huellas de edificios: capa comunitaria MS Buildings Perú (verificada en
    # el catálogo actual de GEE). Respaldo: Google Open Buildings V3.
    buildings_dataset: str = "projects/sat-io/open-datasets/MSBuildings/Peru"
    buildings_dataset_alt: str = "GOOGLE/Research/Open-Buildings/V3/polygons"
    # GRIP4 por región (Perú está en Central-South-America)
    roads_dataset: str = "projects/sat-io/open-datasets/GRIP4/Central-South-America"

    @property
    def is_configured(self) -> bool:
        """True si al menos el proyecto GEE fue definido."""
        return bool(self.project_id)


@dataclass(frozen=True)
class GoogleMapsConfig:
    """Credenciales de Google Maps Static API (mapas de los informes)."""

    api_key: str = field(default_factory=lambda: _env("GOOGLE_MAPS_API_KEY"))

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True)
class TwilioConfig:
    """Credenciales de Twilio para SMS / WhatsApp (Módulo 4)."""

    account_sid: str = field(default_factory=lambda: _env("TWILIO_ACCOUNT_SID"))
    auth_token: str = field(default_factory=lambda: _env("TWILIO_AUTH_TOKEN"))
    from_sms: str = field(default_factory=lambda: _env("TWILIO_FROM_SMS"))
    from_whatsapp: str = field(
        default_factory=lambda: _env("TWILIO_FROM_WHATSPAP", "whatsapp:+14155238886")
    )
    recipients: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            r.strip()
            for r in _env("ALERT_RECIPIENTS").split(",")
            if r.strip()
        )
    )

    @property
    def is_configured(self) -> bool:
        return bool(self.account_sid and self.auth_token)


@dataclass(frozen=True)
class WebhookConfig:
    """Webhook institucional hacia Defensa Civil / COER (Módulo 4)."""

    coer_url: str = field(default_factory=lambda: _env("WEBHOOK_COER_URL"))
    timeout_seconds: int = 10

    @property
    def is_configured(self) -> bool:
        return bool(self.coer_url)


@dataclass(frozen=True)
class ForecastConfig:
    """Configuración del módulo de pronóstico (estilo WeatherNext 3)."""

    # API key de Google Maps Platform Weather (WeatherNext 3); vacío = Open-Meteo
    google_weather_key: str = field(default_factory=lambda: _env("GOOGLE_WEATHER_API_KEY"))

    @property
    def is_configured(self) -> bool:
        return bool(self.google_weather_key)


@dataclass(frozen=True)
class AlertThresholds:
    """
    Umbrales del motor de alertas condicionales (Módulo 4).

    Niveles:
        VERDE    — situación normal.
        AMARILLO — precaución: hidrología favorable a inundación.
        ROJO     — evacuación inminente por desborde / falla de quebrada.
    """

    # Viviendas/infraestructura afectadas
    buildings_yellow: int = 5      # >= 5 edificios afectados -> AMARILLO
    buildings_red: int = 15        # >= 15 edificios afectados -> ROJO
    # Profundidad máxima simulada (m)
    depth_yellow_m: float = 0.30   # agua sobre 30 cm en algún punto -> AMARILLO
    depth_red_m: float = 0.80      # agua sobre 80 cm -> ROJO (peligro para personas)
    # Volumen de agua desbordada (m³)
    volume_yellow_m3: float = 50_000.0
    volume_red_m3: float = 400_000.0

    @classmethod
    def from_env(cls) -> "AlertThresholds":
        """Permite sobreescribir umbrales con HIDROALERT_THRESHOLDS_JSON."""
        raw = _env("THRESHOLDS_JSON")
        if not raw:
            return cls()
        try:
            data = json.loads(raw)
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        except (json.JSONDecodeError, TypeError):
            return cls()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CesiumConfig:
    """Token de Cesium Ion para terreno hiperrealista y OSM Buildings (3D)."""

    ion_token: str = field(default_factory=lambda: _env("CESIUM_ION_TOKEN"))

    @property
    def is_configured(self) -> bool:
        return bool(self.ion_token)


@dataclass(frozen=True)
class AppConfig:
    """Configuración agregada de la aplicación."""

    gee: GEEConfig = field(default_factory=GEEConfig)
    gmaps: GoogleMapsConfig = field(default_factory=GoogleMapsConfig)
    twilio: TwilioConfig = field(default_factory=TwilioConfig)
    webhook: WebhookConfig = field(default_factory=WebhookConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    cesium: CesiumConfig = field(default_factory=CesiumConfig)
    thresholds: AlertThresholds = field(default_factory=AlertThresholds.from_env)


# Instancia única de configuración usada por todos los módulos
CONFIG = AppConfig()

# Bankfull configurable (calibración por quebrada/cuenca)
import os as _os
BANKFULL_DEPTH_M = float(_env("BANKFULL_DEPTH_M", "1.0"))

# ---------------------------------------------------------------------------
# Bankfull calibrado por río (m): capacidad de banco específica de cada cauce
# principal del distrito de La Yarada-Los Palos (provincia de Tacna). Se usa
# cuando la zona de análisis corresponde a uno de estos cauces; el resto de
# zonas usa BANKFULL_DEPTH_M global (1.0 m).
# Valor preliminar de ingeniería: el Caplina es un río costero esporádico
# (seco la mayor parte del año, cauce encajonado/diqueado) cuya crecida
# excepcional (FEN) desborda hacia la pampa de La Yarada; su bankfull es
# menor que el de los ríos andinos. Validable con limnómetros ANA.
BANKFULL_BY_RIVER_M: dict[str, float] = {
    "caplina": 1.2,  # río Caplina: curso inferior y paleovalle de La Yarada
}


def bankfull_for_zone(zone_id: str | None, aoi_name: str | None = None) -> float:
    """Devuelve el bankfull (m) aplicable a una zona del registro.

    Reglas:
      * El prefijo del id (``tacna-yarada`` → ``tacna``) o el nombre de la
        zona (busca el cauce en el nombre) determinan el cauce principal.
      * Si no hay coincidencia, aplica el bankfull global (BANKFULL_DEPTH_M).
    """
    key = (zone_id or "").strip().lower()
    name = (aoi_name or "").lower()
    # Zonas del registro (tacna-*): drenan al sistema del río Caplina.
    if key.startswith("tacna-"):
        return BANKFULL_BY_RIVER_M["caplina"]
    for river, depth in BANKFULL_BY_RIVER_M.items():
        if key == river or key.startswith(river + "-") or river in name:
            return depth
    return BANKFULL_DEPTH_M
