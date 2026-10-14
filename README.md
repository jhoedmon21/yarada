# 🌊 HidroAlert — Monitoreo, Simulación y Alerta Temprana de Inundaciones

Sistema integral de **GIS + Machine Learning + Hidrología Operativa** para la detección
de inundaciones (Sentinel-1 SAR + Otsu + HAND), simulación interactiva de escenarios
(precipitación y caudal), matriz de impacto sobre infraestructura y alertas automáticas
(Twilio WhatsApp/SMS + Webhook Defensa Civil/COER + informe PDF/JSON).

**Workspace:** `jhons-workspace-amus9` · alineado con el workflow de referencia
`satellite-buildings-and-deforestation-detection` (cruce SAR + óptica + edificios).

---

## 🗂️ Estructura del proyecto

```
hidroalert/
├── requirements.txt          # Dependencias (Python 3.10+)
├── .env.example              # Plantilla de credenciales
├── README.md                 # Este archivo
├── config.py                 # Configuración central y umbrales de alerta
├── models.py                 # Dataclasses del dominio (Snapshot, Simulation, Impacto)
├── data_ingest/              # MÓDULO 1 — Ingesta satelital
│   ├── gee_client.py         #   Autenticación Earth Engine (cuenta de servicio / user)
│   ├── aoi.py                #   Área de interés + registro de zonas: distrito de La Yarada-Los Palos (Tacna)
│   ├── sar_flood.py          #   Sentinel-1 GRD: Otsu + máscara HAND → agua observada
│   ├── buildings_roads.py    #   MS Buildings Perú (+ respaldo Google Open Buildings)
│   ├── precipitation.py      #   Lluvia observada IMERG (NASA GPM, base de WN3)
│   ├── forecast.py           #   Pronóstico horario: WeatherNext 3 / Open-Meteo
│   ├── ingest_service.py     #   Orquestador + generador de área demo (offline)
│   └── static_map.py         #   Google Maps Static API (recorte para informes)
├── hydrology/                # MÓDULO 2 — Motor hidrológico
│   ├── simulation_engine.py  #   Lámina por celda (bankfull), volumen, presets
│   ├── concentration.py      #   Concentración de agua sobre terreno (escorrentía D8)
│   ├── timeline.py           #   Pronóstico horario → línea de tiempo de alertas
│   └── impact.py             #   Intersección agua × edificios/vías/agricultura
├── static/                    # MÓDULO 3 — Interfaz web: Centro de Mando 3D
│   └── mando3d.html           #   CesiumJS (servido por la API en /mando3d)
├── dashboard/                # Análisis técnico opcional (no es la UI principal)
│   └── app.py                #   Streamlit + Folium + Plotly
├── alerts/                   # MÓDULO 4 — Alertas
│   ├── engine.py             #   Motor condicional VERDE / AMARILLO / ROJO
│   ├── notifier.py           #   Twilio SMS/WhatsApp + Webhook COER + payload JSON
│   └── report.py             #   Informe PDF/JSON con mapa estático recortado
├── api/                      # API REST
│   └── main.py               #   FastAPI: /simulate, /simulate-and-alert, /health…
└── tests/                    # Pruebas offline (pytest)
    ├── test_hydrology.py
    └── test_api.py
```

---

## 🚀 Puesta en marcha

### 1. Instalación

```bash
cd hidroalert
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Credenciales

```bash
cp .env.example .env
# Edita .env con tus credenciales (GEE, Google Maps, Twilio, Webhook COER)
```

**Google Earth Engine:** el proyecto debe estar [registrado en GEE](https://code.earthengine.google.com/register).
Autenticación interactiva:

```bash
earthengine authenticate
```

o cuenta de servicio: guarda el JSON y apunta `HIDROALERT_GEE_SERVICE_ACCOUNT_JSON` a él.

> 💡 **Sin credenciales también funciona:** el sistema arranca en **modo demo** con un
> valle sintético y edificios hipotéticos, para probar simulación, interfaz 3D, alertas
> e informes end-to-end.

### 3. Interfaz web — Centro de Mando 3D (Módulo 3)

Con la API corriendo (paso 4), abra **http://localhost:8000/mando3d** —
CesiumJS puro en el navegador, sin Streamlit:

* **Zona de vigilancia**: selector con el distrito de La Yarada-Los Palos
  (Tacna) y sus subsectores; la cámara vuela a la pampa costera y el
  bankfull se recalibra por cauce (Caplina 1.2 m).
* **Escenario**: presets *Lluvia Ligera · Moderada · FEN* + sliders finos
  (lluvia 0–200 mm/h, caudal +0–300 %).
* **💧 Escorrentía sobre terreno**: toggle que activa la concentración de
  agua sobre el terreno (ruteo D8) — las pozas se renderizan en 3D, con KPI
  de escorrentía y animación de propagación 6 h que las incluye.
* **🌦️ Predicción 48 h**: línea de tiempo horaria coloreada por nivel de
  alerta con los cruces AMARILLO/ROJO previstos; cada hora reporta el
  volumen y la escorrentía concentrada (tooltip por barra).
* **Capas 3D**: satélite ESRI, agua extruida, **viviendas como puntos**
  (rojas y siempre visibles las afectadas), vías, radar SAR (Sentinel-1),
  red de drenaje con partículas de flujo.
* **Focos críticos**: lista COMPLETA de viviendas afectadas (scrollable,
  ordenada por profundidad, con flyTo por vivienda) + lámina máxima.
* **Acciones**: 🚨 disparar alerta (Twilio/Webhook + captura 3D), 🧾 boletín
  PDF, 🎥 sobrevuelo orbital, focos críticos con flyTo.

### 4. API REST (backend)

```bash
uvicorn hidroalert.api.main:app --reload --port 8000
# Docs interactivas: http://localhost:8000/docs
```

Ejemplo — simulación completa con alerta:

```bash
curl -X POST "http://localhost:8000/simulate-and-alert?notify=true" \
  -H "Content-Type: application/json" \
  -d '{"use_demo_data": true, "precipitation_mm_h": 150, "flow_increase_pct": 300}'
```

### 5. Pruebas

```bash
pytest -q
```

---

## 🧠 Modelo hidrológico (Módulo 2)

Conceptualmente, el campo **HAND** (Height Above Nearest Drainage, del dataset
`MERIT/Hydro`) describe cuánto debe subir el agua en el cauce para alcanzar cada celda:

```
lámina(celda) = max(0, lámina_cauce − HAND(celda))
lámina_cauce  = α · escorrentía(mm/h) + β · √(% incremento caudal)
Volumen (m³)  = Σ lámina(celda) · área_celda   =   Área inundada × Profundidad media
```

La constante de Manning (`Q ∝ h^{5/3}`) justifica la relación raíz-caudal→altura;
los coeficientes (`RAINFALL_DEPTH_COEF`, `FLOW_DEPTH_COEF`) son parámetros de
calibración explícitos en `hydrology/simulation_engine.py`.

### Concentración de agua sobre terreno (escorrentía superficial)

`hydrology/concentration.py` añade la hidrología de ladera al modelo de
desborde: cada celda produce escorrentía local (lluvia × coeficiente 0.35),
el agua fluye celda a celda hacia su vecino de máxima pendiente (ruteo D8
sobre el DEM), se infiltra en tránsito y se **concentra en las depresiones**
(pozas, terrazas bajas, zonas planas sin salida). La red de drenaje
(HAND ≤ 1 m) absorbe el flujo que la alcanza — el desborde del cauce ya lo
modela el motor principal, así que no hay doble conteo.

* Opción `include_overland=true` en `POST /simulate` y `/simulate-and-alert`
  (la lámina total = desborde del cauce + pozas de ladera).
* **Incorporada por defecto en la predicción de lluvias**: cada hora de
  `POST /forecast-timeline` reporta `overland_volume_m3` además del volumen
  total, y el nivel de alerta considera la lámina combinada.
* Mando 3D: `GET /api/geo/simulate?overland=true` renderiza las pozas sobre
  el terreno y expone `kpi.overland_volume_m3`.
* Centro de Mando 3D: toggle *💧 Escorrentía sobre terreno* re-simula con las
  pozas visibles en 3D, y el panel *Predicción 48 h* muestra la línea de tiempo
  horaria con cruces de nivel y escorrentía por hora.

### Matriz de impacto

| Elemento | Fuente (GEE) | Criterio |
|---|---|---|
| Viviendas afectadas | Google Open Buildings V3 | lámina ≥ 0.15 m sobre el centroide |
| Vías bloqueadas | GRIP4 | lámina ≥ 0.20 m en algún vértice/segmento |
| Ha. agrícolas sumergidas | ESA WorldCover (Cropland) | celdas inundadas dentro del polígono |

## 🚨 Motor de alertas (Módulo 4)

| Nivel | Disparadores (configurables en `.env`) |
|---|---|
| 🟢 VERDE | Ningún umbral superado |
| 🟡 AMARILLO | ≥ 5 viviendas · lámina ≥ 0.30 m · volumen ≥ 50 000 m³ · vías bloqueadas |
| 🔴 ROJO | ≥ 15 viviendas · lámina ≥ 0.80 m · volumen ≥ 400 000 m³ · escenario torrencial (≥200 % caudal + ≥80 mm/h) |

Al pasar a AMARILLO/ROJO el sistema: **(1)** genera informe JSON + PDF con el mapa
estático recortado de la zona, **(2)** envía SMS/WhatsApp vía Twilio y **(3)** publica
el payload JSON en el webhook de Defensa Civil / COER.

## 🔌 Integración COER / Defensa Civil

El contrato del webhook es el JSON de `build_alert_payload()`:

```json
{
  "sistema": "HidroAlert",
  "workspace": "jhons-workspace-amus9",
  "alerta": { "level": "ROJO", "reasons": ["…"], "thresholds": { } },
  "simulacion": { "hydrology": { "water_volume_m3": 0, "…": 0 }, "impact": { } }
}
```

## 🌦️ Pronóstico estilo WeatherNext 3 (anticipación, no solo detección)

WeatherNext 3 (Google DeepMind) es el salto de **simular lluvia manual → propagar
pronóstico horario satelital**. Su modelo es cerrado, pero el patrón se replica así:

| Fuente | Estado en HidroAlert | Nota |
|---|---|---|
| **Open-Meteo** (gratis, sin key) | ✅ Activo — `GET /forecast` | Pronóstico horario 48–96 h; fallback automático |
| **WeatherNext 3** (Google Weather API) | 🟡 Con `HIDROALERT_GOOGLE_WEATHER_API_KEY` | Misma interfaz; se elige solo si hay key |
| **IMERG** (NASA, lluvia observada) | ✅ Activo — `data_ingest.precipitation` | La misma fuente con la que Google entrena WN3 |
| GraphCast/GenCast, NowcastNet (open source) | 🛣️ Roadmap | Pesos abiertos, requieren GPU |

Endpoints nuevos:

```bash
# Pronóstico horario del AOI
curl "http://localhost:8000/forecast?hours=48"

# Línea de tiempo de alertas: ¿CUÁNDO se cruza AMARILLO/ROJO?
# (incorpora la concentración de agua sobre terreno en cada hora)
curl -X POST "http://localhost:8000/forecast-timeline" \
  -H "Content-Type: application/json" -d '{"use_demo_data": true}'

# Simulación con escorrentía sobre terreno (pozas en depresiones)
curl -X POST "http://localhost:8000/simulate" \
  -H "Content-Type: application/json" \
  -d '{"use_demo_data": true, "precipitation_mm_h": 150, "include_overland": true}'
```

Calibración por quebrada: `HIDROALERT_BANKFULL_DEPTH_M=1.2` (capacidad del cauce
antes de desbordar).

## 🛣️ Extensiones sugeridas

* Ca2M/hora → hidrograma: reemplazar el modelo conceptual por un traslado tipo
  *Nash cascade* o *Muskingum-Cunge* por tramos.
* Intersección edificio-lámina por huella completa (no solo centroide) con
  `rasterio.features.geometry_mask`.
* Entrenamiento de un clasificador (Random Forest sobre VV/VH + HAND + pendiente)
  para reemplazar/añadir robustez al umbral de Otsu.
* Almacenar series temporales en PostGIS y exponer `/history` en la API.
