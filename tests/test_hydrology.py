"""
Pruebas del motor hidrológico y del motor de alertas (offline, sin GEE).

Ejecutar:  pytest -q
"""

from __future__ import annotations

import numpy as np
import pytest

from hidroalert.data_ingest.ingest_service import build_demo_snapshot
from hidroalert.hydrology.simulation_engine import (
    simulate_flood,
    scenario_presets,
    _channel_depth_m,
    DEFAULT_BANKFULL_DEPTH_M,
)
from hidroalert.hydrology.timeline import build_alert_timeline
from hidroalert.alerts.engine import evaluate_alert
from hidroalert.alerts.notifier import compose_alert_message, build_alert_payload


@pytest.fixture(scope="module")
def snapshot():
    """Snapshot demo compartido por las pruebas del módulo."""
    return build_demo_snapshot()


def test_presets_existen():
    presets = scenario_presets()
    assert set(presets.keys()) >= {"Lluvia Ligera", "Moderada", "Torrencial / FEN Extremo"}


def test_escenario_moderado_contenido_en_cauce(snapshot):
    """Lluvia ligera (5·10): el cauce (bankfull 1 m) contiene el agua; sin desborde."""
    result = simulate_flood(snapshot, precipitation_mm_h=5, flow_increase_pct=10)
    assert _channel_depth_m(5, 10) < DEFAULT_BANKFULL_DEPTH_M, "Llg Ligera debe quedar bajo bankfull"
    assert result.flooded_area_m2 == 0
    assert result.water_volume_m3 == 0


def test_fen_extremo_desborda(snapshot):
    """Escenario torrencial: supera bankfull (1.0 global) e inunda la llanura."""
    assert _channel_depth_m(120, 250) > 3.0, "FEN extremo debe superar incluso bf Tumbes"
    result = simulate_flood(snapshot, precipitation_mm_h=120, flow_increase_pct=250)
    assert result.flood_mask.any(), "FEN debe desbordar"
    assert result.water_volume_m3 > 0
    # La lámina sobre terreno es el excedente sobre bankfull
    assert 0 < result.depth_m.max() < _channel_depth_m(120, 250)


def test_monotonia_con_caudal(snapshot):
    """Más caudal ⇒ mismo o mayor volumen y área (respuesta física esperada)."""
    vols, areas = [], []
    for pct in (0, 100, 200, 300):
        r = simulate_flood(snapshot, precipitation_mm_h=30, flow_increase_pct=pct)
        vols.append(r.water_volume_m3)
        areas.append(r.flooded_area_m2)
    assert all(b >= a for a, b in zip(vols, vols[1:]))
    assert all(b >= a for a, b in zip(areas, areas[1:]))
    # Sin lluvia ni caudal ⇒ sin desborde (lámina en cauce = 0 < bankfull)
    assert vols[0] == 0 and areas[0] == 0


def test_monotonia_con_lluvia(snapshot):
    vols = [
        simulate_flood(snapshot, precipitation_mm_h=p, flow_increase_pct=50).water_volume_m3
        for p in (0, 25, 50, 100)
    ]
    assert all(b >= a for a, b in zip(vols, vols[1:]))


def test_matriz_impacto(snapshot):
    r = simulate_flood(snapshot, precipitation_mm_h=120, flow_increase_pct=250)
    assert r.impact.buildings_total > 0
    assert r.impact.buildings_affected <= r.impact.buildings_total
    assert r.impact.roads_total > 0
    assert 0 <= r.impact.farmland_flooded_ha


def test_alerta_verde_con_escenario_calmo(snapshot):
    r = simulate_flood(snapshot, precipitation_mm_h=0, flow_increase_pct=0)
    d = evaluate_alert(r)
    assert d.level == "VERDE"


def test_alerta_rojo_con_fen(snapshot):
    r = simulate_flood(snapshot, precipitation_mm_h=150, flow_increase_pct=300)
    d = evaluate_alert(r)
    assert d.level == "ROJO"
    assert any("EVACUACI" in reason or "ROJO" in reason for reason in d.reasons)


def test_payload_json_serializable(snapshot):
    r = simulate_flood(snapshot, precipitation_mm_h=60, flow_increase_pct=120)
    d = evaluate_alert(r)
    payload = build_alert_payload(r, d)
    # Debe serializar sin error y contener las claves del contrato COER
    import json

    txt = json.dumps(payload, ensure_ascii=False, default=str)
    assert "alerta" in txt and "simulacion" in txt


def test_mensaje_alerta_contiene_nivel(snapshot):
    r = simulate_flood(snapshot, precipitation_mm_h=150, flow_increase_pct=300)
    d = evaluate_alert(r)
    msg = compose_alert_message(r, d)
    assert "HIDROALERT" in msg and d.level in msg


def test_zonas_registro():
    """Registro de zonas: distrito de La Yarada-Los Palos (Tacna) + subsectores."""
    from hidroalert.data_ingest.aoi import (
        list_zones, get_zone, resolve_aoi, DEFAULT_AOI,
    )

    zonas = list_zones()
    ids = {z["id"] for z in zonas}
    # Cobertura = distrito de La Yarada-Los Palos: completo + 3 subsectores
    assert len(zonas) == 4
    assert all(zid.startswith("tacna-") for zid in ids)
    assert not any("camana" in zid or "piura" in zid for zid in ids)
    # El distrito completo encabeza el registro y es el AOI por defecto.
    assert zonas[0]["id"] == "tacna-yarada-los-palos"  # default primero
    assert [z["id"] for z in zonas] == [
        "tacna-yarada-los-palos", "tacna-yarada", "tacna-los-palos", "tacna-caplina",
    ]

    zc = get_zone("TACNA-YARADA")  # case-insensitive
    assert zc is not None and "Valle agrícola" in zc.name
    assert get_zone("no_existe") is None
    assert get_zone("camana-camana") is None  # zona retirada del registro

    # resolve_aoi: vacío -> default; id -> preset; bbox -> personalizado
    assert resolve_aoi("") is DEFAULT_AOI
    assert resolve_aoi(None) is DEFAULT_AOI
    assert resolve_aoi("tacna-los-palos").name.startswith("La Yarada-Los Palos")
    custom = resolve_aoi("-70.53,-18.30,-70.45,-18.25")  # bbox en la pampa de La Yarada
    assert abs(custom.min_lon - (-70.53)) < 1e-9
    # BBox inválido raise (comportamiento esperado)
    import pytest
    with pytest.raises(ValueError):
        resolve_aoi("99,99,100,100")


def test_bankfull_por_distrito_yarada():
    """Bankfull calibrado por río para el distrito de La Yarada-Los Palos (Tacna)."""
    from hidroalert.config import (
        bankfull_for_zone, BANKFULL_BY_RIVER_M, BANKFULL_DEPTH_M,
    )

    # Todas las zonas del registro drenan al sistema del río Caplina
    for zid in ("tacna-yarada-los-palos", "tacna-yarada",
                "tacna-los-palos", "tacna-caplina"):
        assert bankfull_for_zone(zid) == BANKFULL_BY_RIVER_M["caplina"]
    # El nombre del cauce también mapea (AOIs personalizados)
    assert bankfull_for_zone("bbox-custom", "río Caplina — desembocadura") == \
        BANKFULL_BY_RIVER_M["caplina"]
    # Fuera del sistema Caplina: bankfull global
    assert bankfull_for_zone("piura-chira") == BANKFULL_DEPTH_M


def test_concentracion_agua_sobre_terreno(snapshot):
    """Escorrentía superficial D8: la lluvia se concentra sobre el terreno."""
    from hidroalert.hydrology.concentration import (
        overland_concentration, OVERLAND_RUNOFF_COEFF,
    )

    depth, stats = overland_concentration(snapshot, precipitation_mm_h=120)
    assert depth.shape == snapshot.hand.shape
    assert stats["max_depth_m"] > 0
    # Conservación de masa: no puede haber más agua que la lluvia escorrentada
    n_valid = int(np.isfinite(snapshot.dem).sum())
    generado = 120 / 1000.0 * OVERLAND_RUNOFF_COEFF * n_valid * snapshot.cell_area_m2
    assert stats["volume_m3"] <= generado
    # Sin lluvia no hay escorrentía
    _, stats0 = overland_concentration(snapshot, precipitation_mm_h=0)
    assert stats0["volume_m3"] == 0.0

    # La opción integra las pozas a la lámina del desborde (backward compatible)
    base = simulate_flood(snapshot, precipitation_mm_h=120, flow_increase_pct=30)
    con = simulate_flood(snapshot, precipitation_mm_h=120, flow_increase_pct=30,
                         include_overland=True)
    assert con.overland_volume_m3 > 0
    assert con.water_volume_m3 >= base.water_volume_m3
    hid = con.to_dict()["hydrology"]
    assert hid["overland_included"] is True
    assert hid["overland_volume_m3"] > 0
    # Sin la opción el resultado es el histórico
    assert base.to_dict()["hydrology"]["overland_included"] is False
    assert base.to_dict()["hydrology"]["overland_volume_m3"] == 0.0


def test_timeline_incluye_concentracion(snapshot):
    """La predicción de lluvias incorpora la escorrentía por defecto (por hora)."""
    from datetime import datetime, timedelta, timezone
    from hidroalert.hydrology.timeline import build_alert_timeline

    inicio = datetime.now(timezone.utc)
    serie = [
        ((inicio + timedelta(hours=i)).isoformat(), 60.0 if 5 <= i <= 8 else 0.0)
        for i in range(12)
    ]
    tl = build_alert_timeline(
        snapshot,
        {"series": serie, "peak_mm_h": 60.0, "peak_time": serie[6][0],
         "source": "test", "flow_increase_pct": 0},
        max_hours=12,
    )
    # Cada hora del pronóstico reporta el componente concentrado sobre terreno
    assert all("overland_volume_m3" in t for t in tl["timeline"])
    con_lluvia = [t for t in tl["timeline"] if t["mm_h"] >= 60]
    sin_lluvia = [t for t in tl["timeline"] if t["mm_h"] == 0]
    assert all(t["overland_volume_m3"] > 0 for t in con_lluvia)
    assert all(t["overland_volume_m3"] == 0 for t in sin_lluvia)


def test_timeline_detecta_cruce_a_amarillo(snapshot):
    """Tormenta sintética: la línea de tiempo debe detectar el cruce de nivel."""
    from datetime import datetime, timedelta, timezone

    inicio = datetime.now(timezone.utc)
    serie = [
        ((inicio + timedelta(hours=i)).isoformat(), 120.0 if 10 <= i <= 14 else 2.0)
        for i in range(24)
    ]
    fc = {
        "series": serie,
        "peak_mm_h": 120.0,
        "peak_time": serie[12][0],
        "source": "test",
        # Tormenta 120 mm/h +40 %: desborde 0.36 m ≥ 0.30 m ⇒ AMARILLO;
        # hora seca 2 mm/h +40 %: desborde 0.24 m < 0.30 m ⇒ VERDE.
        "flow_increase_pct": 40,
    }
    tl = build_alert_timeline(snapshot, fc, max_hours=24)
    assert len(tl["timeline"]) == 24
    # Debe existir el cruce a AMARILLO/ROJO durante las horas de tormenta (120 mm/h)
    assert "AMARILLO" in tl["rise_to"] or "ROJO" in tl["rise_to"]
    # Las horas de tormenta deben desbordar (volumen > 0)
    con_lluvia = [t for t in tl["timeline"] if t["mm_h"] >= 120]
    assert all(t["volume_m3"] > 0 for t in con_lluvia)
    # El cruce ocurre DENTRO de la ventana de tormenta; las horas secas
    # (2 mm/h) se mantienen en VERDE aunque el caudal base desborde algo.
    hora_cruce = tl["rise_to"].get("AMARILLO") or tl["rise_to"].get("ROJO")
    assert any(t["time"] == hora_cruce and t["mm_h"] >= 120 for t in tl["timeline"])
    assert all(t["level"] == "VERDE" for t in tl["timeline"] if t["mm_h"] < 120)
