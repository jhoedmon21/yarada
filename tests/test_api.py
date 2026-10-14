"""
Pruebas de la API FastAPI (offline, con datos demo).

Ejecutar:  pytest -q hidroalert/tests/test_api.py
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from hidroalert.api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "gee_configured" in data


def test_scenarios():
    resp = client.get("/scenarios")
    assert resp.status_code == 200
    assert "Lluvia Ligera" in resp.json()["presets"]


def test_zones():
    """Catálogo de zonas: distrito de La Yarada-Los Palos (Tacna) + subsectores."""
    resp = client.get("/zones")
    assert resp.status_code == 200
    data = resp.json()
    assert data["default"] == "tacna-yarada-los-palos"
    assert data["zones"][0]["id"] == "tacna-yarada-los-palos"
    assert len(data["zones"]) == 4
    assert all(z["id"].startswith("tacna-") for z in data["zones"])


def test_simulate_demo():
    resp = client.post(
        "/simulate",
        json={
            "use_demo_data": True,
            "precipitation_mm_h": 150,
            "flow_increase_pct": 250,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    # Con la física de bankfull, solo escenarios que superan 1 m en cauce desbordan
    assert data["hydrology"]["water_volume_m3"] > 0
    assert data["impact"]["buildings_total"] > 0


def test_simulate_demo_moderado_sin_desborde():
    """Escenario bajo bankfull (Lluvia Ligera 5·10): cauce contiene el agua, volumen 0."""
    resp = client.post(
        "/simulate",
        json={"use_demo_data": True, "precipitation_mm_h": 5, "flow_increase_pct": 10},
    )
    assert resp.status_code == 200
    assert resp.json()["hydrology"]["water_volume_m3"] == 0


def test_simulate_overland_demo():
    """La opción include_overland activa la escorrentía sobre terreno en /simulate."""
    resp = client.post(
        "/simulate",
        json={
            "use_demo_data": True,
            "precipitation_mm_h": 120,
            "flow_increase_pct": 30,
            "include_overland": True,
        },
    )
    assert resp.status_code == 200
    h = resp.json()["hydrology"]
    assert h["overland_included"] is True
    assert h["overland_volume_m3"] > 0


def test_simulate_and_alert_demo_rojo():
    resp = client.post(
        "/simulate-and-alert?notify=false&generate_report=false",
        json={
            "use_demo_data": True,
            "precipitation_mm_h": 150,
            "flow_increase_pct": 300,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["alerta"]["level"] == "ROJO"
