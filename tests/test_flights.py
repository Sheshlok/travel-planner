"""Flight tool tests with a mocked SerpAPI client (no network calls)."""

import json

import travel_planner.tools.flights as flights_mod
from travel_planner.tools.flights import search_flights


def _flight(num, dep, arr, price, token=None):
    f = {
        "price": price,
        "total_duration": 50,
        "flights": [{
            "airline": "IndiGo",
            "flight_number": num,
            "departure_airport": {"id": dep, "time": "2026-12-19 07:10"},
            "arrival_airport": {"id": arr, "time": "2026-12-19 07:55"},
        }],
    }
    if token:
        f["departure_token"] = token
    return f


class FakeGoogleSearch:
    calls = []

    def __init__(self, params):
        self.params = params
        FakeGoogleSearch.calls.append(params)

    def get_dict(self):
        if self.params.get("departure_token") == "tok-1":
            return {"best_flights": [_flight("6E 1", "JAI", "DEL", 80)]}
        return {
            "best_flights": [_flight("6E 2164", "DEL", "JAI", 80, token="tok-1")],
            "other_flights": [_flight("AI 2577", "DEL", "JAI", 85, token="tok-2")],
        }


def _run(monkeypatch, **args):
    FakeGoogleSearch.calls = []
    monkeypatch.setattr(flights_mod, "GoogleSearch", FakeGoogleSearch)
    return json.loads(search_flights.invoke(args))


def test_one_way_returns_list_and_makes_one_call(monkeypatch):
    out = _run(monkeypatch, departure_airport="del", arrival_airport="jai", outbound_date="2026-12-19")
    assert isinstance(out, list) and len(out) == 2
    assert out[0]["legs"][0]["flight_number"] == "6E 2164"
    assert len(FakeGoogleSearch.calls) == 1
    assert FakeGoogleSearch.calls[0]["type"] == 2


def test_round_trip_fetches_return_flights_with_departure_token(monkeypatch):
    out = _run(
        monkeypatch,
        departure_airport="DEL", arrival_airport="JAI",
        outbound_date="2026-12-19", return_date="2026-12-22",
    )
    assert out["trip_type"] == "round_trip"
    assert len(out["outbound_options"]) == 2
    assert out["return_options_for_first_outbound"][0]["legs"][0]["departure"].startswith("JAI")
    assert "return_search_error" not in out
    # Second call reuses the search params plus the first flight's token
    assert len(FakeGoogleSearch.calls) == 2
    second = FakeGoogleSearch.calls[1]
    assert second["departure_token"] == "tok-1"
    assert second["return_date"] == "2026-12-22" and second["type"] == 1


def test_round_trip_reports_missing_token(monkeypatch):
    class NoToken(FakeGoogleSearch):
        def get_dict(self):
            return {"best_flights": [_flight("6E 2164", "DEL", "JAI", 80)]}

    FakeGoogleSearch.calls = []
    monkeypatch.setattr(flights_mod, "GoogleSearch", NoToken)
    out = json.loads(search_flights.invoke({
        "departure_airport": "DEL", "arrival_airport": "JAI",
        "outbound_date": "2026-12-19", "return_date": "2026-12-22",
    }))
    assert out["return_options_for_first_outbound"] == []
    assert "return_search_error" in out
