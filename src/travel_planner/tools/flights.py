"""Flight search tool (SerpAPI Google Flights engine)."""

import json
import os
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from serpapi import GoogleSearch


class FlightSearchInput(BaseModel):
    """Input schema for flight search."""

    departure_airport: str = Field(..., description="Departure airport code (e.g., 'JFK')")
    arrival_airport: str = Field(..., description="Arrival airport code (e.g., 'LAX')")
    outbound_date: str = Field(..., description="Departure date (YYYY-MM-DD)")
    return_date: Optional[str] = Field(None, description="Return date (YYYY-MM-DD)")
    adults: int = Field(1, description="Number of adults")
    children: int = Field(0, description="Number of children")
    stops: Optional[int] = Field(None, description="0=Any, 1=Nonstop, 2=1 stop or fewer")


@tool(args_schema=FlightSearchInput)
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: Optional[str] = None,
    adults: int = 1,
    children: int = 0,
    stops: Optional[int] = None
) -> str:
    """Search for flights between airports. For round trips, also returns return-flight options."""

    params = {
        'api_key': os.environ.get('SERPAPI_API_KEY'),
        'engine': 'google_flights',
        'departure_id': departure_airport.upper(),
        'arrival_id': arrival_airport.upper(),
        'outbound_date': outbound_date,
        'adults': adults,
        'children': children,
        'currency': 'USD',
        'type': 1 if return_date else 2
    }

    if return_date:
        params['return_date'] = return_date
    if stops is not None:
        params['stops'] = stops

    try:
        results = GoogleSearch(params).get_dict()
        flights = results.get('best_flights', []) + results.get('other_flights', [])

        if not flights:
            return json.dumps({"message": "No flights found"})

        outbound = [_summarize_flight(f) for f in flights[:10]]

        if not return_date:
            return json.dumps(outbound, indent=2)

        # Round trip: Google Flights returns only outbound flights at first.
        # Fetch return options for the first (best) outbound flight using its
        # departure_token. Prices are round-trip totals for that combination.
        return_options = []
        return_error = None
        token = flights[0].get("departure_token")
        if token:
            try:
                ret = GoogleSearch({**params, 'departure_token': token}).get_dict()
                ret_flights = ret.get('best_flights', []) + ret.get('other_flights', [])
                return_options = [_summarize_flight(f) for f in ret_flights[:10]]
            except Exception as e:
                return_error = str(e)
        else:
            return_error = "No departure_token returned for the outbound flight"

        output = {
            "trip_type": "round_trip",
            "note": (
                "Prices are round-trip totals. Each outbound option's price assumes a "
                "matching return; return options listed are for the FIRST outbound "
                "option. Other outbound/return combinations were not searched."
            ),
            "outbound_options": outbound,
            "return_options_for_first_outbound": return_options,
        }
        if return_error:
            output["return_search_error"] = return_error
        return json.dumps(output, indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


def _summarize_flight(f: dict) -> dict:
    """Keep the useful fields of one SerpAPI flight result."""
    return {
        "price": f.get("price"),
        "duration_mins": f.get("total_duration"),
        "airline_logo": f.get("airline_logo"),
        "legs": [{
            "airline": leg.get("airline"),
            "flight_number": leg.get("flight_number"),
            "departure": f"{leg.get('departure_airport', {}).get('id')} {leg.get('departure_airport', {}).get('time')}",
            "arrival": f"{leg.get('arrival_airport', {}).get('id')} {leg.get('arrival_airport', {}).get('time')}",
            "airline_logo": leg.get("airline_logo")
        } for leg in f.get("flights", [])]
    }
