"""Fire-weather enrichment (current wind / RH / temperature)."""
from vhagar.weather.open_meteo import fetch_weather, parse_current

__all__ = ["fetch_weather", "parse_current"]
