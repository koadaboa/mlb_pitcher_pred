import pytest
from src.scripts.scrape_mlb_boxscores import parse_api_data, MLB_LEAGUE_IDS


def _base_sample(final=True):
    return {
        "gameData": {
            "game": {
                "calendarEventID": "id-2024-07-04-something",
                "gameNumber": 1,
                "doubleHeader": "N",
                "gameDate": "2024-07-04T17:10:00Z",
            },
            "teams": {
                "away": {"abbreviation": "CLE", "league": {"id": 103}},
                "home": {"abbreviation": "MIN", "league": {"id": 104}},
            },
            "venue": {"location": {"elevation": 500}},
            "weather": {"condition": "Sunny", "wind": "3 mph", "temp": 80},
            "datetime": {"dayNight": "day", "firstPitch": "2024-07-04T17:10:00Z"},
            "status": {"abstractGameState": "Final" if final else "Live"},
        },
        "liveData": {
            "boxscore": {
                "officials": [
                    {"officialType": "Home Plate", "official": {"fullName": "U1"}}
                ],
                "teams": {"away": {"pitchers": [1]}, "home": {"pitchers": [2]}},
            }
        },
    }


def test_parse_api_data_success():
    resp = _base_sample()
    data = parse_api_data(resp, 123)
    assert data["game_pk"] == 123
    assert data["away_team"] == "CLE"
    assert data["home_team"] == "MIN"
    assert data["game_date"] == "2024-07-04"
    assert data["hp_umpire"] == "U1"
    assert data["away_pitcher_ids"] == "[1]"
    assert data["double_header"] == "N"


def test_parse_api_data_non_final_returns_none():
    resp = _base_sample(final=False)
    assert parse_api_data(resp, 123) is None


def test_parse_api_data_missing_fields():
    resp = _base_sample()
    # Remove away team abbreviation to trigger essential data missing
    del resp["gameData"]["teams"]["away"]["abbreviation"]
    assert parse_api_data(resp, 123) is None
