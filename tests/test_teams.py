import pandas as pd
import pytest

from footy.data import teams


def _fixtures(rows, league="EPL", season=2024):
    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "home_goals", "away_goals"])
    df["date"] = pd.to_datetime(df["date"])
    return df.assign(league=league, season=season)


REF = _fixtures([
    ("2024-08-17", "Man United", "Fulham", 1, 0),
    ("2024-08-17", "Arsenal", "Wolves", 2, 0),
    ("2024-08-24", "Fulham", "Arsenal", 1, 1),
    ("2024-08-24", "Wolves", "Man United", 1, 0),
    ("2024-08-31", "Arsenal", "Man United", 1, 0),
    ("2024-08-31", "Fulham", "Wolves", 2, 0),  # same score/day as nothing else -> clean
])
OTHER = REF.replace({
    "Man United": "Manchester United FC", "Wolves": "Wolverhampton Wanderers FC",
    "Arsenal": "Arsenal FC", "Fulham": "Fulham FC",
})
# one source reports the date a day later (timezone) - must still match
OTHER.loc[0, "date"] = pd.Timestamp("2024-08-18")


def test_propose_recovers_all_aliases():
    props = {p.alias: p for p in teams.propose_aliases(REF, OTHER)}
    assert props["Manchester United FC"].canonical == "Man United"
    assert props["Wolverhampton Wanderers FC"].canonical == "Wolves"
    assert all(p.confidence == 1.0 for p in props.values())


def test_merge_keeps_manual_entries_and_flags_weak_votes():
    existing = {"openfootball": {"EPL": {"Arsenal FC": "Arsenal"}}}
    weak = teams.AliasProposal("EPL", "Mystery FC", "Fulham", votes=2, appearances=10,
                               runner_up="Wolves", runner_up_votes=2)
    props = teams.propose_aliases(REF, OTHER) + [weak]
    updated, accepted, review = teams.merge_proposals(existing, "openfootball", props)
    assert updated["openfootball"]["EPL"]["Arsenal FC"] == "Arsenal"
    assert "Arsenal FC" not in {p.alias for p in accepted}
    assert [p.alias for p in review] == ["Mystery FC"]
    assert existing["openfootball"]["EPL"] == {"Arsenal FC": "Arsenal"}  # input not mutated


def test_to_canonical_maps_and_keeps_raw():
    aliases, _, _ = teams.merge_proposals({}, "openfootball", teams.propose_aliases(REF, OTHER))
    out = teams.to_canonical(OTHER, "openfootball", "EPL", aliases)
    assert out["home_team"].tolist() == REF["home_team"].tolist()
    assert out["home_team_raw"].iloc[0] == "Manchester United FC"


def test_to_canonical_lists_every_unknown_name():
    with pytest.raises(teams.UnmappedTeamError, match="4 openfootball team name"):
        teams.to_canonical(OTHER, "openfootball", "EPL", aliases={})


def test_reference_source_is_identity():
    out = teams.to_canonical(REF, "football_data", "EPL", aliases={})
    assert out["home_team"].equals(REF["home_team"])


def test_committed_alias_file_is_valid():
    aliases = teams.load_aliases()
    for source, leagues in aliases.items():
        for league, table in leagues.items():
            assert table, f"{source}/{league} empty"
            assert all(isinstance(k, str) and isinstance(v, str) for k, v in table.items())
