"""Canned node modules for case 016, served by the fake model server.

`teams` is broken in its first version so that an offline run exercises the
repair loop; the repaired version is served when the request carries feedback.
"""

NODES = {
    "validate": '''import pandas as pd


def run(inputs):
    df = pd.read_csv(inputs["input:data.csv"])
    if df["player_id"].isna().any() or df["player_id"].duplicated().any():
        raise ValueError("invalid player_id")
    return {"players_clean.csv": df.sort_values("player_id").reset_index(drop=True)}
''',
    "performance": '''def run(inputs):
    df = inputs["players_clean.csv"][["player_id", "points_per_game", "assists_per_game",
                                      "rebounds_per_game", "minutes_per_game", "efficiency_rating",
                                      "usage_rate"]].copy()
    df["production_index"] = df.points_per_game + df.assists_per_game + df.rebounds_per_game
    return {"performance_features.csv": df}
''',
    "context_features": '''def run(inputs):
    df = inputs["players_clean.csv"][["player_id", "team", "age", "games_played",
                                      "injury_games_missed", "all_star"]].copy()
    df["availability_rate"] = (df.games_played - df.injury_games_missed).clip(lower=0) / df.games_played
    return {"context_features.csv": df}
''',
    "integrate": '''def run(inputs):
    merged = inputs["performance_features.csv"].merge(
        inputs["context_features.csv"], on="player_id", validate="one_to_one")
    return {"model_features.csv": merged}
''',
    "cluster": '''from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

FEATURES = ["points_per_game", "assists_per_game", "rebounds_per_game", "minutes_per_game",
            "efficiency_rating", "usage_rate", "production_index", "availability_rate", "age"]


def run(inputs):
    df = inputs["model_features.csv"]
    labels = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(
        StandardScaler().fit_transform(df[FEATURES]))
    out = df[["player_id", "team", "all_star", "production_index"]].copy()
    out["cluster"] = labels
    return {"cluster_assignments.csv": out}
''',
    "profiles": '''def run(inputs):
    df = inputs["cluster_assignments.csv"]
    return {"cluster_profiles.csv": df.groupby("cluster", as_index=False)["production_index"].mean()}
''',
    "teams": '''def run(inputs):
    df = inputs["cluster_assignments.csv"]
    return {"team_cluster_mix.csv": df.groupby(["team", "cluster"], as_index=False).size_typo()}
''',
    "allstars": '''def run(inputs):
    df = inputs["cluster_assignments.csv"]
    return {"cluster_allstar_rates.csv": df.groupby("cluster", as_index=False)["all_star"].mean()}
''',
    "report": '''def run(inputs):
    profiles = inputs["cluster_profiles.csv"]
    teams = inputs["team_cluster_mix.csv"]
    allstars = inputs["cluster_allstar_rates.csv"]
    return {"cluster_report.json": {"clusters": int(len(profiles)),
                                    "teams": int(teams["team"].nunique()),
                                    "max_allstar_rate": float(allstars["all_star"].max())}}
''',
}

REPAIRED = {
    "teams": NODES["teams"].replace("size_typo()", "size()"),
}
