Here is the implementation.

```python
# file: definitions.py
import json
import os
from pathlib import Path

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

IN = Path(os.environ.get("SERVICECRAFT_INPUT_DIR", "inputs"))
OUT = Path(os.environ.get("SERVICECRAFT_OUTPUT_DIR", "outputs"))
FEATURES = ["points_per_game", "assists_per_game", "rebounds_per_game", "minutes_per_game",
            "efficiency_rating", "usage_rate", "production_index", "availability_rate", "age"]


def save(df, name):
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
    return df


@dg.asset
def validate():
    df = pd.read_csv(IN / "data.csv")
    if df["player_id"].isna().any() or df["player_id"].duplicated().any():
        raise ValueError("invalid player_id")
    return save(df.sort_values("player_id").reset_index(drop=True), "players_clean.csv")


@dg.asset
def performance(validate):
    df = validate[["player_id", "points_per_game", "assists_per_game", "rebounds_per_game",
                   "minutes_per_game", "efficiency_rating", "usage_rate"]].copy()
    df["production_index"] = df.points_per_game + df.assists_per_game + df.rebounds_per_game
    return save(df, "performance_features.csv")


@dg.asset
def context_features(validate):
    df = validate[["player_id", "team", "age", "games_played", "injury_games_missed", "all_star"]].copy()
    df["availability_rate"] = (df.games_played - df.injury_games_missed).clip(lower=0) / df.games_played
    return save(df, "context_features.csv")


@dg.asset
def integrate(performance, context_features):
    return save(performance.merge(context_features, on="player_id", validate="one_to_one"), "model_features.csv")


@dg.asset
def cluster(integrate):
    x = StandardScaler().fit_transform(integrate[FEATURES])
    labels = KMeans(n_clusters=4, random_state=42, n_init=20).fit_predict(x)
    out = integrate[["player_id", "team", "all_star", "production_index"]].copy()
    out["cluster"] = labels
    return save(out, "cluster_assignments.csv")


@dg.asset
def profiles(cluster):
    return save(cluster.groupby("cluster", as_index=False)["production_index"].mean(), "cluster_profiles.csv")


@dg.asset
def teams(cluster):
    return save(cluster.groupby(["team", "cluster"], as_index=False).size(), "team_cluster_mix.csv")


@dg.asset
def allstars(cluster):
    return save(cluster.groupby("cluster", as_index=False)["all_star"].mean(), "cluster_allstar_rates.csv")


@dg.asset
def report(profiles, teams, allstars):
    data = {"clusters": int(len(profiles)), "teams": int(teams["team"].nunique()),
            "max_allstar_rate": float(np.nan_to_num(allstars["all_star"].max()))}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cluster_report.json").write_text(json.dumps(data))
    return data


defs = dg.Definitions(assets=[validate, performance, context_features, integrate, cluster,
                              profiles, teams, allstars, report])
```
