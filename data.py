import pandas as pd # type: ignore
import numpy as np # type: ignore
from collections import deque
import time
import sys

START_TIME = time.time()
MINYEAR = 2000
MAXYEAR = 2026

FORM_LENGTH = 5

dfs = [
    pd.read_csv(f"data/{year}.csv")
    for year in range(MINYEAR, MAXYEAR + 1)
]
df = pd.concat(dfs, ignore_index=True)


df["tourney_date"] = pd.to_datetime(
    df["tourney_date"].astype(str),
    format="%Y%m%d"
)

# dates = df["tourney_date"]
# bad = dates.diff().dt.days.lt(0)

# print(df.loc[bad, ["tourney_date", "tourney_id", "match_num"]])
# sys.exit()

df = df.sort_values("tourney_date").reset_index(drop=True)

print(df.shape)
print(df.columns)

df = df.drop(columns=['tourney_id', 'tourney_name', 'winner_seed', 'winner_name', 'winner_ioc','loser_seed', 
        'loser_name','loser_ioc', 'score', 'round', 'minutes', 'w_ace', 'w_df', 'w_svpt', 'w_1stIn',
       'w_1stWon', 'w_2ndWon', 'w_SvGms', 'w_bpSaved', 'w_bpFaced', 'l_ace',
       'l_df', 'l_svpt', 'l_1stIn', 'l_1stWon', 'l_2ndWon', 'l_SvGms',
       'l_bpSaved', 'l_bpFaced', 'loser_rank', 'loser_rank_points', 'winner_rank', 'winner_rank_points'
    ])


print(df.shape)
print(df.columns)




df["winner"] = 1

df = df.rename(columns={
        'winner_id' : 'p1_id', 'winner_entry' : 'p1_entry', 'winner_hand' : 'p1_hand',
        'winner_ht' : 'p1_height', 'winner_age' : 'p1_age',
        'loser_id' : 'p2_id' , 'loser_entry' : 'p2_entry', 'loser_hand': 'p2_hand', 
        'loser_ht': 'p2_height', 'loser_age':'p2_age'   
    })
print(df.columns)

rng = np.random.default_rng(42)
mask = rng.random(len(df)) < 0.5

df.loc[mask, 'winner'] = 0
for i in ['id', 'entry', 'hand', 'height', 'age']:
    df.loc[mask, [f'p1_{i}', f'p2_{i}']] = (
        df.loc[mask, [f'p2_{i}', f'p1_{i}']].to_numpy()
    )
    
    
    
print(df.shape)
print(df.columns)
print(df.head(5))



K_FACTOR = 20
INITIAL_ELO = 1000

class Player:
    def __init__(self, career_start):
        self._matches = {"oa": 0}
        self._wins = {"oa": 0}
        self._elo = {"oa": INITIAL_ELO}
        self._form = {"oa": deque(maxlen=FORM_LENGTH)}
        self._start = {"oa": career_start}

        # self._dob = dob
        # self._hand = hand
        # self._height = height
        
    def graphQL(self, func, surface):
        methods = {
            "elo": self.elo,
            "form": self.form,
            "wins": self.wins,
            "matches": self.matches,
            "winrate": self.winrate,
            "start": self.start,
        }
        return methods[func](surface)
            

    def _init_surface(self, surface, date):
        if surface not in self._matches:
            self._matches[surface] = 0
            self._wins[surface] = 0
            self._elo[surface] = INITIAL_ELO
            self._form[surface] = deque(maxlen=FORM_LENGTH)
            self._start[surface] = date
            

    def expected_score(self, elo_diff):
        return 1 / (1 + 10 ** (elo_diff / 400))

    def update(self, isWin: bool, surface, date, eloDiff=0, surfaceEloDiff=0):
        self._init_surface(surface, date)

        score = 1 if isWin else 0

        expected = self.expected_score(eloDiff)
        surface_expected = self.expected_score(surfaceEloDiff)

        self._elo["oa"] += K_FACTOR * (score - expected)
        self._elo[surface] += K_FACTOR * (score - surface_expected)

        self._matches["oa"] += 1
        self._matches[surface] += 1

        self._wins["oa"] += score
        self._wins[surface] += score

        self._form["oa"].append(score)
        self._form[surface].append(score)

    def wins(self, surface="oa"):
        return self._wins.get(surface, 0)

    def matches(self, surface="oa"):
        return self._matches.get(surface, 0)

    def winrate(self, surface="oa"):
        matches = self.matches(surface)
        return self.wins(surface) / matches if matches else float("nan")

    def form(self, surface="oa"):
        return list(self._form.get(surface, []))

    def elo(self, surface="oa"):
        return self._elo.get(surface, INITIAL_ELO)
    
    def start(self, surface = 'oa'):
        return self._start.get(surface, 0)
     

for prefix in ["p1_", "p2_"]:
    for midfix in ["surface_", ""]:
        for postfix in ["elo", "form", "wins", "matches", "winrate", "exp"]:
            df[f'{prefix}{midfix}{postfix}'] = None
df["elo_diff"] = None
df["surface_elo_diff"] = None
df["height_diff"] = None
df["age_diff"] = None
df["hand_mismatch"] = None



player_dict = {}
for index, row in df.iterrows():
    date = row["tourney_date"]
    p1_id, p2_id = row["p1_id"], row["p2_id"]

    if p1_id not in player_dict:
        player_dict[p1_id] = Player(date)
    if p2_id not in player_dict:
        player_dict[p2_id] = Player(date)

    p1, p2 = player_dict[p1_id], player_dict[p2_id]
    surface = row["surface"]
   
    
    for prefix in ["p1_", "p2_"]:
        for midfix in ["surface_", ""]:
            for postfix in ["elo", "form", "wins", "matches", "winrate", "exp"]:
                if postfix == "exp":
                    start = (p1 if prefix == "p1_" else p2).graphQL("start", (
                        'oa' if midfix == "" else surface
                    ))
                    df.at[index, f"{prefix}{midfix}exp"] = (
                        (date - start).days / 365.25 if start != 0 else 0
                    )
                    continue
                df.at[index, f'{prefix}{midfix}{postfix}'] = (p1 if prefix == "p1_" else p2).graphQL(postfix,('oa' if midfix == "" else surface))
    ed = p1.elo() - p2.elo()
    df.at[index, "elo_diff"] = ed
    sed = p1.elo(surface) - p2.elo(surface)
    df.at[index, "surface_elo_diff"] = sed
    
    df.at[index, "height_diff"] = row["p1_height"]- row["p2_height"]
    df.at[index, "age_diff"] = row["p1_age"]- row["p2_age"]
    df.at[index, "hand_mismatch"] = (
        int(row["p1_hand"] != row["p2_hand"])
        if pd.notna(row["p1_hand"]) and pd.notna(row["p2_hand"])
        else np.nan
    )    
    winner = row["winner"]
    p1.update(winner == 1, surface, date, -ed, -sed)
    p2.update(winner == 0, surface, date, ed, sed)
    
    
    
# Form columns are lists, so exclude them from this conversion.
form_cols = [c for c in df.columns if c.endswith("_form")]

numeric_cols = df.select_dtypes(include="number").columns.tolist()
numeric_cols += [
    c for c in df.columns
    if c not in form_cols
    and c not in numeric_cols
    and c not in ["tourney_date", "p1_id", "p2_id"]
    and c not in ["surface", "tourney_level", "indoor",
                  "p1_entry", "p2_entry", "p1_hand", "p2_hand"]
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="raise")
    

print("\n"*5)
print(df.shape)
print(df.columns)
print(df.tail(5))



#DATA VALIDATION THAT I DIDNT COMPLETELY FUCK UP ANY FEATURE

print("=== dataset ===")
print(f"shape          {df.shape}")
print(f"date range     {df.tourney_date.min().date()} -> {df.tourney_date.max().date()}")
print(f"target balance {df.winner.value_counts(normalize=True).round(4).to_dict()}")

print("\n=== elo ===")
elo_cols = [
    "p1_elo", "p2_elo", "elo_diff",
    "p1_surface_elo", "p2_surface_elo", "surface_elo_diff"
]
print(df[elo_cols].describe().round(1).loc[["mean", "std", "min", "50%", "max"]])

print("\n=== missing values ===")
print(df.isna().sum().loc[lambda s: s > 0].sort_values(ascending=False))

print("\n=== invalid values ===")
count_cols = [
    "p1_matches", "p2_matches",
    "p1_surface_matches", "p2_surface_matches"
]
exp_cols = ["p1_exp", "p2_exp", "p1_surface_exp", "p2_surface_exp"]

print(f"negative match counts  {(df[count_cols] < 0).sum().sum()}")
print(f"negative experience    {(df[exp_cols] < 0).sum().sum()}")
print(f"elo difference errors  {(df.elo_diff - (df.p1_elo - df.p2_elo)).abs().max()}")
print(f"surface elo errors     {(df.surface_elo_diff - (df.p1_surface_elo - df.p2_surface_elo)).abs().max()}")

print("\n=== dtypes ===")
print(df.dtypes.value_counts())
print(f"form columns           {[c for c in df.columns if c.endswith('_form')]}")

print(f'done in: {time.time() - START_TIME}')