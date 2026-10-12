import pandas as pd # type: ignore
import numpy as np # type: ignore
from collections import deque

MINYEAR = 2000
MAXYEAR = 2026

FORM_LENGTH = 5

dfs = [
    pd.read_csv(f"data/{year}.csv")
    for year in range(MINYEAR, MAXYEAR + 1)
]
df = pd.concat(dfs, ignore_index=True)

print(df.shape)
print(df.columns)

df = df.drop(columns=['tourney_name', 'winner_seed', 'winner_name', 'winner_ioc','loser_seed', 
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


player_dict = {}
surface_dict = {}

df["p1_form"] = None
df["p2_form"] = None
df["p1_surface_form"] = None
df["p2_surface_form"] = None
for index, row in df.iterrows():
    p1 = row["p1_id"]
    p2 = row["p2_id"]
    winner = row["winner"]
    
    
    #form
    form_history1 = player_dict.setdefault(p1, deque(maxlen=FORM_LENGTH))
    form_history2 = player_dict.setdefault(p2, deque(maxlen=FORM_LENGTH))
    
    df.at[index, "p1_form"] = list(form_history1)
    df.at[index, "p2_form"] = list(form_history2)   
    
    player_dict[p1].append(0 if winner == 0 else 1)
    player_dict[p2].append(1 if winner == 0 else 0)
    
    #surface form
    surface = row["surface"]
        
    surface_form_history1 = surface_dict.setdefault(
        surface, {}
    ).setdefault(p1, deque(maxlen=FORM_LENGTH))

    surface_form_history2 = surface_dict.setdefault(
        surface, {}
    ).setdefault(p2, deque(maxlen=FORM_LENGTH))
    
    df.at[index, "p1_surface_form"] = list(surface_form_history1)
    df.at[index, "p2_surface_form"] = list(surface_form_history2)   
    
    surface_dict[surface][p1].append(0 if winner == 0 else 1)
    surface_dict[surface][p2].append(1 if winner == 0 else 0)
    


print("\n"*5)
print(df.shape)
print(df.columns)
print(df.tail(5))

