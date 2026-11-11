import pandas as pd

# Let's see what happens if we select the winning rows for Fangio (828), Fangio (784), Moss (780)
results = pd.read_csv('data/results.csv')
drivers = pd.read_csv('data/drivers.csv')
races = pd.read_csv('data/races.csv')
results['grid'] = pd.to_numeric(results['grid'], errors='coerce')
results['positionOrder'] = pd.to_numeric(results['positionOrder'], errors='coerce')

res_clean = results[results['grid'] > 0].copy()

# In current preparation:
# data = data.drop_duplicates(subset=["raceId", "driverId"], keep="first").copy()

# Let's inspect rows of race 828, 784, 780 in res_clean:
for r_id in [828, 784, 780]:
    sub = res_clean[res_clean['raceId'] == r_id].merge(drivers, on='driverId')
    print(f"\n--- RACE {r_id} ALL ROWS WITH GRID > 0 ---")
    print(sub[['resultId', 'driverId', 'surname', 'grid', 'positionOrder', 'points']].to_string())

