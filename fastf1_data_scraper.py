import fastf1
import pandas as pd
import os

# 1. Enable FastF1 cache to speed up repeated data loads.
cache_dir = r"C:\Users\prana\OneDrive\Desktop\Kaggle ML project\F1_variation\cache"
os.makedirs(cache_dir, exist_ok=True)

# 2. Ensure 'data/' folder exists for saving CSVs.
os.makedirs('data', exist_ok=True)

# 3. Define seasons for extraction.
years = [2022, 2023, 2024, 2025]

# 4. Initialize lists to store records.
all_lap_data = []
all_race_results = []

# 5. Loop through each season.
for year in years:
    try:
        # 6. Get that season’s race schedule.
        schedule = fastf1.get_event_schedule(year)

        # 7. Iterate each race in that season.
        for _, race in schedule.iterrows():
            try:
                # 8. Load the race session data (session='R').
                session = fastf1.get_session(year, race['RoundNumber'], 'R')
                session.load()

                # 9. Extract data per driver and per lap.
                for drv in session.drivers:
                    drv_obj = session.get_driver(drv)
                    laps = session.laps.pick_driver(drv_obj['Abbreviation'])

                    # Lap-level telemetry.
                    for _, lap in laps.iterrows():
                        all_lap_data.append({
                            'year': year,
                            'race': race['EventName'],
                            'round': race['RoundNumber'],
                            'driver': drv_obj['Abbreviation'],
                            'team': drv_obj['TeamName'],
                            'grid': drv_obj['GridPosition'],
                            'finish_position': drv_obj['Position'],
                            'points': drv_obj['Points'],
                            'status': drv_obj['Status'],
                            'lap_number': lap['LapNumber'],
                            'lap_time_sec': lap['LapTime'].total_seconds() if pd.notnull(lap['LapTime']) else None,
                            'sector1_time': lap['Sector1Time'].total_seconds() if pd.notnull(lap['Sector1Time']) else None,
                            'sector2_time': lap['Sector2Time'].total_seconds() if pd.notnull(lap['Sector2Time']) else None,
                            'sector3_time': lap['Sector3Time'].total_seconds() if pd.notnull(lap['Sector3Time']) else None,
                            'speed_i1': lap.get('SpeedI1', None),
                            'speed_i2': lap.get('SpeedI2', None),
                            'speed_fl': lap.get('SpeedFL', None),
                            'is_pit_lap': (lap['PitOutTime'] is not pd.NaT) or (lap['PitInTime'] is not pd.NaT)
                        })

                    # Race-level summary.
                    all_race_results.append({
                        'year': year,
                        'race': race['EventName'],
                        'round': race['RoundNumber'],
                        'driver': drv_obj['Abbreviation'],
                        'team': drv_obj['TeamName'],
                        'grid': drv_obj['GridPosition'],
                        'finish_position': drv_obj['Position'],
                        'points': drv_obj['Points'],
                        'status': drv_obj['Status']
                    })

                print(f"✔ Collected data for {race['EventName']} {year}")
            except Exception as e:
                # 10. Skip races that fail to load.
                print(f"⚠ Could not process race {race['EventName']} {year}: {e}")
    except Exception as e:
        # 11. Skip a full season if schedule fetch fails.
        print(f"⚠ Could not retrieve schedule for {year}: {e}")

# 12. Convert lists to DataFrames and save both to CSV.
lap_csv = os.path.join('data', 'f1_lap_telemetry_2022_2025.csv')
pd.DataFrame(all_lap_data).to_csv(lap_csv, index=False)

results_csv = os.path.join('data', 'f1_race_results_2022_2025.csv')
pd.DataFrame(all_race_results).to_csv(results_csv, index=False)

print(f"✅ Lap telemetry saved to    {lap_csv}")
print(f"✅ Race results saved to     {results_csv}")
