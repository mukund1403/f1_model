
import fastf1
import pandas as pd
import os
import argparse

def ensure_dirs():
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/cache', exist_ok=True)

def collect_data(years):
    # Enable cache for speed on subsequent runs
    fastf1.Cache.enable_cache('data/cache')
    
    all_results = []
    
    for year in years:
        print(f"Processing Season {year}...")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
            
            for i, row in schedule.iterrows():
                round_num = row['RoundNumber']
                
                # Fetch the race session
                try:
                    session = fastf1.get_session(year, round_num, 'R')
                    session.load(laps=False, telemetry=False, weather=False, messages=False)
                    
                    race_name = session.event.EventName
                    # FastF1 event object usually has Location, but specific Circuit name might be tricky directly.
                    # We will use Location as Circuit Name proxy for now, or check if 'Circuit' key exists in event
                    circuit_name = session.event.Location 
                    
                    results = session.results
                    
                    # session.results is a DataFrame with drivers as index or columns? 
                    # Usually it's a DF where each row is a driver.
                    
                    for driver_code, driver_result in results.iterrows():
                        # drivers are usually the index if loaded this way, or we reset index?
                        # session.results is indexed by DriverNumber usually, but let's check columns
                        # Actually in recent FastF1, it returns a DataFrame-like object.
                        
                        # Let's handle the row properly.
                        # Important columns in results: 'Abbreviation', 'TeamName'
                        
                        driver_code_val = driver_result.get('Abbreviation', 'UNKNOWN')
                        constructor_val = driver_result.get('TeamName', 'UNKNOWN')
                        
                        all_results.append({
                            'season': year,
                            'round': round_num,
                            'race_name': race_name,
                            'circuit_name': circuit_name,
                            'driver': driver_code_val,
                            'constructor': constructor_val
                        })
                        
                except Exception as e:
                    print(f"  Error processing Round {round_num}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error processing Year {year}: {e}")

    # Save to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_path = 'data/raw/f1_historical_data.csv'
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        print(df.head())
    else:
        print("No data collected.")

if __name__ == "__main__":
    ensure_dirs()
    
    parser = argparse.ArgumentParser(description="Collect F1 data.")
    parser.add_argument('--years', type=int, nargs='+', help="Years to collect data for (e.g. 2024 or 2021 2022)")
    args = parser.parse_args()
    
    if args.years:
        years_to_collect = args.years
    else:
        # Default to 2024-2025 if no arguments provided
        years_to_collect = [2024, 2025]
        
    collect_data(years_to_collect)
