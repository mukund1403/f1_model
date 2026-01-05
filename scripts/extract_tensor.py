
import fastf1
import pandas as pd
import numpy as np
import os

FEATURE_NAMES = [
    "current_position",
    "gap_to_car_ahead",
    "gap_to_leader",
    "lap_time_delta_to_avg",
    "tyre_compound",
    "tyre_age_laps",
    "pit_status",
    "num_pit_stops",
    "is_on_fresh_air",
]

def extract_race_features(year, race, frequency='10Hz'):
    """
    Extracts race features into a 3D numpy array (TimeSteps, Drivers, Features).
    
    Args:
        year (int): Season year
        race (str): GP name
        frequency (str): Resampling frequency string (e.g., '250ms' for 4Hz) or 'original'.
                         Default '4Hz' implies 250ms.
    """
    cache_dir = 'f1_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    fastf1.Cache.enable_cache(cache_dir)
    
    print(f"Loading {year} {race} Race...")
    session = fastf1.get_session(year, race, 'R')
    session.load(telemetry=True, weather=False, messages=False)
    
    drivers = session.drivers
    # Optional: Filter DNF drivers? 
    # For a tensor, we usually need fixed dimensions. 
    # If a driver crashes, we can fill with NaNs.
    
    # 1. Determine Time Grid
    # We need a common time index for the tensor.
    # We'll use the Session Time.
    # Find start and end of the race (approx).
    t_min = session.laps['LapStartTime'].min()
    t_max = session.laps['Time'].max()
    
    if pd.isna(t_min): t_min = pd.Timedelta(seconds=0)
    
    # Frequency parsing
    if frequency == '4Hz':
        freq = '250ms'
    elif frequency.endswith('Hz'):
        try:
            hz_val = float(frequency.replace('Hz', ''))
            ms_val = int(1000 / hz_val)
            freq = f'{ms_val}ms'
        except ValueError:
            freq = '100ms' # Fallback
    else:
        freq = frequency
        
    # Create master time index
    print(f"Creating time grid from {t_min} to {t_max} with freq {freq}...")
    time_index = pd.date_range(start=t_min + pd.Timestamp(0), end=t_max + pd.Timestamp(0), freq=freq)
    # Convert back to Timedelta relative to session start if needed?
    # FastF1 uses Timedelta for 'Time'. 
    # But date_range needs Timestamp.
    # We can just use a relative timedelta range.
    
    # Simpler:
    total_seconds = (t_max - t_min).total_seconds()
    freq_seconds = 0.25 # 4Hz
    if frequency != '4Hz':
        # naive parsing
        pass 
        
    num_steps = int(total_seconds / freq_seconds)
    time_grid_sec = np.linspace(t_min.total_seconds(), t_max.total_seconds(), num_steps)
    
    # Initialize Tensor: (Time, Drivers, Features)
    n_time = len(time_grid_sec)
    n_drivers = len(drivers)
    n_feat = len(FEATURE_NAMES)
    
    tensor = np.full((n_time, n_drivers, n_feat), np.nan, dtype=np.float32)
    
    # Pre-computations
    # 1. Avg Lap Time
    good_laps = session.laps.pick_quicklaps()
    avg_lap_time = good_laps['LapTime'].mean().total_seconds()
    
    # 2. Leader Times per Lap (for GapToLeader)
    # Map LapNumber -> Leader's SessionTime at Lap End
    leader_laps = session.laps[session.laps['Position'] == 1.0]
    leader_times_map = leader_laps.set_index('LapNumber')['Time']
    
    # 3. Pit Stops
    pit_data = {}
    for drv in drivers:
        d_laps = session.laps.pick_driver(drv)
        pit_data[drv] = d_laps['PitInTime'].notna().cumsum().shift().fillna(0)
    
    print(f"Processing {n_drivers} drivers...")
    
    for d_idx, drv in enumerate(drivers):
        print(f"  Driver {drv} ({d_idx+1}/{n_drivers})...")
        laps = session.laps.pick_driver(drv)
        if len(laps) == 0: continue
        
        # Get Telemetry
        try:
            # Use pad to ensure we have data at start/end
            tel = laps.get_telemetry()
        except:
            continue
            
        # Clean Telemetry
        # Forward fill and backfill to handle start/end gaps
        tel = tel.ffill().bfill()
        
        # Resample Telemetry to Master Grid
        tel['TimeSec'] = tel['Time'].dt.total_seconds()
        
        # Common Time Grid
        x_target = time_grid_sec
        x_source = tel['TimeSec'].values
        
        # Handle case where x_source has duplicates or non-monotonic (rare but safe to sort)
        # FastF1 telemetry should be sorted.
        
        # Feature 0: Current Position
        # Moved to Lap Loop (since high-freq position is not in telemetry)
        tensor[:, d_idx, 0] = np.nan # Initialize to NaN, fill in loop
        
        # Feature 1: Gap to Car Ahead
        if 'DistanceToDriverAhead' in tel and 'Speed' in tel:
             dist = tel['DistanceToDriverAhead']
             speed_ms = tel['Speed'] / 3.6
             speed_ms = speed_ms.replace(0, 1.0) 
             
             # Calculate time gap
             time_gap = dist / speed_ms
             
             if time_gap.isna().any():
                 raise ValueError(f"Driver {drv} has missing GapToCarAhead data (NaNs found). No defaults allowed.")
             
             tensor[:, d_idx, 1] = np.interp(x_target, x_source, time_gap.values)
        else:
             raise ValueError(f"Missing telemetry data (DistanceToDriverAhead/Speed) for driver {drv}")
             
        # Feature 2: Gap to Leader
        # Strategy: Calculate Gap at end of each lap, then interpolate between laps.
        # Initial gap (Lap 0) = 0 usually (standing start) or grid pos based.
        # Simplification: Linear interp of (LapEndGap) over time.
        
        # Create a timeline of Gaps for this driver
        # DO NOT force 0.0 at t_min. Let the interpolation/extrapolation handle it 
        # or backfill from Lap 1 gap.
        gap_points_x = []
        gap_points_y = []
        
        for _, lap in laps.iterrows():
            ln = lap['LapNumber']
            if pd.isna(lap['Time']): continue
            
            my_time = lap['Time']
            leader_time = leader_times_map.get(ln, my_time) 
            
            gap = (my_time - leader_time).total_seconds()
            
            gap_points_x.append(my_time.total_seconds())
            gap_points_y.append(gap)
            
        # Interpolate
        # If no laps (Crash L1), raise error
        if not gap_points_x:
            raise ValueError(f"No lap data found for driver {drv} to calculate GapToLeader")
        else:
            # Use np.interp. 
            # Note: np.interp constant-extrapolates by default (uses first/last value).
            # This is PERFECT for t=0. It will backfill the gap from Lap 1 line.
            tensor[:, d_idx, 2] = np.interp(x_target, gap_points_x, gap_points_y)

        # Features 3-7 (Lap based)
        # Features 3-7 (and 0) - Lap based
        for _, lap in laps.iterrows():
            if pd.isna(lap['LapStartTime']) or pd.isna(lap['LapTime']): continue
            
            t_start = lap['LapStartTime'].total_seconds()
            t_end = t_start + lap['LapTime'].total_seconds()
            
            mask = (time_grid_sec >= t_start) & (time_grid_sec < t_end)
            if not np.any(mask): continue
            
            # 0. Position
            tensor[mask, d_idx, 0] = lap['Position']
            
            # 3. Delta
            delta = lap['LapTime'].total_seconds() - avg_lap_time
            tensor[mask, d_idx, 3] = delta
            
            # 4. Compound
            comp = lap['Compound']
            comp_val = 0
            if comp == 'SOFT': comp_val = 1
            elif comp == 'MEDIUM': comp_val = 2
            elif comp == 'HARD': comp_val = 3
            elif comp == 'INTERMEDIATE': comp_val = 4
            elif comp == 'WET': comp_val = 5
            tensor[mask, d_idx, 4] = comp_val
            
            # 5. Tyre Age
            tensor[mask, d_idx, 5] = lap['TyreLife']
            
            # 6. Pit Status
            in_pit = 1.0 if (pd.notna(lap['PitInTime']) or pd.notna(lap['PitOutTime'])) else 0.0
            tensor[mask, d_idx, 6] = in_pit
            
            # 7. Num Pit Stops
            tensor[mask, d_idx, 7] = pit_data[drv].get(lap['LapNumber'], 0)
        
        # Feature 8: Fresh Air
        # From feature 1
        gaps = tensor[:, d_idx, 1]
        fresh_air = np.where((gaps > 2.0), 1.0, 0.0)
        tensor[:, d_idx, 8] = fresh_air
        
    # Final cleanup
    tensor = np.nan_to_num(tensor, nan=0.0)
        
    print(f"Tensor shape: {tensor.shape}")
    return tensor, drivers, FEATURE_NAMES

if __name__ == "__main__":
    t, d, f = extract_race_features(2024, 'Silverstone', '10Hz')
    print("Function execution complete.")
    
    np.set_printoptions(suppress=True, precision=3, linewidth=200)
    
    print("\n--- First Timestep (t=0) ---")
    print(f"Shape: {t[0].shape} (Drivers x Features)")
    print(f"Columns: {f}")
    print(t[0])
