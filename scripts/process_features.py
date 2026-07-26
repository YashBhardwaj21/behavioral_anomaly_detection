import os
import numpy as np
import pandas as pd
from collections import defaultdict, deque

# CONFIGURATION & CONSTANTS
EARTH_RADIUS_KM = 6371.0
COLD_START_N = 20
MAX_PLAUSIBLE_KMH = 1000.0
EWMA_ALPHA = 0.05
FAIL_WINDOW_SECONDS = 300
SENSITIVE_WINDOW_SECONDS = 7 * 24 * 3600  # 7-day trailing window for rare-resource access count
AUTH_METHOD_RISK = {"biometric": 0, "certificate": 1, "token": 2, "password": 3}


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized/Scalar Haversine distance calculation in kilometers."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def circular_hour_diff(h1, h2):
    """Smallest distance between two hours on a 24h clock (handles 23h vs 1h wraparound)."""
    d = abs(h1 - h2) % 24
    return min(d, 24.0 - d)


class EntityState:
    """Running, causally-updated state for one entity — the online 'memory'."""
    def __init__(self, entity_type):
        self.entity_type = entity_type
        self.n_events = 0
        self.last_lat = None
        self.last_lon = None
        self.last_ts = None
        self.seen_resources = set()
        self.seen_devices = set()  # Added to catch Device Spoofing!
        self.seen_ips = set()       # Track known source IPs per entity
        self.hour_mean = None
        self.hour_var = 1.0
        self.duration_mean = None  # Added for running session duration Z-score!
        self.duration_var = 1.0
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.last_action = None
        self.fail_times = deque()  # Trailing window of failed-auth timestamps
        self.sensitive_access_times = deque()  # Trailing 7-day window of rare resource accesses


class GlobalPriors:
    """Population/entity-type level fallback stats for cold-start entities."""
    def __init__(self):
        self.resource_freq = defaultdict(int)
        self.total_resource_events = 0
        self.hour_mean_by_type = defaultdict(lambda: 13.0)
        self.hour_var_by_type = defaultdict(lambda: 9.0)
        self.duration_mean_by_type = defaultdict(lambda: 3.4)
        self.duration_var_by_type = defaultdict(lambda: 0.6)
        self.transition_counts_by_type = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    def resource_rarity(self, resource):
        """Global population frequency -> rarity score in [0,1]. Rare-for-everyone."""
        if self.total_resource_events == 0:
            return 0.5
        freq = self.resource_freq.get(resource, 0) / float(self.total_resource_events)
        return float(1.0 - min(freq * 50.0, 1.0))

    def update(self, resource):
        self.resource_freq[resource] += 1
        self.total_resource_events += 1


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Online Causal Feature Extraction Pipeline.
    Iterates chronologically so event (i) only evaluates against state from events < i.
    """
    print("--- [1/2] Sorting logs strictly by global timestamp ---")
    df = df.sort_values("timestamp").reset_index(drop=True).copy()
    ip_fail_times = defaultdict(deque)

    states = {}
    priors = GlobalPriors()
    out_rows = []

    print(f"--- [2/2] Running Causal Feature Extraction on {len(df):,} events ---")
    
    for row in df.itertuples(index=False):
        eid = row.entity_id
        etype = row.entity_type
        if eid not in states:
            states[eid] = EntityState(etype)
        st = states[eid]
        is_cold_start = st.n_events < COLD_START_N

        # 1. --- Geo-Velocity (Impossible Travel) ---
        if st.last_lat is not None:
            dist = haversine_km(st.last_lat, st.last_lon, row.geo_lat, row.geo_lon)
            delta_hours = max((row.timestamp - st.last_ts).total_seconds() / 3600.0, 1.0 / 3600.0)
            geo_velocity = dist / delta_hours
        else:
            geo_velocity = 0.0
        geo_velocity_flag = float(geo_velocity > MAX_PLAUSIBLE_KMH)

        # 2. Resource Novelty & Global Rarity (Lateral Movement)
        if is_cold_start:
            resource_novelty = 0.5 
        else:
            resource_novelty = 0.0 if row.resource_accessed in st.seen_resources else 1.0
        global_rarity = priors.resource_rarity(row.resource_accessed)

        # 3. Device Novelty (Device Spoofing - Crucial Addition!)
        if is_cold_start:
            device_novelty = 0.0
        else:
            device_novelty = 0.0 if row.device_fingerprint in st.seen_devices else 1.0

        # 4. Auth Failure Velocity (Brute Force in trailing 300s window)
        if row.auth_result == "failure":
            st.fail_times.append(row.timestamp)
            ip_fail_times[row.source_ip].append(row.timestamp)
            
        while st.fail_times and (row.timestamp - st.fail_times[0]).total_seconds() > FAIL_WINDOW_SECONDS:
            st.fail_times.popleft()
        while ip_fail_times[row.source_ip] and (row.timestamp - ip_fail_times[row.source_ip][0]).total_seconds() > FAIL_WINDOW_SECONDS:
            ip_fail_times[row.source_ip].popleft()
            
        entity_fail_velocity = len(st.fail_times)
        ip_fail_velocity = len(ip_fail_times[row.source_ip])

        # 4b. Auth Method Risk (ordinal encoding by security strength)
        auth_method_risk = AUTH_METHOD_RISK.get(getattr(row, 'auth_method', 'password'), 3)

        # 4c. IP Novelty (new source IP for this entity)
        if is_cold_start:
            ip_novelty = 0.0
        else:
            ip_novelty = 0.0 if row.source_ip in st.seen_ips else 1.0

        # 4d. Sensitive/Rare Resource Access Count (trailing 7-day window)
        # Prune entries older than 7 days, then count. The aggregated signal
        # is what catches low-and-slow exfiltration — no single event looks
        # unusual, but the cumulative pattern over days does.
        while st.sensitive_access_times and (row.timestamp - st.sensitive_access_times[0]).total_seconds() > SENSITIVE_WINDOW_SECONDS:
            st.sensitive_access_times.popleft()
        sensitive_access_count_7d = len(st.sensitive_access_times)

        # 5. Time-of-Day Probability (EWMA Drift-Tolerant)
        hour = row.timestamp.hour + (row.timestamp.minute / 60.0)
        if is_cold_start or st.hour_mean is None:
            ref_mean = priors.hour_mean_by_type[etype]
            ref_var = priors.hour_var_by_type[etype]
        else:
            ref_mean, ref_var = st.hour_mean, st.hour_var
        d = circular_hour_diff(hour, ref_mean)
        time_of_day_logprob = -0.5 * (d ** 2) / max(ref_var, 0.5)

        # 6. Session Duration Z-Score (Running EWMA Baseline)
        log_dur = np.log1p(row.session_duration)
        if is_cold_start or st.duration_mean is None:
            ref_dur_mean = priors.duration_mean_by_type[etype]
            ref_dur_var = priors.duration_var_by_type[etype]
        else:
            ref_dur_mean, ref_dur_var = st.duration_mean, st.duration_var
        duration_zscore = abs(log_dur - ref_dur_mean) / max(np.sqrt(ref_dur_var), 0.1)

        # 7. Sequence Transition Surprise (Online Markov Bigrams)
        actions = row.command_sequence.split("|") if isinstance(row.command_sequence, str) else []
        seq_score = 0.0
        n_transitions = 0
        prev_action = st.last_action
        trans_source = st.transition_counts if not is_cold_start else priors.transition_counts_by_type[etype]
        
        for a in actions:
            if prev_action is not None:
                counts = trans_source[prev_action]
                total = sum(counts.values())
                # Laplace smoothing
                p = (counts.get(a, 0) + 1.0) / (total + 25.0)
                seq_score += -np.log(p)
                n_transitions += 1
            prev_action = a
        avg_seq_surprise = seq_score / n_transitions if n_transitions > 0 else 0.0

        # APPEND SCORED EVENT
        out_rows.append({
            "event_id": row.event_id,
            "entity_id": eid,
            "entity_type": etype,
            "timestamp": row.timestamp,
            "label": row.label,
            "is_cold_start": int(is_cold_start),
            "geo_velocity_kmh": geo_velocity,
            "geo_velocity_impossible_flag": geo_velocity_flag,
            "resource_novelty": resource_novelty,
            "global_resource_rarity": global_rarity,
            "device_novelty": device_novelty,
            "entity_fail_velocity_5m": entity_fail_velocity,
            "ip_fail_velocity_5m": ip_fail_velocity,
            "time_of_day_logprob": time_of_day_logprob,
            "duration_zscore": duration_zscore,
            "avg_sequence_surprise": avg_seq_surprise,
            "session_duration": row.session_duration,
            "auth_method_risk": auth_method_risk,
            "ip_novelty": ip_novelty,
            "sensitive_access_count_7d": sensitive_access_count_7d,
        })

        st.n_events += 1
        st.last_lat, st.last_lon, st.last_ts = row.geo_lat, row.geo_lon, row.timestamp
        st.seen_resources.add(row.resource_accessed)
        st.seen_devices.add(row.device_fingerprint)
        st.seen_ips.add(row.source_ip)
        priors.update(row.resource_accessed)

        # Update 7-day sensitive access tracker (after feature computation, so
        # the count for this event doesn't include itself — causal)
        if global_rarity > 0.5:
            st.sensitive_access_times.append(row.timestamp)

        # EWMA updates for hour and duration (The Concept Drift Engine)
        if st.hour_mean is None:
            st.hour_mean, st.hour_var = hour, 9.0
            st.duration_mean, st.duration_var = log_dur, 0.6
        else:
            diff_h = circular_hour_diff(hour, st.hour_mean)
            st.hour_mean = (st.hour_mean + EWMA_ALPHA * (hour - st.hour_mean)) % 24.0
            st.hour_var = (1.0 - EWMA_ALPHA) * st.hour_var + EWMA_ALPHA * (diff_h ** 2)
            
            diff_d = log_dur - st.duration_mean
            st.duration_mean = st.duration_mean + EWMA_ALPHA * diff_d
            st.duration_var = (1.0 - EWMA_ALPHA) * st.duration_var + EWMA_ALPHA * (diff_d ** 2)

        prev_action = None
        for a in actions:
            if prev_action is not None:
                st.transition_counts[prev_action][a] += 1
                priors.transition_counts_by_type[etype][prev_action][a] += 1
            prev_action = a
        st.last_action = actions[-1] if actions else st.last_action

    return pd.DataFrame(out_rows)


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    input_path = os.path.join(data_dir, "synthetic_access_logs.parquet")
    output_path = os.path.join(data_dir, "model_ready_logs.parquet")
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find {input_path}. Make sure you ran Phase 1 generator first!")
    else:
        df_raw = pd.read_parquet(input_path)
        df_features = engineer_features(df_raw)
        
        os.makedirs(data_dir, exist_ok=True)
        df_features.to_parquet(output_path, index=False)
        
        print(f"\n--- Phase 2 Complete! Saved {len(df_features):,} rows to {output_path} ---")
        print("\nMean Feature Values by Attack Class (Signal Separation Check):")
        check_cols = [
            "geo_velocity_kmh", "resource_novelty", "global_resource_rarity", 
            "device_novelty", "ip_fail_velocity_5m", "avg_sequence_surprise", "duration_zscore",
            "auth_method_risk", "ip_novelty", "sensitive_access_count_7d",
        ]
        print(df_features.groupby("label")[check_cols].mean().round(3))
