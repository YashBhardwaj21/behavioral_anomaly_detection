"""
Synthetic behavioral access-log generator for the anomaly detection PoC.

Design principles (why the data is built this way):
- Real city geography (not random lat/lon) so distances/travel-times are meaningful.
- Population concentrated near a few "HQ" cities + a remote-worker tail, like a real org.
- Skewed (Zipf/lognormal) distributions for resource access and session duration,
  because real access logs are never uniform.
- Per-entity behavioral fingerprints (home geo, device, work-hour center, resource subset)
  so "normal" is genuinely learnable, and attacks are genuinely a deviation from it.
"""

import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta
import random
import uuid

fake = Faker()
rng = np.random.default_rng(42)
random.seed(42)

# (city, lat, lon, is_hq) - a mix of corporate hubs + globally distributed remote cities
CITIES = [
    ("New York", 40.7128, -74.0060, True),
    ("London", 51.5074, -0.1278, True),
    ("Bengaluru", 12.9716, 77.5946, True),
    ("Chicago", 41.8781, -87.6298, False),
    ("Toronto", 43.6532, -79.3832, False),
    ("Berlin", 52.5200, 13.4050, False),
    ("Sydney", -33.8688, 151.2093, False),
    ("Tokyo", 35.6762, 139.6503, False),
    ("Sao Paulo", -23.5505, -46.6333, False),
    ("Singapore", 1.3521, 103.8198, False),
    ("Mumbai", 19.0760, 72.8777, False),
    ("Dublin", 53.3498, -6.2603, False),
    ("Cairo", 30.0444, 31.2357, False),
    ("Mexico City", 19.4326, -99.1332, False),
    ("Warsaw", 52.2297, 21.0122, False),
    ("Cape Town", -33.9249, 18.4241, False),
    ("Seoul", 37.5665, 126.9780, False),
    ("Denver", 39.7392, -104.9903, False),
    ("Manila", 14.5995, 120.9842, False),
    ("Moscow", 55.7558, 37.6173, False),
]

# Resource pool with realistic access-frequency tiers (Zipf-like: common vs rare)
COMMON_RESOURCES = [
    "/email/inbox", "/hr/portal", "/wiki/home", "/chat/general", "/calendar",
    "/docs/shared-drive", "/crm/dashboard", "/timesheet/entry", "/vpn/status",
    "/intranet/news",
]
DEPT_RESOURCES = [
    "/repo/frontend", "/repo/backend", "/ci/pipeline", "/jira/board",
    "/analytics/dashboard", "/erp/finance", "/erp/inventory", "/support/tickets",
    "/design/figma-proxy", "/qa/testrail",
]
SENSITIVE_RESOURCES = [
    "/admin/db", "/vault/keys", "/admin/user-mgmt", "/finance/payroll-export",
    "/admin/audit-logs", "/security/siem-console", "/admin/network-config",
    "/backup/full-export",
]

AUTH_METHODS = ["password", "token", "certificate", "biometric"]
AUTH_WEIGHTS = [0.55, 0.30, 0.10, 0.05]

OS_FINGERPRINTS = ["Windows11-x64", "macOS-14", "Ubuntu-22.04", "iOS-17", "Android-14"]

# Shared campaign IPs for credential stuffing — reused across entities so the
# "few IPs, many targets" signature appears in ip_fail_velocity_5m
CREDENTIAL_STUFFING_IPS = [fake.ipv4_public() for _ in range(2)]


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def zipf_choice(pool, size, alpha=1.3):
    """Pick indices from a pool following a Zipf-like popularity skew."""
    ranks = np.arange(1, len(pool) + 1)
    weights = 1 / (ranks ** alpha)
    weights /= weights.sum()
    idx = rng.choice(len(pool), size=size, p=weights)
    return [pool[i] for i in idx]


# 2. ENTITY PROFILES

class EntityProfile:
    """Holds the stable behavioral fingerprint for one entity."""

    def __init__(self, entity_id, entity_type):
        self.entity_id = entity_id
        self.entity_type = entity_type

        # 70% of entities cluster near HQ cities, 30% are globally distributed remote workers
        if rng.random() < 0.7:
            hq_cities = [c for c in CITIES if c[3]]
            city = hq_cities[rng.integers(0, len(hq_cities))]
        else:
            city = CITIES[rng.integers(0, len(CITIES))]
        self.home_city, self.home_lat, self.home_lon = city[0], city[1], city[2]
        # small jitter so not everyone is at the exact same coordinate (different office / home)
        self.home_lat += rng.normal(0, 0.05)
        self.home_lon += rng.normal(0, 0.05)

        # work-hour center: most entities 9-5 local, ~10% night-shift/ops roles
        self.work_hour_center = 13 if rng.random() > 0.10 else 2
        self.work_hour_std = rng.uniform(2.0, 3.5)

        self.primary_device = rng.choice(OS_FINGERPRINTS)
        # ~20% of entities also legitimately use a secondary device (phone + laptop)
        self.secondary_device = rng.choice(OS_FINGERPRINTS) if rng.random() < 0.2 else None

        # a small stable pool of known IPs (home / office / VPN gateway), not a fresh
        # random IP per event - real entities reuse the same handful of source IPs for
        # days or weeks, and this is what makes "IP novelty" a usable detection feature
        self.known_ips = [fake.ipv4_public() for _ in range(int(rng.integers(1, 3)))]

        # each entity's typical resource subset: mostly common resources + a few dept ones
        n_common = rng.integers(3, 6)
        n_dept = rng.integers(2, 5)
        self.typical_resources = (
            list(rng.choice(COMMON_RESOURCES, size=n_common, replace=False))
            + list(rng.choice(DEPT_RESOURCES, size=n_dept, replace=False))
        )
        # service accounts / edge devices almost never touch sensitive resources normally
        self.sensitive_access_prob = 0.0

        # simple per-entity action bigram preferences (for realistic command_sequence)
        self.action_order = list(rng.permutation(self.typical_resources))

        # concept drift: ~15% of entities permanently shift behavior partway through the
        # timeline (new laptop, new team/project) - this is what lets the report demonstrate
        # drift tolerance rather than just claiming it
        self.experiences_drift = rng.random() < 0.15
        self.drift_day = int(rng.integers(35, 60)) if self.experiences_drift else None
        if self.experiences_drift:
            other_devices = [d for d in OS_FINGERPRINTS if d != self.primary_device]
            self.post_drift_device = rng.choice(other_devices)
            n_new_dept = rng.integers(2, 5)
            self.post_drift_resources = (
                list(rng.choice(COMMON_RESOURCES, size=n_common, replace=False))
                + list(rng.choice(DEPT_RESOURCES, size=n_new_dept, replace=False))
            )
            # stable post-drift ordering, established once - not regenerated per event
            self.post_drift_action_order = list(rng.permutation(self.post_drift_resources))


def build_entity_population(n_users=250, n_service=60, n_edge=40):
    entities = []
    for _ in range(n_users):
        entities.append(EntityProfile(f"user_{fake.user_name()}_{uuid.uuid4().hex[:4]}", "user"))
    for _ in range(n_service):
        entities.append(EntityProfile(f"svc_{fake.word()}_{uuid.uuid4().hex[:4]}", "service_account"))
    for _ in range(n_edge):
        entities.append(EntityProfile(f"edge_{fake.word()}_{uuid.uuid4().hex[:4]}", "edge_device"))
    return entities

# 3. NORMAL BASELINE EVENT GENERATION

def sample_login_hour(profile):
    hour = rng.normal(profile.work_hour_center, profile.work_hour_std)
    # 5% chance of legitimate off-hours activity, so off-hours isn't a perfect tell
    if rng.random() < 0.05:
        hour = rng.uniform(0, 24)
    # wrap around the 24h clock rather than clipping, so e.g. a night-shift entity's
    # samples that dip below 0 land at 22-23h instead of piling up at exactly 0:00:00
    return float(hour % 24)


def generate_normal_events(profile, n_events, start_date, day_span=150):
    events = []
    for _ in range(n_events):
        day_offset = int(rng.integers(0, day_span))
        hour = sample_login_hour(profile)
        ts = start_date + timedelta(days=day_offset, hours=hour)

        # apply concept drift: after drift_day, the entity's device and resource pool
        # permanently shift (new laptop / new team) - this is what a drift-tolerant
        # model must learn to accept rather than flag forever
        post_drift = profile.experiences_drift and day_offset >= profile.drift_day
        active_device = profile.post_drift_device if post_drift else profile.primary_device
        active_resources = profile.post_drift_resources if post_drift else profile.typical_resources
        # use the STABLE per-entity order established once in __init__, not a fresh
        # permutation every event - a freshly reshuffled order would be pointless anyway
        # since it's about to be sampled from, but a stable order lets us preserve real
        # sequence structure below instead of just picking a random unordered subset
        active_action_order = profile.post_drift_action_order if post_drift else profile.action_order

        device = active_device
        if profile.secondary_device and rng.random() < 0.15:
            device = profile.secondary_device

        resource = zipf_choice(active_resources, 1)[0]
        session_duration = float(rng.lognormal(mean=3.4, sigma=0.6))  # minutes, right-skewed

        auth_method = rng.choice(AUTH_METHODS, p=AUTH_WEIGHTS)

        # order-preserving command sequence: take a contiguous slice of this entity's
        # stable habitual order (e.g. always checks email -> dashboard -> repo in that
        # sequence), rather than an unordered random subset. A random subset shuffled by
        # rng.choice(replace=False) has zero learnable sequence structure regardless of
        # whether the source list is stable - this contiguous-window approach is what
        # actually gives an LSTM/GRU autoencoder a real "normal" pattern to learn.
        seq_len = min(int(rng.integers(2, 6)), len(active_action_order))
        start = int(rng.integers(0, len(active_action_order)))
        command_sequence = [active_action_order[(start + i) % len(active_action_order)] for i in range(seq_len)]

        events.append({
            "event_id": uuid.uuid4().hex,
            "entity_id": profile.entity_id,
            "entity_type": profile.entity_type,
            "timestamp": ts,
            "source_ip": rng.choice(profile.known_ips),
            "geo_lat": profile.home_lat,
            "geo_lon": profile.home_lon,
            "geo_city": profile.home_city,
            "resource_accessed": resource,
            "auth_method": auth_method,
            "auth_result": "success",
            "session_duration": session_duration,
            "command_sequence": "|".join(command_sequence),
            "device_fingerprint": device,
            "label": "normal",
        })
    return events


# 4. ATTACK INJECTORS

def inject_impossible_travel(profile, base_events, n_cases=1):
    injected = []
    far_cities = [c for c in CITIES if c[0] != profile.home_city]
    for _ in range(n_cases):
        base_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        far_city = far_cities[rng.integers(0, len(far_cities))]
        dist = haversine_km(profile.home_lat, profile.home_lon, far_city[1], far_city[2])
        delta_minutes = rng.uniform(15, 45)  # short enough that implied speed is impossible
        second_ts = base_ts + timedelta(minutes=delta_minutes)

        # base_ts already belongs to an existing normal event in all_events - don't
        # duplicate it, only inject the anomalous far-away jump
        injected.append({
            **_normal_event_template(profile, second_ts),
            "source_ip": fake.ipv4_public(),
            "geo_lat": far_city[1], "geo_lon": far_city[2], "geo_city": far_city[0],
            "label": "impossible_travel",
        })
    return injected


def inject_brute_force(profile, base_events, n_cases=1):
    injected = []
    for _ in range(n_cases):
        attacker_ip = fake.ipv4_public()
        anchor_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        # start the attack sometime after the anchor event, not exactly at it - two
        # events can't legitimately share the same (entity_id, timestamp)
        base_ts = anchor_ts + timedelta(seconds=float(rng.uniform(30, 180)))
        n_attempts = rng.integers(15, 30)
        elapsed = 0.0
        for i in range(n_attempts):
            elapsed += float(rng.uniform(2, 8))  # cumulative gap - guarantees strictly increasing timestamps
            ts = base_ts + timedelta(seconds=elapsed)
            success = (i == n_attempts - 1) and rng.random() < 0.4  # sometimes ends in a breach
            event = {
                **_normal_event_template(profile, ts),
                "source_ip": attacker_ip,
                "auth_result": "success" if success else "failure",
                "label": "brute_force",
            }
            if not success:
                # a failed auth terminates at the gateway almost instantly - short random
                # duration (seconds, not the ~30min lognormal used for real sessions), but
                # NOT a hardcoded 0.0, which would just create a new artificial spike
                event["session_duration"] = float(rng.uniform(0.02, 0.3))  # ~1-18 seconds
            injected.append(event)
    return injected


def inject_lateral_movement(profile, base_events, n_cases=1):
    injected = []
    for _ in range(n_cases):
        anchor_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        base_ts = anchor_ts + timedelta(minutes=float(rng.uniform(2, 10)))
        touched = list(rng.choice(SENSITIVE_RESOURCES, size=min(4, len(SENSITIVE_RESOURCES)), replace=False))
        for i, res in enumerate(touched):
            ts = base_ts + timedelta(minutes=int(i * rng.uniform(3, 10)))
            injected.append({
                **_normal_event_template(profile, ts),
                "resource_accessed": res,
                "command_sequence": "|".join(touched),
                "label": "lateral_movement",
            })
    return injected


def inject_device_spoofing(profile, base_events, n_cases=1):
    injected = []
    # exclude every device that legitimately belongs to this entity - not just primary -
    # so we never accidentally label a real secondary/post-drift device as "spoofed"
    known_devices = {profile.primary_device}
    if profile.secondary_device:
        known_devices.add(profile.secondary_device)
    if profile.experiences_drift:
        known_devices.add(profile.post_drift_device)
    available_spoofs = [d for d in OS_FINGERPRINTS if d not in known_devices]
    if not available_spoofs:
        available_spoofs = ["Unrecognized-Device"]
    spoof_device = rng.choice(available_spoofs)

    for _ in range(n_cases):
        anchor_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        base_ts = anchor_ts + timedelta(minutes=float(rng.uniform(1, 60)))
        injected.append({
            **_normal_event_template(profile, base_ts),
            "device_fingerprint": spoof_device,
            "label": "device_spoof",
        })
    return injected


def inject_credential_stuffing(profile, base_events, n_cases=1):
    """Credential stuffing: a shared campaign IP sprays login attempts against
    this entity. Distinguished from brute force by the shared CREDENTIAL_STUFFING_IPS
    that appear across many different entities — the 'few IPs, many targets' signature."""
    injected = []
    for _ in range(n_cases):
        attacker_ip = rng.choice(CREDENTIAL_STUFFING_IPS)
        anchor_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        base_ts = anchor_ts + timedelta(seconds=float(rng.uniform(30, 300)))
        n_attempts = int(rng.integers(2, 5))  # fewer per-entity than brute force
        elapsed = 0.0
        for i in range(n_attempts):
            elapsed += float(rng.uniform(3, 15))
            ts = base_ts + timedelta(seconds=elapsed)
            success = (i == n_attempts - 1) and rng.random() < 0.15
            event = {
                **_normal_event_template(profile, ts),
                "source_ip": attacker_ip,
                "auth_result": "success" if success else "failure",
                "label": "credential_stuffing",
            }
            if not success:
                event["session_duration"] = float(rng.uniform(0.02, 0.2))
            injected.append(event)
    return injected


def inject_low_and_slow_exfiltration(profile, base_events, n_cases=1):
    """Low-and-slow exfiltration: an insider or compromised credential gradually
    accesses sensitive/rare resources over days, during off-hours, with short
    sessions. Uses the entity's own device and IP to avoid triggering
    device/IP novelty — the signal must come from the aggregated sensitive_access_count_7d
    feature, not from any single event in isolation."""
    injected = []
    for _ in range(n_cases):
        anchor_ts = base_events[rng.integers(0, len(base_events))]["timestamp"]
        n_days = int(rng.integers(5, 15))
        targets = list(rng.choice(
            SENSITIVE_RESOURCES,
            size=min(n_days, len(SENSITIVE_RESOURCES)),
            replace=(len(SENSITIVE_RESOURCES) < n_days),
        ))
        for day_offset, resource in enumerate(targets):
            off_hour = float(rng.uniform(1.0, 5.0))  # 1-5 AM
            ts = anchor_ts + timedelta(days=day_offset, hours=off_hour)
            injected.append({
                **_normal_event_template(profile, ts),
                "resource_accessed": resource,
                "session_duration": float(rng.uniform(0.5, 3.0)),  # quick grabs
                "command_sequence": "|".join(list(rng.choice(
                    SENSITIVE_RESOURCES,
                    size=min(3, len(SENSITIVE_RESOURCES)),
                    replace=False,
                ))),
                "label": "low_and_slow",
            })
    return injected


def _normal_event_template(profile, ts):
    resource = zipf_choice(profile.typical_resources, 1)[0]
    return {
        "event_id": uuid.uuid4().hex,
        "entity_id": profile.entity_id,
        "entity_type": profile.entity_type,
        "timestamp": ts,
        "source_ip": rng.choice(profile.known_ips),
        "geo_lat": profile.home_lat,
        "geo_lon": profile.home_lon,
        "geo_city": profile.home_city,
        "resource_accessed": resource,
        "auth_method": rng.choice(AUTH_METHODS, p=AUTH_WEIGHTS),
        "auth_result": "success",
        "session_duration": float(rng.lognormal(mean=3.4, sigma=0.6)),
        "command_sequence": resource,
        "device_fingerprint": profile.primary_device,
        "label": "normal",
    }


# 5. FULL DATASET ASSEMBLY

def generate_dataset(total_rows=200_000, anomaly_rate=0.02, cold_start_n=20):
    entities = build_entity_population()
    start_date = datetime(2026, 4, 1)

    all_events = []
    events_per_entity = total_rows // len(entities)

    for profile in entities:
        n_events = events_per_entity
        base = generate_normal_events(profile, n_events, start_date)
        all_events.extend(base)

    # inject attacks on a random subset of entities, sized to hit target anomaly_rate
    n_anomalous_entities = max(4, int(len(entities) * 0.65))
    attacked = rng.choice(entities, size=n_anomalous_entities, replace=False)
    injectors = [
        inject_impossible_travel, inject_brute_force, inject_lateral_movement,
        inject_device_spoofing, inject_credential_stuffing, inject_low_and_slow_exfiltration,
    ]

    for i, profile in enumerate(attacked):
        injector = injectors[i % len(injectors)]
        entity_events = [e for e in all_events if e["entity_id"] == profile.entity_id]
        if injector in [inject_impossible_travel, inject_device_spoofing]:
            cases = int(rng.integers(8, 15))
        elif injector in [inject_lateral_movement, inject_credential_stuffing]:
            cases = int(rng.integers(3, 6))
        elif injector == inject_low_and_slow_exfiltration:
            cases = int(rng.integers(2, 4))
        else:  # brute_force
            cases = int(rng.integers(2, 4))
        all_events.extend(injector(profile, entity_events, n_cases=cases))

    for _ in range(cold_start_n):
        new_type = rng.choice(["user", "service_account", "edge_device"])
        new_profile = EntityProfile(f"newentity_{uuid.uuid4().hex[:6]}", new_type)
        all_events.extend(generate_normal_events(new_profile, rng.integers(1, 5), start_date, day_span=3))

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)

    actual_rate = (df["label"] != "normal").mean()
    print(f"Generated {len(df)} rows | anomaly rate: {actual_rate:.2%} | entities: {len(entities)+cold_start_n}")
    return df


if __name__ == "__main__":
    import os
    df = generate_dataset(total_rows=200_000)
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    df.to_parquet(os.path.join(data_dir, "synthetic_access_logs.parquet"), index=False)
    df.to_csv(os.path.join(data_dir, "synthetic_access_logs.csv"), index=False)
    print(df["label"].value_counts())
