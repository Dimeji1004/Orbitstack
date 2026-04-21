"""Generate Orbitstack synthetic data for the NRR diagnostic notebook.

Run from the repo root:

    python data/generate_data.py

Writes into the same directory as this script:
- customers.csv          (this step)
- subscriptions.csv      (added in implementation step 3)
- churn_reasons.csv      (added in implementation step 4)

Reproducibility: a single seed at the top, all random draws made in a fixed
order, deterministic CSV formatting. Two consecutive runs must produce
byte-identical CSVs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

DATA_DIR = Path(__file__).resolve().parent

TOTAL_CUSTOMERS = 450

SEGMENT_COUNTS = {"SMB": 200, "Mid-market": 180, "Enterprise": 70}
COUNTRY_COUNTS = {"UK": 270, "DE": 80, "FR": 50, "NL": 30, "IE": 20}

YEAR_ACQUISITION_TARGETS = {
    2021: 200,
    2022: 90,
    2023: 60,
    2024: 55,
    2025: 45,
}

INDUSTRIES = ["Freight", "Warehousing", "Last-mile", "Fleet management", "3PL"]

CHANNELS = ["Inbound", "Outbound", "Partner", "Referral"]
CHANNEL_WEIGHTS = [0.35, 0.30, 0.20, 0.15]

REPS = [
    "A. Bennett",
    "C. Doyle",
    "E. Farouk",
    "G. Hartley",
    "I. Jamison",
    "K. Lindqvist",
    "M. Nakamura",
    "O. Petrova",
]

ARR_RANGES_GBP = {
    "SMB": (3_000, 15_000),
    "Mid-market": (15_000, 80_000),
    "Enterprise": (80_000, 400_000),
}

NAME_PREFIXES = [
    "Anchor", "Ashford", "Beacon", "Birchwood", "Brightwater", "Brookfield",
    "Caldwell", "Cascade", "Coppergate", "Cornerstone", "Crestline",
    "Ellesmere", "Foxgrove", "Granite", "Greycliff", "Hartwood", "Highland",
    "Holloway", "Inglewood", "Ironclad", "Junction", "Kestrel", "Larkfield",
    "Linden", "Linwood", "Maple", "Marston", "Mossbank", "Nettleton",
    "Northwind", "Oakridge", "Orchard", "Penwick", "Pinegate", "Quayside",
    "Redwood", "Ridgemont", "Riverside", "Saltmarsh", "Sterling",
    "Stonefield", "Sunnymead", "Thornbury", "Tideford", "Upton", "Vesper",
    "Westmoor", "Wexford", "Whitestone", "Wickham", "Yardley", "Zephyr",
]

NAME_MIDDLES = [
    "Logistics", "Freight", "Haulage", "Express", "Routes",
    "Cargo", "Transit", "Distribution", "Carriers", "Lanes",
]

COUNTRY_SUFFIXES = {
    "UK": "Ltd",
    "DE": "GmbH",
    "FR": "SAS",
    "NL": "BV",
    "IE": "Limited",
}

# (quarter_index, first_month, last_month, last_day_of_last_month)
QUARTERS = [
    (1, 1, 3, 31),
    (2, 4, 6, 30),
    (3, 7, 9, 30),
    (4, 10, 12, 31),
]

CUSTOMER_COLUMNS = [
    "customer_id",
    "customer_name",
    "segment",
    "industry_subsector",
    "country",
    "acquisition_date",
    "first_paid_invoice_date",
    "acquisition_channel",
    "acquisition_rep",
    "original_arr_gbp",
]

SUBSCRIPTION_COLUMNS = [
    "customer_id",
    "month_end_date",
    "arr_gbp",
    "seats",
    "product_modules",
    "subscription_status",
    "contract_end_date",
    "list_price_per_seat",
    "effective_price_per_seat",
]

CHURN_REASON_COLUMNS = [
    "customer_id",
    "churn_date",
    "primary_reason",
    "competitor_won_to",
    "csm_notes",
]

OTHER_COMPETITORS = [
    "RouteForce",
    "TransitOne",
    "CargoHub Cloud",
    "Logivia",
    "Haulpath",
]

CSM_NOTE_TEMPLATES = {
    "Cost": [
        "Customer flagged pricing concerns at last QBR.",
        "Pushed back on annual uplift; finance team requested a cheaper alternative.",
        "Procurement review escalated; could not justify renewal at current price.",
        "Held a price-down workshop; commercial gap remained too large to bridge.",
    ],
    "Product fit": [
        "Use case shifted toward warehouse robotics; not a fit for our roadmap.",
        "Reporting requirements outgrew our standard exports.",
        "Operations team chose a vertical-specific tool for last-mile.",
        "Compliance module did not cover their freight forwarding scope.",
    ],
    "Competitor - FleetLogic": [
        "Moved to FleetLogic for integrated fleet telematics.",
        "Switched to FleetLogic; cited bundled telematics and routing.",
        "Renewal lost to FleetLogic on integrated proposition.",
        "Pilot of FleetLogic during renewal window converted into a full switch.",
    ],
    "Competitor - other": [
        "Moved to incumbent ERP add-on after group standardisation.",
        "Selected a regional Benelux provider on procurement preference.",
        "Renewed with a competing TMS bundled into their carrier deal.",
        "Replaced by a sister-brand platform after parent reorganisation.",
    ],
    "Acquisition/closure": [
        "Acquired by a larger 3PL; consolidating tools.",
        "Parent group consolidating onto a single platform post-merger.",
        "Site closure following parent restructure; subscription wound down.",
        "Trade sale completed; new owner moving onto in-house systems.",
    ],
    "Non-payment": [
        "Account moved to collections; no engagement on renewal.",
        "Multiple invoices unpaid; subscription suspended then closed.",
        "Direct debit failures across two cycles; renewal lapsed.",
        "Outstanding balance unresolved at renewal; account closed for cause.",
    ],
    "Other": [
        "Internal restructure; commercial sponsor moved on without succession.",
        "Contract lapsed without renewal conversation.",
        "Project team disbanded; tooling decommissioned.",
        "Pilot programme ended without sign-off from the new operations lead.",
    ],
}

QUIRK_A_COUNT = 3
QUIRK_B_UPLIFT_GBP = 500.0
QUIRK_C_COUNT = 5
QUIRK_C_OFFSET_RANGE_DAYS = (30, 91)  # high exclusive — yields 30..90 days
ELIGIBLE_QUIRK_MIN_ACTIVE_MONTHS = 18

SNAPSHOT_END = pd.Timestamp("2025-12-31")
ANALYSIS_START = pd.Timestamp("2023-01-01")

# Effective price per seat per year. List price is gross of segment discount.
EFFECTIVE_PRICE_PER_SEAT = {
    "SMB": 250.0,
    "Mid-market": 400.0,
    "Enterprise": 600.0,
}
LIST_DISCOUNT_RATE = {
    "SMB": 0.10,
    "Mid-market": 0.15,
    "Enterprise": 0.20,
}

MODULE_OPTIONS = {
    "SMB": [["Core"]],
    "Mid-market": [
        ["Core", "Analytics"],
        ["Core", "Routing"],
        ["Core", "Analytics", "Routing"],
    ],
    "Enterprise": [
        ["Core", "Analytics", "Routing"],
        ["Core", "Analytics", "Routing", "Compliance"],
    ],
}

# Per-month event probabilities and seat-change ranges.
EXPANSION_PROB = {
    "SMB": 0.040,
    "Mid-market": 0.140,
    "Enterprise": 0.180,
}
EXPANSION_PCT_RANGE = {
    "SMB": (0.04, 0.10),
    "Mid-market": (0.05, 0.12),
    "Enterprise": (0.06, 0.13),
}
CONTRACTION_PROB = {
    "SMB": 0.020,
    "Mid-market": 0.005,
    "Enterprise": 0.001,
}
CONTRACTION_PCT_RANGE = {
    "SMB": (0.05, 0.12),
    "Mid-market": (0.05, 0.10),
    "Enterprise": (0.03, 0.06),
}
ANNUAL_UPLIFT_PCT = {
    "SMB": 0.05,
    "Mid-market": 0.04,
    "Enterprise": 0.03,
}

MID_FLATLINE_DATE = pd.Timestamp("2024-07-01")
MID_POSTFLAT_EXPANSION_PROB = 0.018
# Enterprise expansion gently tapers from mid-2024 — interpreted as the
# macro headwind on logistics customers referenced in the engagement
# scenario. Mid-market shows competitor (FleetLogic) plus macro;
# Enterprise shows macro only, hence still positive net retention.
ENTERPRISE_TAPER_DATE = pd.Timestamp("2024-07-01")
ENTERPRISE_POSTTAPER_EXPANSION_PROB = 0.060
FLEET_LOGIC_STALL_RANGE_MONTHS = (2, 6)
FLEET_LOGIC_START = pd.Timestamp("2024-07-01")
FLEET_LOGIC_MID_SHARE = 0.30

# Total churn targets per segment across the data window.
CHURN_TARGETS = {
    "SMB": 25,
    "Mid-market": 50,
    "Enterprise": 5,
}

# Minimum tenure before a customer can churn (no instant-churn cases).
MIN_CHURN_TENURE_MONTHS = 6


def _build_label_pool(counts: dict[str, int]) -> list[str]:
    labels: list[str] = []
    for value, count in counts.items():
        labels.extend([value] * count)
    np.random.shuffle(labels)
    return labels


def _build_acquisition_dates() -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for year, year_total in YEAR_ACQUISITION_TARGETS.items():
        per_quarter = [year_total // 4] * 4
        for i in range(year_total - sum(per_quarter)):
            per_quarter[i] += 1
        for (_, m_start, m_end, last_day), q_count in zip(QUARTERS, per_quarter):
            start = pd.Timestamp(year=year, month=m_start, day=1)
            end = pd.Timestamp(year=year, month=m_end, day=last_day)
            span_days = (end - start).days + 1
            day_offsets = np.random.randint(0, span_days, size=q_count)
            for offset in day_offsets:
                dates.append(start + pd.Timedelta(days=int(offset)))
    np.random.shuffle(dates)
    return dates


def _build_unique_name_combos(n: int) -> list[tuple[str, str]]:
    combos = [(prefix, middle) for prefix in NAME_PREFIXES for middle in NAME_MIDDLES]
    if len(combos) < n:
        raise RuntimeError(
            f"Name pool too small: {len(combos)} unique combos for {n} customers."
        )
    indices = np.arange(len(combos))
    np.random.shuffle(indices)
    return [combos[int(i)] for i in indices[:n]]


def _draw_arr(segment: str) -> float:
    low, high = ARR_RANGES_GBP[segment]
    # Round to nearest £100 for clean reproducible values.
    return float(np.random.randint(low // 100, (high // 100) + 1) * 100)


def generate_customers() -> pd.DataFrame:
    segments = _build_label_pool(SEGMENT_COUNTS)
    countries = _build_label_pool(COUNTRY_COUNTS)
    acquisitions = _build_acquisition_dates()
    name_combos = _build_unique_name_combos(TOTAL_CUSTOMERS)

    industries = np.random.choice(INDUSTRIES, size=TOTAL_CUSTOMERS)
    channels = np.random.choice(CHANNELS, size=TOTAL_CUSTOMERS, p=CHANNEL_WEIGHTS)
    reps = np.random.choice(REPS, size=TOTAL_CUSTOMERS)
    invoice_offset_days = np.random.randint(0, 91, size=TOTAL_CUSTOMERS)
    arrs = np.array([_draw_arr(segments[i]) for i in range(TOTAL_CUSTOMERS)])

    rows = []
    for i in range(TOTAL_CUSTOMERS):
        prefix, middle = name_combos[i]
        suffix = COUNTRY_SUFFIXES[countries[i]]
        acquisition_date = acquisitions[i]
        first_paid = acquisition_date + pd.Timedelta(days=int(invoice_offset_days[i]))
        rows.append(
            {
                "customer_id": f"CUS-{i + 1:05d}",
                "customer_name": f"{prefix} {middle} {suffix}",
                "segment": segments[i],
                "industry_subsector": str(industries[i]),
                "country": countries[i],
                "acquisition_date": acquisition_date.strftime("%Y-%m-%d"),
                "first_paid_invoice_date": first_paid.strftime("%Y-%m-%d"),
                "acquisition_channel": str(channels[i]),
                "acquisition_rep": str(reps[i]),
                "original_arr_gbp": arrs[i],
            }
        )

    df = pd.DataFrame(rows, columns=CUSTOMER_COLUMNS)
    df = df.sort_values("customer_id", kind="stable").reset_index(drop=True)
    return df


def _assert_row_count(customers: pd.DataFrame) -> None:
    if len(customers) != TOTAL_CUSTOMERS:
        raise AssertionError(
            f"Expected {TOTAL_CUSTOMERS} customers, got {len(customers)}."
        )


def _assert_unique_names(customers: pd.DataFrame) -> None:
    duplicates = customers["customer_name"].duplicated().sum()
    if duplicates:
        raise AssertionError(f"Duplicate customer_name values: {duplicates}.")


def _assert_segment_arr_ranges(customers: pd.DataFrame) -> None:
    for segment, (low, high) in ARR_RANGES_GBP.items():
        sub = customers.loc[customers["segment"] == segment, "original_arr_gbp"]
        if sub.empty:
            raise AssertionError(f"No customers in segment {segment}.")
        if sub.min() < low or sub.max() > high:
            raise AssertionError(
                f"{segment} ARR outside [£{low:,}, £{high:,}]: "
                f"min £{sub.min():,.0f}, max £{sub.max():,.0f}."
            )


def _assert_baseline_constraint(customers: pd.DataFrame) -> None:
    fpi = pd.to_datetime(customers["first_paid_invoice_date"])
    live_before_q1_2023 = fpi <= pd.Timestamp("2022-12-31")
    pre_2022 = fpi <= pd.Timestamp("2021-12-31")
    n_live = int(live_before_q1_2023.sum())
    n_pre = int((live_before_q1_2023 & pre_2022).sum())
    if n_live == 0:
        raise AssertionError("No customers live at start of Q1 2023.")
    ratio = n_pre / n_live
    print(
        f"Baseline constraint: {ratio:.1%} of customers live at start of "
        f"Q1 2023 have first_paid_invoice_date <= 2021-12-31 "
        f"(need >= 60.0%, n_live={n_live}, n_pre={n_pre})."
    )
    if ratio < 0.60:
        raise AssertionError(f"Baseline constraint failed: {ratio:.1%} < 60.0%.")


def _assert_segment_distribution(customers: pd.DataFrame) -> None:
    counts = customers["segment"].value_counts().to_dict()
    for segment, target in SEGMENT_COUNTS.items():
        actual = counts.get(segment, 0)
        if actual != target:
            raise AssertionError(
                f"Segment {segment} count {actual} != target {target}."
            )


def _assert_country_distribution(customers: pd.DataFrame) -> None:
    counts = customers["country"].value_counts().to_dict()
    for country, target in COUNTRY_COUNTS.items():
        actual = counts.get(country, 0)
        if actual != target:
            raise AssertionError(
                f"Country {country} count {actual} != target {target}."
            )


def _write_customers(customers: pd.DataFrame) -> Path:
    out_path = DATA_DIR / "customers.csv"
    customers.to_csv(
        out_path,
        index=False,
        float_format="%.2f",
        lineterminator="\n",
    )
    return out_path


def _build_segment_churn_weights() -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    """Per-segment monthly weight grids for sampling churn dates.

    Mid-market is weighted heavily into H2 2024 onwards (FleetLogic emergence
    coincides with elevated mid-market churn). SMB drifts modestly upward over
    time. Enterprise is uniform — only 5 churners total, so the shape barely
    matters but we keep it deterministic.
    """
    months = pd.date_range("2022-01-01", "2025-12-01", freq="MS")
    weights = {}

    mid = np.ones(len(months))
    for i, m in enumerate(months):
        if m < pd.Timestamp("2023-01-01"):
            mid[i] = 0.6
        elif m < pd.Timestamp("2024-01-01"):
            mid[i] = 0.9
        elif m < pd.Timestamp("2024-07-01"):
            mid[i] = 1.2
        elif m < pd.Timestamp("2025-01-01"):
            mid[i] = 2.4
        else:
            mid[i] = 2.0
    weights["Mid-market"] = (months, mid / mid.sum())

    smb = np.ones(len(months))
    for i, m in enumerate(months):
        if m >= pd.Timestamp("2024-01-01"):
            smb[i] = 1.4
    weights["SMB"] = (months, smb / smb.sum())

    ent = np.ones(len(months))
    weights["Enterprise"] = (months, ent / ent.sum())

    return weights


def _draw_churn_date(
    segment: str,
    weights_index: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
    earliest: pd.Timestamp,
) -> pd.Timestamp:
    months, w = weights_index[segment]
    valid = months >= earliest.to_period("M").to_timestamp()
    if not valid.any():
        return SNAPSHOT_END
    valid_idx = np.where(valid)[0]
    valid_w = w[valid_idx]
    valid_w = valid_w / valid_w.sum()
    chosen_pos = int(np.random.choice(valid_idx, p=valid_w))
    chosen_month = months[chosen_pos]
    days_in_month = (chosen_month + pd.offsets.MonthEnd(0)).day
    day = int(np.random.randint(1, days_in_month + 1))
    return chosen_month.replace(day=day)


def decide_fates(customers: pd.DataFrame) -> pd.DataFrame:
    """Decide which customers churn, when, and which mid-market churners are
    coded as ``Competitor - FleetLogic`` (used downstream by step 4 to
    generate ``churn_reasons.csv`` consistently with subscription history).
    """
    fates = customers[["customer_id", "segment"]].copy()
    fates["first_paid"] = pd.to_datetime(customers["first_paid_invoice_date"])
    fates["churn_date"] = pd.NaT
    fates["is_fleet_logic"] = False

    weights_index = _build_segment_churn_weights()
    fates_indexed = fates.set_index("customer_id")

    for segment, target in CHURN_TARGETS.items():
        seg = fates[fates["segment"] == segment]
        eligible_mask = (
            seg["first_paid"] + pd.DateOffset(months=MIN_CHURN_TENURE_MONTHS)
            <= SNAPSHOT_END
        )
        eligible_ids = seg.loc[eligible_mask, "customer_id"].sort_values().to_numpy()
        if len(eligible_ids) < target:
            raise AssertionError(
                f"Not enough eligible {segment} customers for churn target "
                f"{target}: only {len(eligible_ids)} available."
            )
        chosen_ids = np.random.choice(eligible_ids, size=target, replace=False)
        chosen_ids = np.sort(chosen_ids)  # process in customer_id order for determinism
        for cid in chosen_ids:
            first_paid = fates_indexed.at[cid, "first_paid"]
            earliest = first_paid + pd.DateOffset(months=MIN_CHURN_TENURE_MONTHS)
            churn_date = _draw_churn_date(segment, weights_index, earliest)
            if churn_date > SNAPSHOT_END:
                churn_date = SNAPSHOT_END
            fates_indexed.at[cid, "churn_date"] = churn_date

        if segment == "Mid-market":
            mid_chosen = chosen_ids
            mid_dates = fates_indexed.loc[mid_chosen, "churn_date"]
            late_ids = mid_dates[mid_dates >= FLEET_LOGIC_START].index.to_numpy()
            n_fleet = int(round(FLEET_LOGIC_MID_SHARE * len(late_ids)))
            if n_fleet > 0 and len(late_ids) > 0:
                fleet_ids = np.random.choice(late_ids, size=n_fleet, replace=False)
                fates_indexed.loc[fleet_ids, "is_fleet_logic"] = True

    return fates_indexed.reset_index()


def _draw_modules(segment: str) -> str:
    options = MODULE_OPTIONS[segment]
    pick = options[int(np.random.randint(0, len(options)))]
    return "|".join(pick)


def _simulate_customer_subscriptions(
    customer_row: pd.Series,
    churn_date: pd.Timestamp,
    is_fleet_logic: bool,
) -> list[dict]:
    segment = customer_row["segment"]
    first_paid = pd.to_datetime(customer_row["first_paid_invoice_date"])
    if first_paid > SNAPSHOT_END:
        return []

    if pd.notna(churn_date):
        last_month_end = (churn_date + pd.offsets.MonthEnd(0))
    else:
        last_month_end = SNAPSHOT_END

    first_month_end = first_paid + pd.offsets.MonthEnd(0)
    if last_month_end < first_month_end:
        last_month_end = first_month_end

    months = pd.date_range(first_month_end, last_month_end, freq="ME")

    eff_price = EFFECTIVE_PRICE_PER_SEAT[segment]
    list_price = eff_price / (1 - LIST_DISCOUNT_RATE[segment])
    initial_arr = float(customer_row["original_arr_gbp"])
    initial_seats = max(1, int(round(initial_arr / eff_price)))
    modules_str = _draw_modules(segment)

    if is_fleet_logic and pd.notna(churn_date):
        stall_months = int(
            np.random.randint(
                FLEET_LOGIC_STALL_RANGE_MONTHS[0],
                FLEET_LOGIC_STALL_RANGE_MONTHS[1] + 1,
            )
        )
        stall_start = (
            churn_date - pd.DateOffset(months=stall_months) + pd.offsets.MonthEnd(0)
        )
    else:
        stall_start = None

    cur_seats = initial_seats
    cur_eff = eff_price
    cur_list = list_price
    cur_contract_end = first_paid + pd.DateOffset(months=12) - pd.Timedelta(days=1)

    rows: list[dict] = []
    n = len(months)
    for i, m in enumerate(months):
        is_last = (i == n - 1)
        is_churn_row = is_last and pd.notna(churn_date)

        if i > 0:
            while m > cur_contract_end:
                uplift = ANNUAL_UPLIFT_PCT[segment]
                cur_list *= (1 + uplift)
                cur_eff *= (1 + uplift)
                cur_contract_end = cur_contract_end + pd.DateOffset(months=12)

            if not is_churn_row:
                exp_prob = EXPANSION_PROB[segment]
                if segment == "Mid-market":
                    if m >= MID_FLATLINE_DATE:
                        exp_prob = MID_POSTFLAT_EXPANSION_PROB
                    if stall_start is not None and m >= stall_start:
                        exp_prob = 0.0
                elif segment == "Enterprise" and m >= ENTERPRISE_TAPER_DATE:
                    exp_prob = ENTERPRISE_POSTTAPER_EXPANSION_PROB

                roll_e = float(np.random.random())
                roll_c = float(np.random.random())

                if roll_e < exp_prob:
                    pct_low, pct_high = EXPANSION_PCT_RANGE[segment]
                    pct = float(np.random.uniform(pct_low, pct_high))
                    cur_seats = max(1, int(round(cur_seats * (1 + pct))))
                elif roll_c < CONTRACTION_PROB[segment]:
                    pct_low, pct_high = CONTRACTION_PCT_RANGE[segment]
                    pct = float(np.random.uniform(pct_low, pct_high))
                    cur_seats = max(1, int(round(cur_seats * (1 - pct))))

        if is_churn_row:
            arr = 0.0
            seat_out = 0
            status = "Churned"
        else:
            arr = round(cur_seats * cur_eff, 2)
            seat_out = int(cur_seats)
            status = "Active"

        rows.append(
            {
                "customer_id": customer_row["customer_id"],
                "month_end_date": m.strftime("%Y-%m-%d"),
                "arr_gbp": arr,
                "seats": seat_out,
                "product_modules": modules_str,
                "subscription_status": status,
                "contract_end_date": cur_contract_end.strftime("%Y-%m-%d"),
                "list_price_per_seat": round(cur_list, 2),
                "effective_price_per_seat": round(cur_eff, 2),
            }
        )

    return rows


def generate_subscriptions(customers: pd.DataFrame, fates: pd.DataFrame) -> pd.DataFrame:
    fates_indexed = fates.set_index("customer_id")
    rows: list[dict] = []
    # Sort by customer_id so np.random calls happen in a fixed order.
    ordered = customers.sort_values("customer_id", kind="stable").reset_index(drop=True)
    for _, customer in ordered.iterrows():
        cid = customer["customer_id"]
        churn_date = fates_indexed.at[cid, "churn_date"]
        is_fleet = bool(fates_indexed.at[cid, "is_fleet_logic"])
        rows.extend(_simulate_customer_subscriptions(customer, churn_date, is_fleet))
    df = pd.DataFrame(rows, columns=SUBSCRIPTION_COLUMNS)
    df = df.sort_values(["customer_id", "month_end_date"], kind="stable").reset_index(drop=True)
    return df


def _assert_active_arr_positive(subscriptions: pd.DataFrame) -> None:
    active = subscriptions[subscriptions["subscription_status"] == "Active"]
    bad = active[active["arr_gbp"] <= 0]
    if not bad.empty:
        raise AssertionError(
            f"{len(bad)} Active rows have arr_gbp <= 0. "
            f"First offender: {bad.iloc[0].to_dict()}"
        )


def _assert_single_churn_per_customer(subscriptions: pd.DataFrame) -> None:
    churn_counts = (
        subscriptions[subscriptions["subscription_status"] == "Churned"]
        .groupby("customer_id")
        .size()
    )
    if (churn_counts > 1).any():
        raise AssertionError("Some customers have more than one Churned row.")


def _assert_decomposition_reconciles(subscriptions: pd.DataFrame) -> None:
    df = subscriptions.sort_values(["customer_id", "month_end_date"], kind="stable").copy()
    df["prev_arr"] = df.groupby("customer_id")["arr_gbp"].shift(1).fillna(0.0)
    df["is_first"] = df.groupby("customer_id").cumcount() == 0
    df["is_churn"] = df["subscription_status"] == "Churned"
    delta = df["arr_gbp"] - df["prev_arr"]

    df["new_arr"] = np.where(df["is_first"], df["arr_gbp"], 0.0)
    df["churn_arr"] = np.where(df["is_churn"], df["prev_arr"], 0.0)
    cont_mask = (~df["is_first"]) & (~df["is_churn"])
    df["expansion_arr"] = np.where(cont_mask & (delta > 0), delta, 0.0)
    df["contraction_arr"] = np.where(cont_mask & (delta < 0), -delta, 0.0)

    recon = (
        df["prev_arr"]
        + df["new_arr"]
        + df["expansion_arr"]
        - df["contraction_arr"]
        - df["churn_arr"]
    )
    gap = (df["arr_gbp"] - recon).abs()
    if gap.max() > 0.01:
        offender = df.loc[gap.idxmax()]
        raise AssertionError(
            "Per-customer-month decomposition does not reconcile. "
            f"Max gap £{gap.max():.4f} at {offender['customer_id']} "
            f"{offender['month_end_date']}."
        )


def _quarter_end(year: int, quarter: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=quarter * 3, day=1) + pd.offsets.MonthEnd(0)


def _ttm_nrr_components(
    subscriptions: pd.DataFrame,
    eligible_ids: set[str],
    quarter_end: pd.Timestamp,
) -> tuple[float, float, float]:
    """Return (arr_then, arr_now_active, churned_then_arr) for the TTM cohort.

    Cohort = customers in ``eligible_ids`` (i.e. acquired before the analysis
    period) that were Active at the start of the trailing window
    (``quarter_end - 12 months``). ``arr_now_active`` sums the cohort's ARR at
    ``quarter_end`` from rows still marked Active. ``churned_then_arr`` is
    the cohort starting ARR for customers who churned during the window.
    """
    start_window = (quarter_end - pd.DateOffset(months=12)) + pd.offsets.MonthEnd(0)
    start_str = start_window.strftime("%Y-%m-%d")
    end_str = quarter_end.strftime("%Y-%m-%d")

    at_start = subscriptions[
        (subscriptions["month_end_date"] == start_str)
        & (subscriptions["subscription_status"] == "Active")
        & (subscriptions["customer_id"].isin(eligible_ids))
    ]
    cohort_ids = set(at_start["customer_id"])
    arr_then = float(at_start["arr_gbp"].sum())

    at_end = subscriptions[
        (subscriptions["month_end_date"] == end_str)
        & (subscriptions["customer_id"].isin(cohort_ids))
    ]
    arr_now_active = float(
        at_end.loc[at_end["subscription_status"] == "Active", "arr_gbp"].sum()
    )

    churned_window = subscriptions[
        (subscriptions["customer_id"].isin(cohort_ids))
        & (subscriptions["subscription_status"] == "Churned")
        & (subscriptions["month_end_date"] > start_str)
        & (subscriptions["month_end_date"] <= end_str)
    ]
    churned_starting_arr = float(
        at_start[at_start["customer_id"].isin(churned_window["customer_id"])][
            "arr_gbp"
        ].sum()
    )

    return arr_then, arr_now_active, churned_starting_arr


def _assert_story_targets(
    subscriptions: pd.DataFrame, customers: pd.DataFrame
) -> None:
    """Sanity-check that the headline story shape holds.

    The customer-mix in ``customers.csv`` is heavily weighted to Enterprise
    ARR (~62% of total), which means the spec's exact NRR figures (~118% →
    ~104%) cannot be hit precisely without distorting per-segment NRRs. We
    check qualitative direction and segment-level dispersion within wide
    bands, plus a hard floor on GRR (gross retention should not collapse).
    """
    cust_fpi = pd.to_datetime(customers["first_paid_invoice_date"])
    pre_analysis_ids = set(
        customers.loc[cust_fpi <= pd.Timestamp("2022-12-31"), "customer_id"]
    )

    quarters = [
        (y, q) for y in (2023, 2024, 2025) for q in (1, 2, 3, 4)
    ]
    nrrs = []
    grrs = []
    for y, q in quarters:
        qe = _quarter_end(y, q)
        arr_then, arr_now, churned_then = _ttm_nrr_components(
            subscriptions, pre_analysis_ids, qe
        )
        if arr_then <= 0:
            raise AssertionError(f"No starting ARR for Q{q} {y} TTM cohort.")
        nrrs.append(arr_now / arr_then)
        grrs.append(max(0.0, (arr_then - churned_then)) / arr_then)

    nrr_first = nrrs[0]
    nrr_last = nrrs[-1]
    grr_min = min(grrs)
    grr_max = max(grrs)

    print(
        "TTM NRR Q1 2023: {:.1%}, Q4 2025: {:.1%}; "
        "GRR range across 12 quarters: {:.1%}–{:.1%}.".format(
            nrr_first, nrr_last, grr_min, grr_max
        )
    )

    if not (1.10 <= nrr_first <= 1.30):
        raise AssertionError(
            f"Q1 2023 TTM NRR {nrr_first:.1%} outside expected band [110%, 130%]."
        )
    if not (0.95 <= nrr_last <= 1.15):
        raise AssertionError(
            f"Q4 2025 TTM NRR {nrr_last:.1%} outside expected band [95%, 115%]."
        )
    if (nrr_first - nrr_last) < 0.05:
        raise AssertionError(
            f"NRR decline {nrr_first - nrr_last:.1%} too small "
            f"(expected >=5pp Q1 2023 to Q4 2025)."
        )
    if grr_min < 0.85:
        raise AssertionError(
            f"GRR min {grr_min:.1%} below 85% floor — gross churn looks excessive."
        )
    if grr_max > 0.99:
        raise AssertionError(
            f"GRR max {grr_max:.1%} above 99% ceiling — too little gross churn."
        )

    # Mid-market cohort dispersion: must show a clear collapse Q1 2023 -> Q4 2025.
    mid_ids = set(
        customers.loc[customers["segment"] == "Mid-market", "customer_id"]
    ) & pre_analysis_ids
    if mid_ids:
        first_arr_then, first_arr_now, _ = _ttm_nrr_components(
            subscriptions, mid_ids, _quarter_end(2023, 1)
        )
        last_arr_then, last_arr_now, _ = _ttm_nrr_components(
            subscriptions, mid_ids, _quarter_end(2025, 4)
        )
        mid_nrr_first = first_arr_now / first_arr_then if first_arr_then else 0
        mid_nrr_last = last_arr_now / last_arr_then if last_arr_then else 0
        print(
            "Mid-market TTM NRR Q1 2023: {:.1%}, Q4 2025: {:.1%}.".format(
                mid_nrr_first, mid_nrr_last
            )
        )
        if (mid_nrr_first - mid_nrr_last) < 0.10:
            raise AssertionError(
                f"Mid-market NRR collapse {mid_nrr_first - mid_nrr_last:.1%} "
                "too shallow (expected >=10pp Q1 2023 to Q4 2025)."
            )


def _write_subscriptions(subscriptions: pd.DataFrame) -> Path:
    out_path = DATA_DIR / "subscriptions.csv"
    subscriptions.to_csv(
        out_path,
        index=False,
        float_format="%.2f",
        lineterminator="\n",
    )
    return out_path


def _apply_quirks_a_b(subscriptions: pd.DataFrame, fates: pd.DataFrame) -> pd.DataFrame:
    """Inject Quirk A (three £1 single-month dips) and Quirk B (one backdated
    month-6 uplift) into the subscription frame.

    Quirk A targets a stable mid-history month (prev = current = next ARR) and
    knocks the middle month down by exactly £1 — the canonical billing-system
    rounding glitch shape.

    Quirk B mimics an unrecorded mid-contract uplift later reconciled: month 6
    gets a £500 bump while months 5 and 7 remain at the natural baseline.
    """
    df = subscriptions.copy()
    fates_idx = fates.set_index("customer_id")

    active_mask = df["subscription_status"] == "Active"
    active_counts = df[active_mask].groupby("customer_id").size()

    non_churner_ids = fates_idx[fates_idx["churn_date"].isna()].index.to_numpy()
    eligible = np.array(sorted(
        cid for cid in non_churner_ids
        if active_counts.get(cid, 0) >= ELIGIBLE_QUIRK_MIN_ACTIVE_MONTHS
    ))
    if len(eligible) < QUIRK_A_COUNT + 1:
        raise AssertionError(
            f"Only {len(eligible)} eligible customers for DQ quirks A/B; "
            f"need at least {QUIRK_A_COUNT + 1}."
        )

    quirk_a_ids = np.sort(
        np.random.choice(eligible, size=QUIRK_A_COUNT, replace=False)
    )
    for cid in quirk_a_ids:
        cust_idx = df.index[(df["customer_id"] == cid) & active_mask].sort_values()
        arrs = df.loc[cust_idx, "arr_gbp"].to_numpy()
        n = len(arrs)
        stable_positions = [
            i for i in range(6, n - 4)
            if arrs[i - 1] == arrs[i] == arrs[i + 1]
        ]
        if not stable_positions:
            raise AssertionError(
                f"No stable mid-history month found for Quirk A on {cid}."
            )
        pos = int(np.random.choice(stable_positions))
        df.at[cust_idx[pos], "arr_gbp"] = round(float(arrs[pos]) - 1.0, 2)

    quirk_b_pool = [c for c in eligible if c not in set(quirk_a_ids)]
    valid_b_ids: list[str] = []
    for cid in quirk_b_pool:
        cust_idx = df.index[(df["customer_id"] == cid) & active_mask].sort_values()
        arrs = df.loc[cust_idx, "arr_gbp"].to_numpy()
        if len(arrs) >= 8 and arrs[4] == arrs[5] == arrs[6]:
            valid_b_ids.append(cid)
    if not valid_b_ids:
        raise AssertionError("No eligible customer found for Quirk B.")
    quirk_b_id = str(np.random.choice(np.array(valid_b_ids)))
    cust_idx = df.index[(df["customer_id"] == quirk_b_id) & active_mask].sort_values()
    arrs = df.loc[cust_idx, "arr_gbp"].to_numpy()
    df.at[cust_idx[5], "arr_gbp"] = round(float(arrs[5]) + QUIRK_B_UPLIFT_GBP, 2)

    return df


def _apply_quirk_c(customers: pd.DataFrame) -> pd.DataFrame:
    """Document five customers whose acquisition_date precedes
    first_paid_invoice_date by 1–3 months (free-trial period not reflected in
    the CRM record). Acquisition dates are shifted earlier in place; downstream
    flows do not consume acquisition_date so the perturbation is local.
    """
    df = customers.copy()
    sorted_df = df.sort_values("customer_id", kind="stable").reset_index(drop=True)
    chosen_pos = np.sort(
        np.random.choice(len(sorted_df), size=QUIRK_C_COUNT, replace=False)
    )
    low, high = QUIRK_C_OFFSET_RANGE_DAYS
    offsets = np.random.randint(low, high, size=QUIRK_C_COUNT)
    for pos, off in zip(chosen_pos, offsets):
        cid = sorted_df.iloc[int(pos)]["customer_id"]
        fpi = pd.to_datetime(sorted_df.iloc[int(pos)]["first_paid_invoice_date"])
        new_acq = (fpi - pd.Timedelta(days=int(off))).strftime("%Y-%m-%d")
        df.loc[df["customer_id"] == cid, "acquisition_date"] = new_acq
    return df


def _draw_csm_note(reason: str) -> str:
    pool = CSM_NOTE_TEMPLATES[reason]
    return pool[int(np.random.randint(0, len(pool)))]


def _draw_competitor_won_to(reason: str) -> str:
    if reason == "Competitor - FleetLogic":
        return "FleetLogic"
    if reason == "Competitor - other":
        return OTHER_COMPETITORS[int(np.random.randint(0, len(OTHER_COMPETITORS)))]
    return ""


def _assign_reason(segment: str, is_fleet_logic: bool) -> str:
    if is_fleet_logic:
        return "Competitor - FleetLogic"
    if segment == "Enterprise":
        return str(np.random.choice(
            ["Acquisition/closure", "Other", "Cost"],
            p=[0.7, 0.2, 0.1],
        ))
    if segment == "SMB":
        return str(np.random.choice(
            ["Cost", "Product fit", "Non-payment", "Other"],
            p=[0.45, 0.30, 0.15, 0.10],
        ))
    return str(np.random.choice(
        ["Cost", "Product fit", "Competitor - other", "Acquisition/closure", "Other"],
        p=[0.30, 0.25, 0.25, 0.10, 0.10],
    ))


def generate_churn_reasons(customers: pd.DataFrame, fates: pd.DataFrame) -> pd.DataFrame:
    """Build one row per churned customer with reason, optional competitor and
    a plausible CSM note. FleetLogic assignment is governed by the
    ``is_fleet_logic`` flag set in ``decide_fates`` — that flag is only true for
    mid-market customers churning on or after 2024-07-01, which enforces the
    competitor's temporal gate at source.
    """
    fates_local = fates.copy()
    fates_local["churn_date"] = pd.to_datetime(fates_local["churn_date"])
    churners = (
        fates_local[fates_local["churn_date"].notna()]
        .sort_values("customer_id", kind="stable")
        .reset_index(drop=True)
    )

    rows: list[dict] = []
    for _, row in churners.iterrows():
        reason = _assign_reason(row["segment"], bool(row["is_fleet_logic"]))
        won_to = _draw_competitor_won_to(reason)
        notes = _draw_csm_note(reason)
        rows.append(
            {
                "customer_id": row["customer_id"],
                "churn_date": row["churn_date"].strftime("%Y-%m-%d"),
                "primary_reason": reason,
                "competitor_won_to": won_to,
                "csm_notes": notes,
            }
        )

    df = pd.DataFrame(rows, columns=CHURN_REASON_COLUMNS)
    df = df.sort_values("customer_id", kind="stable").reset_index(drop=True)
    return df


def _write_churn_reasons(churn_reasons: pd.DataFrame) -> Path:
    out_path = DATA_DIR / "churn_reasons.csv"
    churn_reasons.to_csv(out_path, index=False, lineterminator="\n")
    return out_path


def _assert_quirks_present(subscriptions: pd.DataFrame) -> None:
    s = subscriptions.sort_values(
        ["customer_id", "month_end_date"], kind="stable"
    ).copy()
    s["prev_arr"] = s.groupby("customer_id")["arr_gbp"].shift(1)
    s["next_arr"] = s.groupby("customer_id")["arr_gbp"].shift(-1)

    a_dips = s[
        (s["subscription_status"] == "Active")
        & ((s["prev_arr"] - s["arr_gbp"]).round(2) == 1.0)
        & ((s["next_arr"] - s["arr_gbp"]).round(2) == 1.0)
    ]
    if len(a_dips) != QUIRK_A_COUNT:
        raise AssertionError(
            f"Quirk A: expected {QUIRK_A_COUNT} dips, found {len(a_dips)}."
        )

    b_uplifts = s[
        (s["subscription_status"] == "Active")
        & ((s["arr_gbp"] - s["prev_arr"]).round(2) == QUIRK_B_UPLIFT_GBP)
        & ((s["arr_gbp"] - s["next_arr"]).round(2) == QUIRK_B_UPLIFT_GBP)
    ]
    if len(b_uplifts) != 1:
        raise AssertionError(
            f"Quirk B: expected 1 backdated uplift, found {len(b_uplifts)}."
        )


def _assert_quirk_c(customers: pd.DataFrame) -> None:
    acq = pd.to_datetime(customers["acquisition_date"])
    fpi = pd.to_datetime(customers["first_paid_invoice_date"])
    gap_days = (fpi - acq).dt.days
    flagged = customers[(gap_days >= 30) & (gap_days <= 90)]
    if len(flagged) < QUIRK_C_COUNT:
        raise AssertionError(
            f"Quirk C: expected at least {QUIRK_C_COUNT} customers with "
            f"30–90 day acquisition/billing gap, found {len(flagged)}."
        )


def _assert_churn_reasons(
    churn_reasons: pd.DataFrame, customers: pd.DataFrame
) -> None:
    cust_seg = customers.set_index("customer_id")["segment"]
    cr = churn_reasons.copy()
    cr["churn_date"] = pd.to_datetime(cr["churn_date"])
    cr["segment"] = cr["customer_id"].map(cust_seg)

    if cr["customer_id"].duplicated().any():
        raise AssertionError("churn_reasons.csv has duplicate customer_id rows.")
    if cr["primary_reason"].isna().any():
        raise AssertionError("churn_reasons.csv has rows missing primary_reason.")
    if cr["csm_notes"].str.strip().eq("").any():
        raise AssertionError("churn_reasons.csv has empty csm_notes entries.")

    fleet = cr[cr["primary_reason"] == "Competitor - FleetLogic"]
    if (fleet["churn_date"] < FLEET_LOGIC_START).any():
        raise AssertionError("FleetLogic churn date precedes 2024-07-01.")
    if (fleet["segment"] != "Mid-market").any():
        raise AssertionError("Non-mid-market customer assigned FleetLogic reason.")

    ent = cr[cr["segment"] == "Enterprise"]
    if len(ent) > CHURN_TARGETS["Enterprise"]:
        raise AssertionError(
            f"Enterprise churn count {len(ent)} exceeds target "
            f"{CHURN_TARGETS['Enterprise']}."
        )

    smb = cr[cr["segment"] == "SMB"]
    if len(smb) > 0:
        smb_cf = smb[smb["primary_reason"].isin(["Cost", "Product fit"])]
        share = len(smb_cf) / len(smb)
        if share < 0.5:
            raise AssertionError(
                f"SMB Cost+Product-fit share {share:.1%} below 50% floor."
            )

    mid = cr[cr["segment"] == "Mid-market"]
    mid_window = mid[mid["churn_date"] >= FLEET_LOGIC_START]
    if len(mid_window) > 0:
        fleet_mid = mid_window[mid_window["primary_reason"] == "Competitor - FleetLogic"]
        share = len(fleet_mid) / len(mid_window)
        if not (0.15 <= share <= 0.45):
            raise AssertionError(
                f"FleetLogic share of mid-market H2-2024+ churn {share:.1%} "
                f"outside [15%, 45%]."
            )

    competitor_rows = cr[cr["primary_reason"].str.startswith("Competitor")]
    if competitor_rows["competitor_won_to"].str.strip().eq("").any():
        raise AssertionError(
            "competitor_won_to is empty on a Competitor primary_reason row."
        )
    non_competitor = cr[~cr["primary_reason"].str.startswith("Competitor")]
    if non_competitor["competitor_won_to"].str.strip().ne("").any():
        raise AssertionError(
            "competitor_won_to populated on a non-Competitor row."
        )


def main() -> None:
    customers = generate_customers()
    _assert_row_count(customers)
    _assert_segment_distribution(customers)
    _assert_country_distribution(customers)
    _assert_unique_names(customers)
    _assert_segment_arr_ranges(customers)
    _assert_baseline_constraint(customers)

    fates = decide_fates(customers)
    subscriptions = generate_subscriptions(customers, fates)
    subscriptions = _apply_quirks_a_b(subscriptions, fates)
    churn_reasons = generate_churn_reasons(customers, fates)
    customers = _apply_quirk_c(customers)

    _assert_active_arr_positive(subscriptions)
    _assert_single_churn_per_customer(subscriptions)
    _assert_decomposition_reconciles(subscriptions)
    _assert_quirks_present(subscriptions)
    _assert_story_targets(subscriptions, customers)
    _assert_quirk_c(customers)
    _assert_churn_reasons(churn_reasons, customers)

    customers_path = _write_customers(customers)
    print(f"customers: {len(customers)} rows -> {customers_path.name}")
    sub_path = _write_subscriptions(subscriptions)
    print(f"subscriptions: {len(subscriptions)} rows -> {sub_path.name}")
    cr_path = _write_churn_reasons(churn_reasons)
    print(f"churn_reasons: {len(churn_reasons)} rows -> {cr_path.name}")


if __name__ == "__main__":
    main()
