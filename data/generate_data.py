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


def main() -> None:
    customers = generate_customers()
    _assert_row_count(customers)
    _assert_segment_distribution(customers)
    _assert_country_distribution(customers)
    _assert_unique_names(customers)
    _assert_segment_arr_ranges(customers)
    _assert_baseline_constraint(customers)
    out_path = _write_customers(customers)
    print(f"customers: {len(customers)} rows -> {out_path.name}")


if __name__ == "__main__":
    main()
