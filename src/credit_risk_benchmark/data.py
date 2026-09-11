"""Leakage-safe ingestion and cohort construction for SBA 7(a) loan data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = [
    "AsOfDate",
    "Program",
    "LocationID",
    "BankState",
    "GrossApproval",
    "SBAGuaranteedApproval",
    "ApprovalDate",
    "ApprovalFY",
    "FirstDisbursementDate",
    "ProcessingMethod",
    "InitialInterestRate",
    "FixedorVariableInterestInd",
    "TermInMonths",
    "NaicsCode",
    "ProjectState",
    "BusinessType",
    "BusinessAge",
    "LoanStatus",
    "PaidInFullDate",
    "ChargeOffDate",
    "GrossChargeOffAmount",
    "RevolverStatus",
    "JobsSupported",
    "CollateralInd",
    "SoldSecMrktInd",
]

# These columns are needed to construct outcomes but must never enter a model.
LEAKAGE_COLUMNS = frozenset(
    {
        "AsOfDate",
        "LoanStatus",
        "PaidInFullDate",
        "ChargeOffDate",
        "GrossChargeOffAmount",
        "SoldSecMrktInd",
    }
)

NUMERIC_FEATURES = [
    "log_gross_approval",
    "guarantee_percentage",
    "term_months",
    "initial_interest_rate",
    "log_jobs_supported",
]

CATEGORICAL_FEATURES = [
    "lender_id",
    "bank_state",
    "processing_method",
    "fixed_or_variable",
    "naics_sector",
    "project_state",
    "business_type",
    "business_age_band",
    "collateral_indicator",
    "revolver_indicator",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
AUDIT_COLUMNS = ["approval_date", "approval_year", "business_type", "business_age_band"]


@dataclass(frozen=True)
class CohortConfig:
    """Define the prediction point, outcome horizon, and modelling window."""

    horizon_days: int = 1_826
    first_model_year: int = 2010
    last_model_year: int = 2020

    def __post_init__(self) -> None:
        if self.horizon_days < 1:
            raise ValueError("horizon_days must be positive")
        if self.first_model_year > self.last_model_year:
            raise ValueError("first_model_year cannot exceed last_model_year")


def discover_csv_files(data_dir: str | Path) -> list[Path]:
    """Return the four expected SBA 7(a) CSV files in chronological order."""
    paths = sorted(Path(data_dir).glob("sba_7a_*.csv"))
    if len(paths) != 4:
        raise FileNotFoundError(
            f"expected four SBA 7(a) CSV files in {Path(data_dir)}, found {len(paths)}"
        )
    return paths


def _text(series: pd.Series, *, missing: str = "UNKNOWN") -> pd.Series:
    values = series.astype("string").str.strip()
    return values.mask(values.isna() | values.eq(""), missing).astype(str)


def normalize_loan_status(series: pd.Series) -> pd.Series:
    """Normalise status variants such as ``P I F`` to ``PIF``."""
    return _text(series).str.replace(r"\s+", "", regex=True).str.upper()


def normalize_business_age(series: pd.Series) -> pd.Series:
    """Map changing SBA business-age vocabularies to stable analytic bands."""
    values = _text(series).str.lower()
    result = pd.Series("other", index=series.index, dtype="string")
    result[values.str.contains("startup", regex=False)] = "startup"
    result[values.str.contains("new", regex=False)] = "new_or_young"
    result[values.str.contains("less than", regex=False)] = "new_or_young"
    result[values.str.contains("existing", regex=False)] = "established"
    result[values.str.contains("change of ownership", regex=False)] = "change_of_ownership"
    result[values.isin(["unknown", "unanswered"])] = "unknown"
    return result.astype(str)


def _clean_numeric(series: pd.Series, *, lower: float, upper: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    return values.where(values.between(lower, upper))


def prepare_sba_chunk(
    raw: pd.DataFrame,
    *,
    config: CohortConfig | None = None,
    restrict_model_window: bool = True,
) -> pd.DataFrame:
    """Create a five-year charge-off cohort from one raw SBA chunk.

    The prediction point is first disbursement. A loan is included only after
    the complete five-year performance window is observable in the snapshot.
    Cancelled and undisbursed commitments are excluded. A charge-off after the
    five-year horizon is correctly treated as a non-event for this target.
    """
    config = config or CohortConfig()
    missing_columns = sorted(set(RAW_COLUMNS).difference(raw.columns))
    if missing_columns:
        raise ValueError(f"missing required SBA columns: {missing_columns}")

    approval_date = pd.to_datetime(raw["ApprovalDate"], errors="coerce")
    disbursement_date = pd.to_datetime(raw["FirstDisbursementDate"], errors="coerce")
    charge_off_date = pd.to_datetime(raw["ChargeOffDate"], errors="coerce")
    as_of_date = pd.to_datetime(raw["AsOfDate"], errors="coerce")
    status = normalize_loan_status(raw["LoanStatus"])
    approval_year = pd.to_numeric(raw["ApprovalFY"], errors="coerce")

    observed_days = (as_of_date - disbursement_date).dt.days
    charge_off_days = (charge_off_date - disbursement_date).dt.days
    excluded_status = status.isin(["CANCLD", "COMMIT"])
    invalid_dated_charge_off = status.eq("CHGOFF") & charge_off_date.isna()
    impossible_dates = (
        approval_date.isna()
        | disbursement_date.isna()
        | as_of_date.isna()
        | (disbursement_date < approval_date - pd.Timedelta(days=90))
        | (charge_off_days < 0)
    )
    eligible = (
        ~excluded_status
        & ~invalid_dated_charge_off
        & ~impossible_dates
        & observed_days.ge(config.horizon_days)
    )
    if restrict_model_window:
        eligible &= approval_year.between(config.first_model_year, config.last_model_year)

    selected = raw.loc[eligible].copy()
    if selected.empty:
        return pd.DataFrame(columns=MODEL_FEATURES + AUDIT_COLUMNS + ["target"])

    selected_approval = approval_date.loc[eligible]
    selected_charge_days = charge_off_days.loc[eligible]
    gross_approval = _clean_numeric(
        selected["GrossApproval"], lower=1.0, upper=100_000_000.0
    )
    guaranteed_approval = _clean_numeric(
        selected["SBAGuaranteedApproval"], lower=0.0, upper=100_000_000.0
    )

    output = pd.DataFrame(index=selected.index)
    output["log_gross_approval"] = np.log1p(gross_approval)
    output["guarantee_percentage"] = (guaranteed_approval / gross_approval).where(
        gross_approval.gt(0)
    )
    output["guarantee_percentage"] = output["guarantee_percentage"].where(
        output["guarantee_percentage"].between(0.0, 1.5)
    )
    output["term_months"] = _clean_numeric(
        selected["TermInMonths"], lower=1.0, upper=600.0
    )
    output["initial_interest_rate"] = _clean_numeric(
        selected["InitialInterestRate"], lower=0.0, upper=50.0
    )
    jobs = _clean_numeric(selected["JobsSupported"], lower=0.0, upper=100_000.0)
    output["log_jobs_supported"] = np.log1p(jobs)

    output["lender_id"] = _text(selected["LocationID"])
    output["bank_state"] = _text(selected["BankState"]).str.upper()
    output["processing_method"] = _text(selected["ProcessingMethod"])
    output["fixed_or_variable"] = _text(
        selected["FixedorVariableInterestInd"]
    ).str.upper()
    naics = pd.to_numeric(selected["NaicsCode"], errors="coerce")
    output["naics_sector"] = (
        naics.round().astype("Int64").astype("string").str.zfill(6).str[:2].fillna("UNKNOWN")
    ).astype(str)
    output["project_state"] = _text(selected["ProjectState"]).str.upper()
    output["business_type"] = _text(selected["BusinessType"]).str.upper()
    output["business_age_band"] = normalize_business_age(selected["BusinessAge"])
    output["collateral_indicator"] = _text(selected["CollateralInd"]).str.upper()
    output["revolver_indicator"] = _text(selected["RevolverStatus"]).str.upper()

    output["approval_date"] = selected_approval
    output["approval_year"] = approval_year.loc[eligible].astype(int)
    output["target"] = selected_charge_days.between(0, config.horizon_days).astype(int)
    return output.reset_index(drop=True)


def load_sba_model_cohort(
    paths: Iterable[str | Path],
    *,
    config: CohortConfig | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Load all eligible model-window loans without direct borrower identifiers."""
    config = config or CohortConfig()
    frames: list[pd.DataFrame] = []
    for path in paths:
        chunks = pd.read_csv(
            path,
            usecols=RAW_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
            encoding_errors="replace",
        )
        for raw_chunk in chunks:
            prepared = prepare_sba_chunk(raw_chunk, config=config, restrict_model_window=True)
            if not prepared.empty:
                frames.append(prepared)
    if not frames:
        raise ValueError("no eligible SBA loans were found")
    return pd.concat(frames, ignore_index=True)


def build_vintage_summary(
    paths: Iterable[str | Path],
    *,
    config: CohortConfig | None = None,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Aggregate five-year outcomes across every mature vintage since FY1991."""
    config = config or CohortConfig()
    summaries: list[pd.DataFrame] = []
    for path in paths:
        chunks = pd.read_csv(
            path,
            usecols=RAW_COLUMNS,
            chunksize=chunksize,
            low_memory=False,
            encoding_errors="replace",
        )
        for raw_chunk in chunks:
            prepared = prepare_sba_chunk(raw_chunk, config=config, restrict_model_window=False)
            if prepared.empty:
                continue
            summary = prepared.groupby("approval_year")["target"].agg(["size", "sum"])
            summaries.append(summary.reset_index())
    combined = pd.concat(summaries, ignore_index=True)
    output = combined.groupby("approval_year", as_index=False).agg(
        loans=("size", "sum"), charge_offs=("sum", "sum")
    )
    output["five_year_charge_off_rate"] = output["charge_offs"] / output["loans"]
    return output.sort_values("approval_year").reset_index(drop=True)


def temporal_split(
    cohort: pd.DataFrame,
    *,
    train_end_year: int = 2014,
    validation_end_year: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by approval vintage, keeping the future test period untouched."""
    train = cohort.loc[cohort["approval_year"] <= train_end_year].copy()
    validation = cohort.loc[
        cohort["approval_year"].between(train_end_year + 1, validation_end_year)
    ].copy()
    test = cohort.loc[cohort["approval_year"] > validation_end_year].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("temporal split produced an empty partition")
    if not (
        train["approval_date"].max() < validation["approval_date"].min()
        and validation["approval_date"].max() < test["approval_date"].min()
    ):
        raise RuntimeError("temporal leakage detected between partitions")
    return train, validation, test
