from __future__ import annotations

import pandas as pd
import pytest

from credit_risk_benchmark.data import (
    LEAKAGE_COLUMNS,
    MODEL_FEATURES,
    prepare_sba_chunk,
    temporal_split,
)


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "AsOfDate": "2026-06-30",
        "Program": "7A",
        "LocationID": 123,
        "BankState": "CA",
        "GrossApproval": 100_000,
        "SBAGuaranteedApproval": 75_000,
        "ApprovalDate": "2018-01-01",
        "ApprovalFY": 2018,
        "FirstDisbursementDate": "2018-02-01",
        "ProcessingMethod": "7a General",
        "InitialInterestRate": 7.5,
        "FixedorVariableInterestInd": "V",
        "TermInMonths": 120,
        "NaicsCode": 541511,
        "ProjectState": "CA",
        "BusinessType": "CORPORATION",
        "BusinessAge": "Existing, 5 or more years",
        "LoanStatus": "P I F",
        "PaidInFullDate": "2024-01-01",
        "ChargeOffDate": None,
        "GrossChargeOffAmount": 0,
        "RevolverStatus": "N",
        "JobsSupported": 4,
        "CollateralInd": "Y",
        "SoldSecMrktInd": "Y",
    }
    row.update(overrides)
    return row


def test_five_year_target_and_leakage_quarantine() -> None:
    raw = pd.DataFrame(
        [
            _row(LoanStatus="CHGOFF", ChargeOffDate="2020-01-01"),
            _row(LocationID=456, ChargeOffDate="2025-01-01", LoanStatus="CHGOFF"),
        ]
    )

    prepared = prepare_sba_chunk(raw)

    assert prepared["target"].tolist() == [1, 0]
    assert set(MODEL_FEATURES).issubset(prepared.columns)
    assert LEAKAGE_COLUMNS.isdisjoint(prepared.columns)


def test_immature_cancelled_and_missing_chargeoff_date_are_excluded() -> None:
    raw = pd.DataFrame(
        [
            _row(ApprovalDate="2024-01-01", ApprovalFY=2024, FirstDisbursementDate="2024-02-01"),
            _row(LoanStatus="CANCLD"),
            _row(LoanStatus="CHGOFF", ChargeOffDate=None),
        ]
    )

    assert prepare_sba_chunk(raw).empty


def test_temporal_split_is_strictly_ordered() -> None:
    raw = pd.DataFrame(
        [
            _row(ApprovalDate="2014-01-01", ApprovalFY=2014, FirstDisbursementDate="2014-02-01"),
            _row(ApprovalDate="2016-01-01", ApprovalFY=2016, FirstDisbursementDate="2016-02-01"),
            _row(ApprovalDate="2019-01-01", ApprovalFY=2019, FirstDisbursementDate="2019-02-01"),
        ]
    )
    cohort = prepare_sba_chunk(raw)

    train, validation, test = temporal_split(cohort)

    assert train["approval_year"].tolist() == [2014]
    assert validation["approval_year"].tolist() == [2016]
    assert test["approval_year"].tolist() == [2019]


def test_missing_schema_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing required SBA columns"):
        prepare_sba_chunk(pd.DataFrame({"LoanStatus": ["PIF"]}))
