# Data Card: SBA 7(a) FOIA Loan-Level Data

## Source

- **Publisher:** U.S. Small Business Administration, Office of Capital Access
- **Official catalogue:** <https://data.sba.gov/dataset/7a-504-foia>
- **Pinned snapshot:** 30 June 2026
- **Coverage:** four 7(a) files spanning FY1991–present
- **Raw records:** 1,961,455
- **Format:** CSV plus an XLSX data dictionary
- **Catalogue licence:** U.S. Government Works/Public Domain
- **Refresh cadence:** quarterly

Every official URL, byte size, record count, and SHA-256 checksum is pinned in
`data/sources.json`. Raw files are downloaded from SBA and are not redistributed in this
repository.

## Unit of observation

One approved SBA 7(a) loan. The data includes origination characteristics, current lifecycle
status, payoff or charge-off information, geography, lender information, and borrower business
characteristics.

## Intended use in this repository

Retrospective research on five-year charge-off prediction among approved, disbursed loans;
temporal validation; calibration; threshold analysis; drift monitoring; and model explanation.

## Features used

Only fields plausibly available at or near first disbursement are used:

- approved and guaranteed amounts;
- loan term and initial interest rate;
- processing method and fixed/variable indicator;
- lender identifier and lender state;
- two-digit NAICS sector and project state;
- business type and harmonised business-age band;
- reported jobs supported, collateral indicator, and revolver indicator.

Amounts and jobs are log transformed. The guarantee percentage is derived from the two approval
amounts. Changing business-age labels are harmonised into stable bands.

## Explicitly excluded

- Borrower and lender street addresses, borrower name, city, ZIP code, and franchise name
- Current status, paid-in-full date, charge-off date, and charge-off amount as features
- Secondary-market indicator because it can be updated after origination
- Cancelled and undisbursed loans
- Loans without a complete five-year observation window

## Known limitations

- The population includes approved loans only; rejected applications are absent.
- Current active-loan status is partially suppressed as `EXEMPT` under FOIA, although the fixed
  mature horizon permits outcome construction without treating current status as a feature.
- Economic conditions, SBA programmes, data systems, and field vocabularies changed over time.
- Jobs supported are lender-reported estimates that SBA states it does not audit or validate.
- The dataset has no protected-class attributes, applicant financial statements, credit bureau
  histories, or payment sequences.
- A charge-off is an operational outcome, not a complete measure of borrower welfare or social
  value.

## Privacy and ethics

Public availability does not remove the obligation to minimise personal data. The ingestion
code never loads direct borrower identifiers. Published artifacts are aggregate metrics only;
no row-level probabilities or reconstructed identifiers are retained.
