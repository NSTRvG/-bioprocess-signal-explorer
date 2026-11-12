# BioProcess Excursion Monitor

**A portfolio project for exploring bioprocess signals and making unusual patterns easier to inspect.**

The project starts with a practical question: how can a CSV of process measurements become a useful view of potential excursions without confusing a statistical flag with a biological diagnosis? The answer is a local Streamlit app that prepares time-series data, calculates causal rolling bands, and groups out-of-band observations for review.

> **Scope:** This is an educational data-analysis and software-development demo. It is not a validated biological model, process alarm, or tool for operational, quality, or safety decisions.

## Project story

1. **Start with traceable data.** The app includes a reproducible synthetic series and a reduced sample from the public IndPenSim benchmark. The benchmark sample records its provenance and license, and keeps labels separate from measurements.
2. **Prepare signals carefully.** Input timestamps must be valid and unique. Rows are sorted chronologically, mixed numeric/text measurement columns are rejected, and only short internal gaps are interpolated.
3. **Build a causal reference.** Each variable can use a robust band—median ± `k × 1.4826 × MAD`—or a classic band—mean ± `k` standard deviations. Bands use observations in the timestamp-aligned interval `[t - window, t)`, so the current reading cannot influence its own reference.
4. **Turn points into reviewable events.** Out-of-band observations are grouped by direction and time proximity. The app displays signals, bands, and events without attempting to assign a cause.
5. **Compare results and state limits.** Automated tests check software behavior, while a separate report summarizes results on the bundled data. Neither is a substitute for biological or industrial validation.

## What this project demonstrates

- Time-series analysis with Python, pandas, and NumPy.
- Robust statistics and causal rolling bands.
- Input validation, bounded missing-data handling, and event grouping.
- An interactive Streamlit interface with Plotly visualizations.
- Attention to units, provenance, licensing, and separation of signals from labels.
- Transparent evaluation, including modest results and known limitations.

## What the data show

### Synthetic series

Run `py scripts\generate_synthetic.py` to generate a 180-hour series sampled every five minutes, with three injected anomaly intervals. With a 60-minute window, `k=3`, and a 10-point baseline, the detector flagged 6/6 pH, 5/30 dissolved-oxygen, and 6/120 temperature samples inside the injected intervals. It also flagged samples outside those labels: 133 for pH, 83 for dissolved oxygen, and 89 for temperature.

This illustrates both detection and its limits: labels cover only injected anomalies, so these counts do not estimate a general false-alarm rate.

The synthetic series is not a real culture or a physiological or mass-balance model. Its data dictionary is explicit: pH is dimensionless; temperature is in °C; dissolved oxygen uses an arbitrary signal scale (not mg/L or calibrated percent saturation); and biomass uses arbitrary model units.

### IndPenSim benchmark

The repository includes batch 95: 1,055 rows, a median sampling interval of 12 minutes, pH, dissolved oxygen (mg/L), temperature (K), and a separate process-level fault-label file. IndPenSim is a **mathematical simulation** of industrial-scale penicillin fermentation, not a collection of plant measurements. UTC timestamps in the sample are derived from process time and anchored at 2000; they are not actual measurement dates.

With a 180-minute window, `k=3`, and a 10-point baseline, the union of flags across the three variables compared pointwise with the process-level `Fault reference` label produced:

| True positives | False positives | False negatives | Precision | Recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 48 | 223 | 140 | 0.177 | 0.255 | 0.209 |

This is a weak exploratory result, not a detector pass. The simulator label identifies process-level faults, not the affected variable at each time point, so this comparison cannot precisely measure variable-specific fault detection.

## Run locally

Requires Python 3.10 or later.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
py scripts\generate_synthetic.py
py -m streamlit run app.py
```

The app starts with the synthetic series selected. Selecting IndPenSim sets the initial rolling window to 180 minutes, better suited to its cadence and the default 10-point baseline. You can also upload a CSV with a `timestamp` column and one or more numeric measurement columns. Timestamps must be unique and parseable; columns mixing numeric measurements with invalid text are rejected. Only internal missing runs whose bounding observations are no more than 30 minutes apart are interpolated.

To retrieve the public sample again:

```powershell
py scripts\download_indpensim_sample.py --batch-reference 95
```

The script checks provider metadata and license, then uses HTTP range requests to retrieve one batch without downloading the full source CSV (about 2.57 GB). To process batches from a full source file you already have:

```powershell
py scripts\preprocess_indpensim.py data\full\100_Batches_IndPenSim_V3.csv `
  --batch-ids 1,95 --time-unit h
```

## Reproducible checks

```powershell
py -m pytest -q
py scripts\validate_datasets.py
```

The tests cover software behavior, including causal bands with irregular sampling. The validation report is written to `data\sample\validation_report.json`. The current suite has 23 passing tests. Software checks and descriptive results do not validate biological performance or plant readiness.

## Deliberate limitations

- No experimental laboratory or plant data has been used for validation.
- The project does not define an organism, strain, medium, cultivation phase, or mechanistic model.
- Variables are analyzed independently; correlations and multivariate dynamics are not modeled.
- Windows, `k` values, and baseline sizes are exploratory parameters, not process limits.
- Severity is a heuristic, not a validated risk classification.
- Results do not diagnose causes or recommend actions.
- Do not connect this prototype to equipment or use it as the sole basis for process decisions.

The portfolio value is in showing the complete workflow—from data provenance through analysis, interface, and critical evaluation—and in being explicit about what the results **do not** establish.

## Sources and data provenance

- Goldrick, S., Stefan, A., Lovett, D., Montague, G., and Lennox, B. (2015). “The development of an industrial-scale fed-batch fermentation simulation.” *Journal of Biotechnology*, 193, 70–82. Cited in the dataset record.
- Goldrick, S. et al. (2019). “Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process.” *Computers & Chemical Engineering*, 130, 106471. [doi:10.1016/j.compchemeng.2019.05.037](https://doi.org/10.1016/j.compchemeng.2019.05.037).
- Goldrick, S. et al. *Data for: Modern day monitoring and control challenges outlined on an industrial-scale benchmark fermentation process*. Mendeley Data, version 1. [doi:10.17632/npt257bjxn.1](https://doi.org/10.17632/npt257bjxn.1). Licensed under CC BY 4.0.

Attribution, transformations, and sample-file hashes are recorded in `data\sample\indpensim_batch_095_manifest.json`. The provider's SHA-256 for the complete source is included as metadata but was not verified locally because the complete file was not downloaded.

## License

The project's original source code, documentation, and synthetic data are licensed under MIT; see `LICENSE`. IndPenSim measurements and fault labels are third-party material under CC BY 4.0 and are **not** covered by the MIT License. Retain attribution and indicate transformations when redistributing the sample.
