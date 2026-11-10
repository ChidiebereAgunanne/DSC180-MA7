# Bikeshare demand capstone

Reference repository for the DSC 180A methodology lesson on agentic
programming. Forecasts daily Capital Bikeshare ridership (`cnt`) for
Washington, DC, 2011-2012, with a leakage trap in the data.

## Data

Daily aggregation of the UCI Bike Sharing Dataset (Fanaee-T and Gama, 2013),
via the processed mirror used in Molnar's *Interpretable Machine Learning*:

    https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv

The mirror differs from the UCI file. See `src/config.py` for the list.
`data/raw/bike_daily.csv` is a checked-in copy so the pipeline runs offline;
it is immutable (writes are denied in `.claude/settings.json`).

## Run

    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    make all        # data -> features -> train, writes reports/metrics.json
    make test       # 14 tests

`python -m src.leakage_demo` reproduces the R^2 = 1.000 result from the lesson.

## Layout

See `CLAUDE.md` for the standing rules and `src/config.py` for every constant.
`.claude/` holds the hook, permissions, reviewer subagent and `new-feature`
skill the lesson builds in Steps 7-8. `agent/audit.py` is the Agent SDK batch
auditor from Part 4. Record methodological decisions in `docs/DECISIONS.md`.
