# Increment

Compare treatment with control from a dataframe or SQL warehouse. Increment
reports effects, uncertainty intervals and the checks that qualify each result.
Use your existing assignments or [assign units locally](guides/assign-experiments.md).
Increment does not store events or host a feature-flag service.

**Start with the [dataframe quickstart](guides/quickstart.md).** It runs without
YAML or a warehouse connection and explains the resulting estimate.

## Install

For the latest prerelease and the Polars quickstart:

```bash
python -m pip install --pre increment polars
```

Requires Python 3.12 or newer. Core/dataframe workflows are tested on
3.12–3.15; warehouse workflows on 3.12–3.14. DuckDB-backed extras on 3.15
need a compatible DuckDB wheel.

Install extras for the task you need:

| Task | Extra |
|---|---|
| Local DuckDB examples | `increment[demo]` |
| SQL warehouse | `increment[warehouse]` plus an Ibis backend |
| Formatted notebook tables | `increment[tables]` |
| HTML experiment dashboard | `increment[dashboard]` |

For example, `python -m pip install --pre "increment[demo]"` installs the local
warehouse dependencies. For Snowflake, install
`"increment[warehouse]" "ibis-framework[snowflake]"` instead. The
[warehouse guide](guides/data-model.md#other-warehouses) covers verified
backends, query translation and required permissions.

## Quickstart

Run the [complete dataframe example](guides/quickstart.md), then use
[Reading results](guides/reading-results.md) to distinguish an available
estimate from evidence strong enough to act on.

Prefer an interactive walkthrough? Open the
[dataframe notebook](examples/analysis_from_a_dataframe.md).

## Analyze from your warehouse

The [warehouse setup guide](guides/data-model.md) walks through source tables,
assignment, metrics and YAML definitions. The
[warehouse notebook](examples/analysis_from_a_warehouse.md) runs the complete
workflow against a local DuckDB database.

## What do you want to do next?

| Your question | Start here |
|---|---|
| Which analysis fits my experiment? | [Choose a method](guides/choose-a-method.md) |
| What does this result mean? | [Reading results](guides/reading-results.md) |
| How much data do I need? | [Power analysis](guides/power-analysis.md) |
| Can pre-experiment data help? | [CUPED](guides/cuped.md) |
| Can I act while the test is running? | [Sequential inference](guides/sequential-inference.md) |
| Why was my request refused? | [Find the next step](guides/refusals.md) |

Before relying on an analysis, read its [assumptions and limitations](limitations.md).
Methods have different foundations: exact constructions, asymptotic
approximations, or empirical calibration. [Validation](validation.md) separates what has
been checked from what has not. The [API reference](api.md) covers exact calls.
