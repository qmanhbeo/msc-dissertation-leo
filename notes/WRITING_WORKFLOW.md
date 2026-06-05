# Writing Workflow

Canonical build:

```bash
python main.py --build-pdf --overwrite
```

Canonical artifact:
- `outputs/dissertation.pdf`

The manuscript source reads canonical generated tables from `outputs/tables/` and figures from `outputs/figures/`.
When the optional genre-adjustment stage is run, the appendix also reads robustness tables and figures from `outputs/robustness/genre_adjustment/`.
Versioned build history under `writing/builds/` is no longer part of the project contract.
