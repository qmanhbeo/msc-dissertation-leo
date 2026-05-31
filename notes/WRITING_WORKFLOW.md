# Writing Workflow

Canonical build:

```bash
python main.py --build-pdf --overwrite
```

Canonical artifact:
- `outputs/dissertation.pdf`

The manuscript source reads generated tables from `outputs/tables/` and figures from `outputs/figures/`.
Versioned build history under `writing/builds/` is no longer part of the project contract.
