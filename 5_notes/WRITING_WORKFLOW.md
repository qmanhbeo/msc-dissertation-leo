# Writing Workflow

Canonical build:

```bash
python main.py --build-pdf --overwrite
```

Canonical artifact:
- `outputs/dissertation.pdf`
The manuscript source reads canonical generated tables from `outputs/main/tables/` and figures from `outputs/main/figures/`.

When the optional register-adjustment stage is run, the appendix also reads tables and figures from `4_outputs/appendix/d_register_adjustment/`.
Versioned build history under `writing/builds/` is no longer part of the project contract.
