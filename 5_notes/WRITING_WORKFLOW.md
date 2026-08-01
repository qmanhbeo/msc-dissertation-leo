# Writing Workflow

Canonical build:

```bash
python main.py --build-pdf --overwrite
```

Canonical artifact:
- `outputs/dissertation.pdf`
The manuscript source reads canonical generated tables from `outputs/main/tables/` and figures from `outputs/main/figures/`.

When the optional register-adjustment stage is run, its canonical outputs (iterative diagnostic, decomposition) are read from `4_outputs/{model}/tables/`. The old appendix script `2_appendix/f_register_adjustment.py` was folded into the canon flow and deleted.
Versioned build history under `writing/builds/` is no longer part of the project contract.
