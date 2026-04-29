# Writing Workflow

Last updated: 2026-04-10

---

## Build the dissertation PDF

```bash
cd /home/manh/study-BHam/dissertation/writing
./build_versioned_pdf.sh
```

This produces:
- `writing/dissertation.pdf`
- `writing/builds/dissertation_YYYY-MM-DD_HHMMSS_TZ.pdf`

## Inspect recent build artifacts

```bash
ls -t /home/manh/study-BHam/dissertation/writing/builds/dissertation_*.pdf | head -n 5
```

## Send the PDF to a phone over Tailscale

Check the target device name first:

```bash
tailscale status
```

Exact successful command used on 2026-04-10:

```bash
tailscale file cp /home/manh/study-BHam/dissertation/writing/builds/dissertation_2026-04-10_183449_BST.pdf iphone-14-pm:
```

Reusable version that sends the latest build:

```bash
latest_pdf="$(ls -t /home/manh/study-BHam/dissertation/writing/builds/dissertation_*.pdf | head -n 1)"
tailscale file cp "$latest_pdf" iphone-14-pm:
```

Notes:
- The target device must be online in the same tailnet.
- If the phone name changes, replace `iphone-14-pro-max` with the current Tailscale device name.
- Tailscale stores received files in the iOS Files app under the Tailscale transfer location.
