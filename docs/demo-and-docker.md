# Demo and Docker

## The demo page

`demo/` is a self-contained, pre-rendered presentation of the results: the tunnel
reconstruction, the three-candidate comparison, the per-class table, playbacks of all three
candidates on the same simulated drive, the real-bag run, and a 3D view of what the 256-d latent
preserves. No model runs in the browser. Every frame is rendered offline by this repo's scripts and
embedded, and each playback states the checkpoint it came from.

It works three ways:

```bash
# 1. as a folder: open demo/index.html in a browser (no server, no Python, no network)

# 2. served by nginx
docker compose up -d demo                 # -> http://localhost:8080 (DEMO_PORT to change)

# 3. regenerated from checkpoints and data
python demo/build_demo.py --all
```

The page has to work from `file://`, where an opaque origin blocks `fetch()` and ES modules. So
every payload is a classic script that registers itself (`DEMO.register(kind, id, payload)`), and
since three.js's PLY/PCD loaders cannot run there, geometry is pre-quantized to base64 and decoded
into typed arrays by hand.

| file | role |
|---|---|
| `demo/build_demo.py` | exporter command line |
| `demo/sources.json` | what to export: checkpoints, recordings, geometry |
| `demo/metrics_template.json` | written text; numbers are filled in from measured JSON |
| `demo/payloads/manifest.js` | page structure, sections, thesis metadata |
| `demo/schema.py` | validation, standard library only |

```bash
python demo/build_demo.py --check                       # validate; no torch, no data needed
python demo/build_demo.py --report                      # per-file sizes against the budget
python demo/build_demo.py --only sim_player:pointnet2_sim
```

`--check` asserts what the page cannot check by itself: every telemetry field a payload declares is
present in every frame, referenced frames and figures exist, timestamps never go backwards, and no
resource loads from a remote origin (fonts are self-hosted, so the page works offline and makes no
third-party requests).

A missing input is skipped and the build carries on. Checkpoints, `scenario_data/`, the bags and
the `.ply` geometry are gitignored, so a fresh clone rebuilds what it can and keeps the committed
payloads for the rest.

Adding a fourth candidate is a data change: one entry in `demo/sources.json`, one in
`demo/metrics_template.json` and one tab in `demo/payloads/manifest.js`.

The tunnel section currently uses `noslil.ply`, a stand-in scan. To swap in a cleaner capture,
replace the file and run `python demo/build_demo.py --only tunnel`.

## Docker

```bash
docker compose up -d demo                     # nginx + pre-rendered payloads, no GPU
docker compose --profile pipeline build       # CUDA image, only to regenerate payloads
docker compose --profile pipeline run --rm pipeline python demo/build_demo.py --all
```

- `demo` is small, with no GPU requirement and no data mounts, so `docker compose up` works on
  any laptop. It serves the page with nginx because `python -m http.server` has no HTTP Range
  support, and seeking in the videos fails without it.
- `pipeline` sits behind a profile so it never starts by accident: on a machine without the
  NVIDIA container toolkit its GPU reservation is a hard error. It installs
  `requirements-pipeline.txt` and a CUDA build of PyTorch (`TORCH_CUDA`, default `cu130`; pass
  `--build-arg TORCH_CUDA=cu121` for an older driver). Data stays out of the image: it is
  bind-mounted at run time, and the root `.dockerignore` keeps the bags out of the build context.
  On a clone without the root `.ply` files, comment out their two mounts in `docker-compose.yml`,
  or Docker creates empty directories in their place.

The images have not been built and tested yet; see [continuing.md](continuing.md).
