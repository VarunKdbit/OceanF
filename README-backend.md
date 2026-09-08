# OceanEmbed Backend (SIH26066)

Two services implementing the pipeline:

```
React (frontend)
      ↓ REST/JSON
Spring Boot  (backend-api)   — validation, persistence, orchestration, auth
      ↓ REST/JSON
FastAPI      (ml-service)    — loads OceanEmbed, runs inference
      ↓
OceanEmbed model (PyTorch)
```

## Why this split

- **backend-api (Spring Boot)** never touches the model. It validates requests,
  stores prediction jobs + results in PostgreSQL, and calls `ml-service` over HTTP.
- **ml-service (FastAPI)** never touches the database or the frontend. It only
  knows how to turn `(lat, lon, date, surface variables, depths)` into
  `(temperature per depth)`. Swapping the placeholder model for the real
  trained PyTorch model only requires editing `ml-service/app/model.py`.

## Data contract (per the research papers)

**Input** — satellite-observed surface fields on a 0.25° grid, for a given day:

| Field | Meaning | Source |
|---|---|---|
| `latitude`, `longitude` | Point on the grid | frontend map click |
| `date` | Day of the surface fields | frontend date picker |
| `sst` | Sea Surface Temperature (°C) | satellite |
| `sss` | Sea Surface Salinity (PSU) | satellite |
| `ssh` | Sea Surface Height / Sea Level Anomaly (m) | satellite altimetry |
| `wind_u`, `wind_v` | Surface wind components (m/s), optional | satellite scatterometer |
| `depths` | Requested depth levels (m), e.g. `[0,10,20,50,100,200,500]` | frontend |

**Output** — one predicted temperature per requested depth, plus metadata
(`model_version`, `grid_resolution_deg`) and, if implemented, an
`uncertainty_c` per depth.

**ARGO** floats are used only to *validate* predictions against real
in-situ subsurface measurements — they are never part of the input the
model receives at inference time, and are kept separate from training data
(GLORYS reanalysis) to avoid leakage.

## API — backend-api (port 8080)

`POST /api/v1/predictions`
```json
{
  "latitude": 12.5,
  "longitude": 75.2,
  "date": "2026-09-01",
  "regionName": "Arabian Sea",
  "surface": { "sst": 28.4, "sss": 35.1, "ssh": 0.12, "windU": 3.2, "windV": -1.1 },
  "depths": [0, 10, 20, 50, 100, 200, 500]
}
```
Returns a `PredictionResponseDTO` with `jobId`, `status`, and `predictions[]`.

`GET /api/v1/predictions/{id}` — fetch a previously computed job.
`GET /api/v1/health` — reports whether `ml-service` is reachable.

## API — ml-service (port 8000)

`POST /predict` — same field names, model-internal contract (see
`ml-service/app/schemas.py`). `GET /health`, `GET /model/info`.

## Running locally

```bash
docker compose up --build
```
This starts PostgreSQL, `ml-service` (:8000) and `backend-api` (:8080).

Without Docker:
```bash
# ml-service
cd ml-service
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# backend-api (needs a local Postgres matching application.yml, or override via env vars)
cd backend-api
mvn spring-boot:run
```

## Plugging in the real trained model

Edit `ml-service/app/model.py`:
1. Load your trained weights in `OceanEmbedModel.load()`.
2. Replace `_synthetic_profile()` in `predict()` with real inference
   (build the model's input tensor from `request.surface` + grid position,
   run forward pass, map outputs to `request.depths`).

Nothing in `backend-api` needs to change — it only talks to the stable
`/predict` HTTP contract.

## Project structure

```
oceanembed-backend/
├── docker-compose.yml
├── ml-service/              FastAPI ML service
│   ├── app/
│   │   ├── main.py          routes: /predict, /health, /model/info
│   │   ├── schemas.py       Pydantic request/response contracts
│   │   ├── model.py         OceanEmbedModel wrapper (plug real model here)
│   │   └── config.py        settings
│   ├── requirements.txt
│   └── Dockerfile
└── backend-api/             Spring Boot orchestration API
    ├── pom.xml
    ├── Dockerfile
    └── src/main/
        ├── java/com/oceanembed/backend/
        │   ├── controller/  PredictionController, HealthController
        │   ├── service/     PredictionService, FastApiClient
        │   ├── entity/      PredictionJob, PredictionResult (JPA)
        │   ├── repository/  Spring Data JPA repos
        │   ├── dto/         frontend-facing + ml-service-facing DTOs
        │   ├── config/      RestTemplate config
        │   └── exception/   GlobalExceptionHandler
        └── resources/
            ├── application.yml
            └── schema.sql
```
