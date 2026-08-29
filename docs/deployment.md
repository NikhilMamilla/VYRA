# Deploying VYRA

Two supported targets. **Local Docker Compose already satisfies the assessment's
deployment requirement** (§11: "Local Docker Compose deployment is acceptable").
The cloud path below is optional and gives a public URL.

---

## 1. Local — Docker Compose (baseline)

```bash
git clone https://github.com/NikhilMamilla/VYRA.git
cd VYRA
docker compose up --build
```

| Service | URL |
|---|---|
| Frontend | http://localhost:5173 |
| API | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health | http://localhost:8000/health |
| Metrics | http://localhost:8000/metrics |

Everything has a working default — no `.env` needed. `docker compose down -v`
stops and wipes the database + upload volumes.

The backend image bundles the inference model at `/app/model`
(`REQUIRE_ANALYZER=true`, so a broken bundle fails startup loudly). Postgres runs
as a container with no published port; the backend reaches it on the compose
network. nginx serves the SPA and proxies `/api` to the backend, so the browser
talks to one origin.

---

## 2. Cloud — Render.com (free, public URL)

Render is a good fit: it runs the backend `Dockerfile` unchanged, gives a free
managed Postgres, and hosts the SPA as a static site. Everything is described in
[`render.yaml`](../render.yaml) at the repo root.

**This project is deployed at:**

| | URL |
|---|---|
| Frontend | https://vyra-frontend.onrender.com |
| API docs | https://vyra-backend-gaig.onrender.com/docs |
| Health | https://vyra-backend-gaig.onrender.com/health |
| Metrics | https://vyra-backend-gaig.onrender.com/metrics |

### Prerequisites

- A [Render](https://render.com) account (free, GitHub sign-in).
- This repo pushed to GitHub (it already is).

### Steps

1. **Render dashboard → New → Blueprint.**
2. Connect the GitHub repo `NikhilMamilla/VYRA`. Render finds `render.yaml` and
   shows three resources: `vyra-db`, `vyra-backend`, `vyra-frontend`.
3. Click **Apply**. Render provisions the database, builds the backend Docker
   image (~3–5 min the first time), and builds the SPA.
4. When all three are live, open each web service and copy its real URL from the
   top of its page — Render appends a suffix if `vyra-backend` /
   `vyra-frontend` are already taken (e.g. `https://vyra-backend-a1b2.onrender.com`).
5. **Wire the two URLs together** (the one manual step):
   - `vyra-backend` → **Environment** → set `CORS_ORIGINS` to the real frontend
     URL → **Save** (this redeploys the backend).
   - `vyra-frontend` → **Environment** → set `VITE_API_BASE_URL` to the real
     backend URL → **Save**, then **Manual Deploy → Deploy latest commit** (Vite
     bakes the value in at build time, so it needs a rebuild).
6. Open the frontend URL. The nav bar should show the model version pulled from
   `GET /health`; upload an image.

### Notes

- **Cold starts.** Free web services sleep after 15 min idle; the first request
  then takes ~30–60 s while the container and model reload. This is normal on the
  free plan.
- **Memory.** The backend (opencv + scikit-learn + scikit-image + the model)
  sits near the 512 MB free cap; `MALLOC_ARENA_MAX=2` in `render.yaml` keeps it
  under. If the service log shows `Out of memory (used over 512Mi)`, upgrade just
  `vyra-backend` to the **Standard** instance (2 GB) or use the Fly.io path below.
- **Database.** Render's free Postgres is dropped after 30 days of the account
  being on the free plan; `DATABASE_AUTO_CREATE=true` recreates the schema on the
  next deploy. Analyses are history only — nothing else depends on persistence.
- **Auto-deploy.** Every push to `main` redeploys both services.

### Submission

Deployed URL for the submission: **https://vyra-frontend.onrender.com**
(API + Swagger docs at https://vyra-backend-gaig.onrender.com/docs).

---

## 3. Fallback — Fly.io backend + Neon Postgres (more memory, needs a card)

Use this only if the Render backend won't stay under 512 MB.

1. **Neon** ([neon.tech](https://neon.tech), free): create a project, copy the
   connection string (`postgresql://…`). VYRA rewrites the driver automatically.
2. **Backend on Fly.io:**
   ```bash
   npm i -g @flyctl/flyctl   # or: curl -L https://fly.io/install.sh | sh
   fly auth login
   fly launch --dockerfile backend/Dockerfile --no-deploy --name vyra-api
   # In the generated fly.toml set [http_service] internal_port = 8000
   fly secrets set DATABASE_URL="postgresql://…from neon…" \
     REQUIRE_ANALYZER=true ENVIRONMENT=production LOG_JSON=true \
     CORS_ORIGINS="https://<your-frontend>"
   fly deploy
   fly scale memory 1024
   ```
3. **Frontend:** deploy `frontend/` as a static site on Render, Vercel, Netlify
   or Cloudflare Pages with build `npm run build`, output `dist`, and
   `VITE_API_BASE_URL=https://vyra-api.fly.dev`.

---

## How the model is loaded after deployment

Identical in every target: `app.main:create_app` runs `load_analyzer(settings)`
during ASGI lifespan startup, which builds one `VyraAnalyzer` from `MODEL_PATH`
(the bundle baked into the image at `/app/model`) and parks it on `app.state`.
Every request reuses that instance; the CPU-bound CV + inference step runs on a
worker thread (`anyio.to_thread`) so it never blocks the event loop. `/health`
reports `analyzer.status` and `analyzer_model_version`; `REQUIRE_ANALYZER=true`
makes a missing or corrupt bundle abort startup instead of serving a degraded
API. See [architecture.md](architecture.md).
