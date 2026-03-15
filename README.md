# SignalShield: Real-Time Hybrid Fraud Intelligence (Demo)

This project is a compact demo of a **bank-style fraud detection system** with:

- **Transformer-based sequence model** for fraud probability
- **Rules engine** for known risky patterns
- **FastAPI backend** that fuses rules + model into a single risk score and confidence interval
- **User web app** for registration, login, and transaction submission
- **Analyst web app** for reviewing borderline (medium-risk) transactions

> In a real production system you would add Kafka, Redis, a real database, and proper auth. This repo focuses on a clean, end‑to‑end demo you can run locally and extend.

## Project structure

- `backend/`
  - `train_transformer.py` – trains a simple Transformer encoder on transaction sequences
  - `app/main.py` – FastAPI app exposing registration, login, transaction scoring, and admin review APIs
  - `requirements.txt` – Python dependencies
- `apps/user/`
  - `index.html`, `styles.css`, `script.js` – user portal to register, login, and submit transactions
- `apps/admin/`
  - `index.html`, `styles.css`, `script.js` – analyst console to review “borderline” cases (decision = `review`)

## 1. Python environment

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate  # on Windows

pip install -r backend/requirements.txt
```

## 2. Prepare training data

The Transformer training script expects a CSV at `data/transactions.csv` with columns:

- `device_id`
- `timestamp`
- `country` (geo, e.g. US, CA, GB, DE) (ISO string, e.g. `2024-01-01T12:34:56Z`)
- `amount`
- `channel` (e.g. `online`, `pos`, `atm`)
- `is_fraud` (0 or 1)

You can:

- Use a public fraud dataset and map/rename columns, or
- Create a synthetic CSV for testing.

## 3. Train the Transformer model (optional but recommended)

From the project root (with venv activated):

```bash
python -m backend.train_transformer --csv data/transactions.csv --seq-len 20 --epochs 5
```

This will save the best model to:

- `backend/artifacts/transformer_fraud.pt`

If this file is missing, the backend will still run and use only the rules engine.

## 4. Run the FastAPI backend

From the project root:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

Key endpoints:

- `POST /api/register`
- `POST /api/login`
- `POST /api/transaction/score`
- `GET  /api/admin/review-queue`
- `POST /api/admin/review/{review_id}`
- `GET  /health`

Each transaction score response includes:

- `risk_score` in \[0, 1\]
- `confidence_low`, `confidence_high` (simple confidence interval)
- `decision`: `allow`, `step_up`, `review`, or `block`
- `reasons`: human-readable explanation strings
- `review_id`: present when decision = `review`

## 5. Run the frontends

The frontends are static HTML/JS. You can open them directly in a browser or serve them via a simple HTTP server.

### Option A: open files directly

- Open `apps/user/index.html` in your browser for the **user portal**
- Open `apps/admin/index.html` in your browser for the **analyst console**

Make sure the backend is running at `http://localhost:8000` (frontends call that URL).

### Option B: simple static server (optional)

From the project root:

```bash
cd apps
python -m http.server 5173
```

Then navigate to:

- `http://localhost:5173/user/` for the user app
- `http://localhost:5173/admin/` for the analyst app

## 6. Typical demo flow

1. **Start backend**: `uvicorn backend.app.main:app --reload --port 8000`
2. (Optional) **Train model** to enable Transformer scoring.
3. Open the **user portal**:
   - Register a new user
   - Log in from a device/IP
   - Submit transactions of different sizes, channels, and locations
4. Watch returned:
   - `risk_score`
   - confidence interval
   - final decision and reasons
5. For **medium risk** (`decision = "review"`), a `review_id` is assigned and the transaction appears in the **analyst console**.
6. Open the **analyst console**, refresh the queue, and:
   - Mark items as `allow`, `block`, or `step_up`
   - The queue updates and decisions are logged.

## 7. Next steps / extensions

- Replace in‑memory stores with PostgreSQL + Redis
- Add proper auth / JWT tokens
- Log all scored events to a database for offline retraining
- Extend the Transformer to ingest richer sequences (login, password reset, add‑payee, transfer, etc.)
- Add graph features or a separate graph service to detect mule networks and rings

This setup already gives you a strong **portfolio/hackathon‑grade** demonstration: hybrid rules + Transformer, risk fusion, confidence interval, and a human‑in‑the‑loop analyst workflow. 

