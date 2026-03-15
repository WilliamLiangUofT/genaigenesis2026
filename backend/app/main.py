from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.train_transformer import FraudTransformer, country_to_id


class Decision(str, Enum):
    allow = "allow"
    step_up = "step_up"
    review = "review"
    block = "block"


class ReviewState(str, Enum):
    pending = "pending"
    resolved = "resolved"


class UserRegister(BaseModel):
    username: str
    password: str
    country: str = "US"


class UserLogin(BaseModel):
    username: str
    password: str
    device_id: str
    ip_address: str
    location: str = "US"


class TransactionRequest(BaseModel):
    username: str
    amount: float
    merchant: str
    channel: str = Field("online", description="e.g. online, pos, atm")
    location: str = "US"
    device_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictionResponse(BaseModel):
    risk_score: float
    confidence_low: float
    confidence_high: float
    decision: Decision
    reasons: List[str]
    review_id: Optional[int] = None


class ReviewItem(BaseModel):
    id: int
    username: str
    amount: float
    merchant: str
    channel: str
    location: str
    created_at: datetime
    risk_score: float
    reasons: List[str]
    status: ReviewState = ReviewState.pending
    final_decision: Optional[Decision] = None
    resolved_at: Optional[datetime] = None
    comment: Optional[str] = None


class ReviewDecisionRequest(BaseModel):
    decision: Decision
    comment: Optional[str] = None


app = FastAPI(title="SignalShield Fraud Detection API")

# Allow browser frontends (React or plain HTML) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_PATH = Path("backend") / "artifacts" / "transformer_fraud.pt"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model: Optional[FraudTransformer] = None


def load_model() -> Optional[FraudTransformer]:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        # For demo: backend can run without a trained model
        return None
    # Feature dim must match training script (4: amount, is_night, channel_id, country_id)
    model = FraudTransformer(feature_dim=4)
    state = torch.load(MODEL_PATH, map_location=_device)
    model.load_state_dict(state)
    model.to(_device)
    model.eval()
    _model = model
    return _model


# In-memory demo stores. Replace with Postgres/Redis in production.
USERS: Dict[str, Dict] = {}
FAILED_LOGINS: Dict[str, List[datetime]] = {}
LAST_LOGIN: Dict[str, Dict] = {}
REVIEW_ITEMS: Dict[int, ReviewItem] = {}
_review_counter = 0


def add_failed_login(username: str):
    FAILED_LOGINS.setdefault(username, []).append(datetime.utcnow())


def recent_failed_login_count(username: str, minutes: int = 10) -> int:
    cutoff = datetime.utcnow() - timedelta(minutes=minutes)
    return sum(1 for t in FAILED_LOGINS.get(username, []) if t >= cutoff)


def rule_engine_for_login(login: UserLogin) -> (float, List[str]):
    """
    Simple rule scoring for login / account takeover risk.
    Returns (rule_score in [0,1], reasons)
    """
    reasons = []
    score = 0.0

    # New device or IP change
    last = LAST_LOGIN.get(login.username)
    if last:
        if login.device_id != last.get("device_id"):
            score += 0.2
            reasons.append("New device detected")
        if login.ip_address != last.get("ip"):
            score += 0.2
            reasons.append("New IP address detected")
        if login.location != last.get("location"):
            score += 0.2
            reasons.append("New location detected")

    # Many failed logins recently
    failed_10m = recent_failed_login_count(login.username, minutes=10)
    if failed_10m >= 5:
        score += 0.4
        reasons.append("5+ failed logins in last 10 minutes")
    elif failed_10m >= 3:
        score += 0.2
        reasons.append("3+ failed logins in last 10 minutes")

    return min(score, 1.0), reasons


def rule_engine_for_transaction(tx: TransactionRequest) -> (float, List[str]):
    """
    Simple rule-based transaction risk scoring.
    """
    reasons = []
    score = 0.0

    # High amount thresholds
    if tx.amount >= 10000:
        score += 0.5
        reasons.append("Very large transaction amount")
    elif tx.amount >= 2000:
        score += 0.3
        reasons.append("Large transaction amount")

    # Night-time transactions
    hour = tx.timestamp.hour
    if hour < 6 or hour > 22:
        score += 0.2
        reasons.append("Transaction at unusual night-time hour")

    # Risky channel example
    if tx.channel.lower() == "online":
        score += 0.1
        reasons.append("Online channel")

    # Simple risky country example
    if tx.location.upper() not in ("US", "CA", "EU"):
        score += 0.2
        reasons.append("High-risk geography")

    return min(score, 1.0), reasons


def model_score_for_transaction(tx: TransactionRequest) -> (Optional[float], List[str]):
    """
    Calls the Transformer model if available.
    Builds a single-step sequence using [amount, is_night, channel_id, country_id].
    """
    model = load_model()
    if model is None:
        # No model deployed yet
        return None, ["Model not available; using rules only"]

    is_night = 1.0 if (tx.timestamp.hour < 6 or tx.timestamp.hour > 22) else 0.0
    channel_map = {"online": 0.0, "pos": 1.0, "atm": 2.0}
    ch_id = channel_map.get(tx.channel.lower(), 0.0)
    country_id = float(country_to_id(tx.location))

    feat = torch.tensor([[tx.amount, is_night, ch_id, country_id]], dtype=torch.float32)  # (1, 4)
    feat = feat.unsqueeze(1)  # (batch=1, seq_len=1, feature_dim=4)
    feat = feat.to(_device)

    with torch.no_grad():
        logits = model(feat)
        probs = torch.softmax(logits, dim=-1)[0]  # (2,)
        fraud_prob = float(probs[1].item())

    return fraud_prob, ["Transformer model fraud probability used"]


def fuse_scores(
    rule_score: float,
    model_score: Optional[float],
    extra_rule_reasons: List[str],
    model_reasons: List[str],
) -> (float, float, float, Decision, List[str], bool):
    """
    Combine rules + model into final risk score and decision.
    Returns:
        risk_score, ci_low, ci_high, decision, reasons, needs_review
    """
    reasons = []
    reasons.extend(extra_rule_reasons)
    reasons.extend(model_reasons)

    if model_score is None:
        # Only rules
        risk = rule_score
        ci_low, ci_high = max(0.0, risk - 0.1), min(1.0, risk + 0.1)
    else:
        # Simple weighted average fusion
        risk = 0.6 * model_score + 0.4 * rule_score

        # Crude "confidence interval": narrower when rules and model agree,
        # wider when they disagree.
        disagreement = abs(model_score - rule_score)
        base_width = 0.15 + 0.35 * disagreement
        ci_low = max(0.0, risk - base_width)
        ci_high = min(1.0, risk + base_width)

    # Decision thresholds
    if risk < 0.2:
        decision = Decision.allow
        needs_review = False
    elif risk < 0.4:
        decision = Decision.step_up
        needs_review = False
    elif risk < 0.7:
        decision = Decision.review
        needs_review = True
    else:
        decision = Decision.block
        needs_review = False

    return risk, ci_low, ci_high, decision, reasons, needs_review


@app.post("/api/register")
def register(user: UserRegister):
    if user.username in USERS:
        raise HTTPException(status_code=400, detail="Username already exists")
    USERS[user.username] = {
        "username": user.username,
        "password": user.password,  # Never store plaintext in real systems.
        "country": user.country,
        "created_at": datetime.utcnow(),
    }
    return {"message": "Registered successfully"}


@app.post("/api/login")
def login(body: UserLogin):
    user = USERS.get(body.username)
    if not user or user["password"] != body.password:
        add_failed_login(body.username)
        raise HTTPException(status_code=401, detail="Invalid credentials")

    rule_score, reasons = rule_engine_for_login(body)

    # Very simple login decisioning: just back a risk score & reasons
    risk = min(rule_score, 1.0)
    ci_low, ci_high = max(0.0, risk - 0.1), min(1.0, risk + 0.1)

    LAST_LOGIN[body.username] = {
        "device_id": body.device_id,
        "ip": body.ip_address,
        "location": body.location,
        "timestamp": datetime.utcnow(),
    }

    return {
        "risk_score": risk,
        "confidence_low": ci_low,
        "confidence_high": ci_high,
        "reasons": reasons,
    }


@app.post("/api/transaction/score", response_model=PredictionResponse)
def score_transaction(tx: TransactionRequest):
    if tx.username not in USERS:
        raise HTTPException(status_code=404, detail="Unknown user")

    # Require a recent successful login before transactions
    last = LAST_LOGIN.get(tx.username)
    if not last:
        raise HTTPException(status_code=401, detail="Login required to submit transactions")
    if last.get("timestamp") < datetime.utcnow() - timedelta(minutes=30):
        raise HTTPException(status_code=401, detail="Login session expired, please log in again")

    rule_score, rule_reasons = rule_engine_for_transaction(tx)
    model_score, model_reasons = model_score_for_transaction(tx)

    risk, ci_low, ci_high, decision, reasons, needs_review = fuse_scores(
        rule_score=rule_score,
        model_score=model_score,
        extra_rule_reasons=rule_reasons,
        model_reasons=model_reasons,
    )

    review_id: Optional[int] = None
    if needs_review:
        global _review_counter
        _review_counter += 1
        review_id = _review_counter
        REVIEW_ITEMS[review_id] = ReviewItem(
            id=review_id,
            username=tx.username,
            amount=tx.amount,
            merchant=tx.merchant,
            channel=tx.channel,
            location=tx.location,
            created_at=datetime.utcnow(),
            risk_score=risk,
            reasons=reasons,
        )

    return PredictionResponse(
        risk_score=risk,
        confidence_low=ci_low,
        confidence_high=ci_high,
        decision=decision,
        reasons=reasons,
        review_id=review_id,
    )


@app.get("/api/admin/review-queue", response_model=List[ReviewItem])
def get_review_queue():
    # Only pending items are shown in the active review queue
    return [item for item in REVIEW_ITEMS.values() if item.status == ReviewState.pending]


@app.post("/api/admin/review/{review_id}")
def decide_review(review_id: int, body: ReviewDecisionRequest):
    item = REVIEW_ITEMS.get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")

    item.status = ReviewState.resolved
    item.final_decision = body.decision
    item.resolved_at = datetime.utcnow()
    item.comment = body.comment
    REVIEW_ITEMS[review_id] = item

    return {
        "message": "Decision recorded",
        "review_id": review_id,
        "decision": body.decision,
        "comment": body.comment,
    }


@app.get("/api/transaction/review-status/{review_id}", response_model=ReviewItem)
def get_review_status(review_id: int):
    item = REVIEW_ITEMS.get(review_id)
    if not item:
        raise HTTPException(status_code=404, detail="Review item not found")
    return item


@app.get("/health")
def health():
    return {"status": "ok"}