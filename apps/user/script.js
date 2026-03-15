const API_BASE = "http://localhost:8000";

let currentUser = null;

function prettyPrint(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

async function apiCall(path, payload, outputEl) {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || `Request failed with ${res.status}`);
    }
    prettyPrint(outputEl, data);
    return data;
  } catch (err) {
    prettyPrint(outputEl, { error: err.message || String(err) });
    throw err;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const regForm = document.getElementById("register-form");
  const regOut = document.getElementById("register-output");

  const loginForm = document.getElementById("login-form");
  const loginOut = document.getElementById("login-output");

  const txForm = document.getElementById("tx-form");
  const txOut = document.getElementById("tx-output");

  const txResult = document.getElementById("tx-result");
  const txRisk = document.getElementById("tx-risk");
  const txCi = document.getElementById("tx-ci");
  const txDecision = document.getElementById("tx-decision");
  const txReviewId = document.getElementById("tx-review-id");
  const txReasons = document.getElementById("tx-reasons");
  const txStatus = document.getElementById("tx-status");

  // Register
  regForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("reg-username").value.trim();
    const password = document.getElementById("reg-password").value;
    const country = document.getElementById("reg-country").value.trim() || "US";

    await apiCall(
      "/api/register",
      { username, password, country },
      regOut
    );
  });

  // Login
  loginForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("login-username").value.trim();
    const password = document.getElementById("login-password").value;
    const device_id = document.getElementById("login-device").value.trim() || "device-1";
    const ip_address = document.getElementById("login-ip").value.trim() || "10.0.0.1";
    const location = document.getElementById("login-location").value.trim() || "US";

    const data = await apiCall(
      "/api/login",
      { username, password, device_id, ip_address, location },
      loginOut
    );

    currentUser = username;
    console.log("Login risk_score:", data.risk_score);
  });

  // Transaction score
  txForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const username = document.getElementById("tx-username").value.trim();
    const amountStr = document.getElementById("tx-amount").value;
    const merchant = document.getElementById("tx-merchant").value.trim() || "Acme Store";
    const channel = document.getElementById("tx-channel").value;
    const location = document.getElementById("tx-location").value.trim() || "US";
    const device_id = document.getElementById("tx-device").value.trim() || "device-1";

    if (!currentUser || currentUser !== username) {
      prettyPrint(txOut, {
        error: "You must log in as this user before initiating a transaction.",
      });
      return;
    }

    const amount = parseFloat(amountStr);

    const data = await apiCall(
      "/api/transaction/score",
      { username, amount, merchant, channel, location, device_id },
      txOut
    );

    txResult.classList.remove("hidden");
    txRisk.textContent = data.risk_score.toFixed(3);
    txCi.textContent = `[${data.confidence_low.toFixed(3)}, ${data.confidence_high.toFixed(3)}]`;
    txDecision.textContent = data.decision;
    txReviewId.textContent = data.review_id ?? "—";

    // Map backend decision to clear user-facing status
    txStatus.className = "tx-status";
    if (data.decision === "block") {
      txStatus.classList.add("blocked");
      txStatus.textContent = "Transaction blocked – suspected fraud.";
    } else if (data.decision === "review") {
      txStatus.classList.add("hold");
      txStatus.textContent = "On hold – awaiting bank approval.";
    } else if (data.decision === "allow") {
      txStatus.classList.add("success");
      txStatus.textContent = "Transaction approved.";
    } else if (data.decision === "step_up") {
      txStatus.classList.add("hold");
      txStatus.textContent = "Additional verification required (step-up).";
    } else {
      txStatus.textContent = "";
    }

    txReasons.innerHTML = "";
    (data.reasons || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      txReasons.appendChild(li);
    });

    // If this transaction is pending review, start polling for admin decision
    if (data.review_id && data.decision === "review") {
      const reviewId = data.review_id;
      let attempts = 0;
      const maxAttempts = 40; // ~2 minutes if interval is 3s

      const poll = async () => {
        attempts += 1;
        if (attempts > maxAttempts) {
          return;
        }
        try {
          const res = await fetch(`${API_BASE}/api/transaction/review-status/${reviewId}`);
          if (!res.ok) {
            return;
          }
          const status = await res.json();
          if (status.status === "resolved" && status.final_decision) {
            // Update UI based on final decision
            txDecision.textContent = status.final_decision;
            txReasons.innerHTML = "";
            (status.reasons || []).forEach((r) => {
              const li = document.createElement("li");
              li.textContent = r;
              txReasons.appendChild(li);
            });

            txStatus.className = "tx-status";
            if (status.final_decision === "block") {
              txStatus.classList.add("blocked");
              txStatus.textContent = "Transaction blocked – bank confirmed fraud.";
            } else if (status.final_decision === "allow") {
              txStatus.classList.add("success");
              txStatus.textContent = "Transaction approved – bank released hold.";
            } else if (status.final_decision === "step_up") {
              txStatus.classList.add("hold");
              txStatus.textContent = "Bank requires additional verification (step-up).";
            }
            return; // stop polling
          }
        } catch (e) {
          // ignore transient errors and keep polling
        }
        setTimeout(poll, 3000);
      };

      setTimeout(poll, 3000);
    }
  });
});