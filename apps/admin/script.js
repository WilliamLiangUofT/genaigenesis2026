const API_BASE = "http://localhost:8000";

function prettyPrint(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE}${path}`);
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed with ${res.status}`);
  }
  return data;
}

async function apiPost(path, payload) {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.detail || `Request failed with ${res.status}`);
  }
  return data;
}

function renderQueue(rows, tbody, outputEl) {
  tbody.innerHTML = "";

  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 9;
    td.className = "muted";
    td.textContent = "No items currently in the review queue.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  rows.forEach((item) => {
    const tr = document.createElement("tr");

    const addCell = (text) => {
      const td = document.createElement("td");
      td.textContent = text;
      tr.appendChild(td);
      return td;
    };

    addCell(item.id);
    addCell(item.username);
    addCell(item.amount.toFixed(2));
    addCell(item.merchant);
    addCell(item.channel);
    addCell(item.location);
    addCell(item.risk_score.toFixed(3));

    const reasonsCell = document.createElement("td");
    reasonsCell.className = "reasons";
    const ul = document.createElement("ul");
    (item.reasons || []).forEach((r) => {
      const li = document.createElement("li");
      li.textContent = r;
      ul.appendChild(li);
    });
    reasonsCell.appendChild(ul);
    tr.appendChild(reasonsCell);

    const actionsCell = document.createElement("td");
    actionsCell.className = "actions";

    const approveBtn = document.createElement("button");
    approveBtn.textContent = "Mark Legit";
    approveBtn.className = "btn-neutral";
    approveBtn.onclick = async () => {
      try {
        const res = await apiPost(`/api/admin/review/${item.id}`, {
          decision: "allow",
          comment: "Analyst: marked legitimate",
        });
        prettyPrint(outputEl, res);
        await loadQueue(tbody, outputEl);
      } catch (err) {
        prettyPrint(outputEl, { error: err.message || String(err) });
      }
    };

    const fraudBtn = document.createElement("button");
    fraudBtn.textContent = "Mark Fraud";
    fraudBtn.className = "btn-danger";
    fraudBtn.onclick = async () => {
      try {
        const res = await apiPost(`/api/admin/review/${item.id}`, {
          decision: "block",
          comment: "Analyst: confirmed fraud",
        });
        prettyPrint(outputEl, res);
        await loadQueue(tbody, outputEl);
      } catch (err) {
        prettyPrint(outputEl, { error: err.message || String(err) });
      }
    };

    const stepUpBtn = document.createElement("button");
    stepUpBtn.textContent = "Require Step-Up";
    stepUpBtn.onclick = async () => {
      try {
        const res = await apiPost(`/api/admin/review/${item.id}`, {
          decision: "step_up",
          comment: "Analyst: require additional verification",
        });
        prettyPrint(outputEl, res);
        await loadQueue(tbody, outputEl);
      } catch (err) {
        prettyPrint(outputEl, { error: err.message || String(err) });
      }
    };

    actionsCell.appendChild(approveBtn);
    actionsCell.appendChild(fraudBtn);
    actionsCell.appendChild(stepUpBtn);
    tr.appendChild(actionsCell);

    tbody.appendChild(tr);
  });
}

async function loadQueue(tbody, outputEl) {
  try {
    const queue = await apiGet("/api/admin/review-queue");
    renderQueue(queue, tbody, outputEl);
    document.getElementById("queue-count").textContent = `${queue.length} item(s) in queue`;
  } catch (err) {
    prettyPrint(outputEl, { error: err.message || String(err) });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const tbody = document.getElementById("queue-body");
  const outputEl = document.getElementById("admin-output");
  const refreshBtn = document.getElementById("refresh-btn");

  refreshBtn.addEventListener("click", () => loadQueue(tbody, outputEl));

  // Initial load
  loadQueue(tbody, outputEl);
});