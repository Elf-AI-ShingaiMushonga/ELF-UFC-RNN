const bootstrapEl = document.getElementById("bootstrap-data");
const initialState = bootstrapEl ? JSON.parse(bootstrapEl.textContent || "{}") : {};

const state = {
  dashboard: initialState,
  lastPrediction: null,
  lastRecommendation: null,
  lastSavedPrediction: null,
  filters: {
    betsMode: "all",
    betsSearch: "",
    predictionsMode: "all",
    predictionsSearch: "",
  },
};

const DRAFT_STORAGE_KEY = "ufc-prediction-desk-draft-v1";

const analysisForm = document.getElementById("analysisForm");
const analysisFlash = document.getElementById("analysisFlash");
const modelFlash = document.getElementById("modelFlash");
const maintenanceFlash = document.getElementById("maintenanceFlash");

const fighterOneInput = document.getElementById("fighterOneInput");
const fighterTwoInput = document.getElementById("fighterTwoInput");
const oddsFighterOneInput = document.getElementById("oddsFighterOneInput");
const oddsFighterTwoInput = document.getElementById("oddsFighterTwoInput");
const bankrollInput = document.getElementById("bankrollInput");
const pickSelect = document.getElementById("pickSelect");
const betOddsInput = document.getElementById("betOddsInput");
const stakeInput = document.getElementById("stakeInput");

const analyzeBtn = document.getElementById("analyzeBtn");
const savePredictionBtn = document.getElementById("savePredictionBtn");
const useRecommendationBtn = document.getElementById("useRecommendationBtn");
const addBetBtn = document.getElementById("addBetBtn");
const swapFightersBtn = document.getElementById("swapFightersBtn");

const deployModelBtn = document.getElementById("deployModelBtn");
const reloadModelBtn = document.getElementById("reloadModelBtn");
const retrainModelBtn = document.getElementById("retrainModelBtn");
const stopTrainBtn = document.getElementById("stopTrainBtn");
const stopPipelineBtn = document.getElementById("stopPipelineBtn");
const exportSnapshotBtn = document.getElementById("exportSnapshotBtn");

const maintenanceButtons = [...document.querySelectorAll(".maintenance-btn")];

const predictionHeadline = document.getElementById("predictionHeadline");
const predictionBody = document.getElementById("predictionBody");
const candidateOddsBody = document.getElementById("candidateOddsBody");
const analysisConfidenceValue = document.getElementById("analysisConfidenceValue");
const analysisFairOddsF1 = document.getElementById("analysisFairOddsF1");
const analysisFairOddsF2 = document.getElementById("analysisFairOddsF2");
const analysisMarketSignal = document.getElementById("analysisMarketSignal");
const analysisTimestamp = document.getElementById("analysisTimestamp");

const recommendationEmpty = document.getElementById("recommendationEmpty");
const recommendationCard = document.getElementById("recommendationCard");
const recommendationVerdict = document.getElementById("recommendationVerdict");
const recommendationPick = document.getElementById("recommendationPick");
const recommendationSummary = document.getElementById("recommendationSummary");
const recommendationEdge = document.getElementById("recommendationEdge");
const recommendationEv = document.getElementById("recommendationEv");
const recommendationKelly = document.getElementById("recommendationKelly");
const recommendationStake = document.getElementById("recommendationStake");

const betsTableBody = document.getElementById("betsTableBody");
const predictionsTableBody = document.getElementById("predictionsTableBody");
const dataAssetsGrid = document.getElementById("dataAssetsGrid");
const betsSearchInput = document.getElementById("betsSearchInput");
const betsFilterSelect = document.getElementById("betsFilterSelect");
const predictionsSearchInput = document.getElementById("predictionsSearchInput");
const predictionsFilterSelect = document.getElementById("predictionsFilterSelect");

const trainLogTail = document.getElementById("trainLogTail");
const pipelineLogTail = document.getElementById("pipelineLogTail");

const modelSelect = document.getElementById("modelSelect");
const modelSummary = document.getElementById("modelSummary");
const modelDetail = document.getElementById("modelDetail");
const trainStateText = document.getElementById("trainStateText");
const latestCandidateText = document.getElementById("latestCandidateText");
const pipelineStateText = document.getElementById("pipelineStateText");
const pipelineActionText = document.getElementById("pipelineActionText");

const heroModelLabel = document.getElementById("heroModelLabel");
const heroModelAuc = document.getElementById("heroModelAuc");
const heroOpenBets = document.getElementById("heroOpenBets");
const heroNetPnl = document.getElementById("heroNetPnl");
const heroRoi = document.getElementById("heroRoi");
const heroDataFreshness = document.getElementById("heroDataFreshness");

const metricOpenBets = document.getElementById("metricOpenBets");
const metricSettledBets = document.getElementById("metricSettledBets");
const metricWinRate = document.getElementById("metricWinRate");
const metricRoi = document.getElementById("metricRoi");
const metricPnl = document.getElementById("metricPnl");
const metricAvgEdge = document.getElementById("metricAvgEdge");
const metricPredictions = document.getElementById("metricPredictions");
const metricRecommended = document.getElementById("metricRecommended");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value) {
  if (value === null || value === undefined || value === "") return "-";
  return String(value);
}

function fmtFixed(value, digits = 2) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return "-";
  return num.toFixed(digits);
}

function fmtPct(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return "-";
  return `${(num * 100).toFixed(2)}%`;
}

function fmtSigned(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num)) return "-";
  const sign = num > 0 ? "+" : "";
  return `${sign}${num.toFixed(2)}`;
}

function fmtOdds(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (Number.isNaN(num) || num === 0) return "-";
  return num > 0 ? `+${num}` : String(num);
}

function fmtDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

function probabilityToAmerican(probability) {
  const prob = Number(probability);
  if (!Number.isFinite(prob) || prob <= 0 || prob >= 1) return null;
  if (prob >= 0.5) {
    return -Math.round((prob / (1 - prob)) * 100);
  }
  return Math.round(((1 - prob) / prob) * 100);
}

function normalized(value) {
  return String(value || "").trim().toLowerCase();
}

function showFlash(node, message, kind = "ok") {
  node.textContent = message;
  node.classList.remove("hidden", "flash-ok", "flash-error");
  node.classList.add(kind === "error" ? "flash-error" : "flash-ok");
}

function clearFlash(node) {
  node.textContent = "";
  node.classList.add("hidden");
  node.classList.remove("flash-ok", "flash-error");
}

function setPill(node, value) {
  const stateValue = String(value || "idle").toLowerCase();
  node.textContent = stateValue;
  node.className = "pill";
  if (["running"].includes(stateValue)) {
    node.classList.add("pill-running");
  } else if (["succeeded", "strong", "win"].includes(stateValue)) {
    node.classList.add("pill-succeeded");
  } else if (["lean", "push"].includes(stateValue)) {
    node.classList.add("pill-lean");
  } else if (["failed", "loss"].includes(stateValue)) {
    node.classList.add("pill-failed");
  } else {
    node.classList.add("pill-neutral");
  }
}

function setResultPill(result) {
  const value = String(result || "open").toLowerCase();
  if (value === "win") return "pill pill-win";
  if (value === "loss") return "pill pill-loss";
  if (value === "push") return "pill pill-push";
  return "pill pill-open";
}

function syncPickOptions() {
  const f1 = fighterOneInput.value.trim();
  const f2 = fighterTwoInput.value.trim();
  const previous = pickSelect.value;
  const options = [{ value: "", label: "Select bet side" }];
  if (f1) options.push({ value: f1, label: f1 });
  if (f2 && normalized(f2) !== normalized(f1)) options.push({ value: f2, label: f2 });

  pickSelect.innerHTML = "";
  for (const option of options) {
    const el = document.createElement("option");
    el.value = option.value;
    el.textContent = option.label;
    pickSelect.appendChild(el);
  }

  if (options.some((option) => option.value === previous)) {
    pickSelect.value = previous;
  }
  syncBetOddsFromPick();

  if (
    state.lastPrediction &&
    (normalized(state.lastPrediction.fighter_1) !== normalized(f1) ||
      normalized(state.lastPrediction.fighter_2) !== normalized(f2))
  ) {
    clearAnalysis();
  }
}

function syncBetOddsFromPick() {
  const pick = pickSelect.value.trim();
  if (!pick) return;
  if (normalized(pick) === normalized(fighterOneInput.value)) {
    betOddsInput.value = oddsFighterOneInput.value;
  } else if (normalized(pick) === normalized(fighterTwoInput.value)) {
    betOddsInput.value = oddsFighterTwoInput.value;
  }
}

function buildFormPayload() {
  const data = new FormData(analysisForm);
  const payload = Object.fromEntries(data.entries());
  payload.is_title_bout = document.getElementById("titleBoutInput").checked;
  payload.matchup = `${fighterOneInput.value.trim()} vs ${fighterTwoInput.value.trim()}`.trim();
  return payload;
}

function saveDraft() {
  try {
    const payload = buildFormPayload();
    localStorage.setItem(DRAFT_STORAGE_KEY, JSON.stringify(payload));
  } catch (error) {
    // Ignore local storage issues.
  }
}

function restoreDraft() {
  try {
    const raw = localStorage.getItem(DRAFT_STORAGE_KEY);
    if (!raw) return;
    const payload = JSON.parse(raw);
    if (!payload || typeof payload !== "object") return;
    for (const [key, value] of Object.entries(payload)) {
      const field = analysisForm.elements.namedItem(key);
      if (!(field instanceof HTMLElement)) continue;
      if (field instanceof HTMLInputElement && field.type === "checkbox") {
        field.checked = Boolean(value);
      } else if ("value" in field && !field.value) {
        field.value = String(value ?? "");
      }
    }
  } catch (error) {
    // Ignore malformed drafts.
  }
}

function clearAnalysis() {
  state.lastPrediction = null;
  state.lastRecommendation = null;
  state.lastSavedPrediction = null;

  predictionHeadline.textContent = "Run an analysis to see the projection.";
  predictionBody.textContent =
    "The desk will estimate win probability, confidence, and a recommended moneyline bet when both sides have odds.";
  candidateOddsBody.innerHTML = `<tr><td colspan="5" class="empty-cell">No market comparison yet.</td></tr>`;
  analysisConfidenceValue.textContent = "-";
  analysisFairOddsF1.textContent = "-";
  analysisFairOddsF2.textContent = "-";
  analysisMarketSignal.textContent = "-";
  analysisTimestamp.textContent = "No analysis yet.";
  recommendationCard.classList.add("hidden");
  recommendationEmpty.classList.remove("hidden");
}

function renderPrediction(prediction) {
  if (!prediction) {
    clearAnalysis();
    return;
  }
  state.lastPrediction = prediction;
  const p1 = Number(prediction.p_fighter_1 || 0);
  const p2 = Number(prediction.p_fighter_2 || 0);
  const winnerProb = Math.max(p1, p2);
  predictionHeadline.textContent = `${prediction.winner} projected winner (${(winnerProb * 100).toFixed(2)}%)`;
  predictionBody.textContent =
    `${prediction.fighter_1}: ${(p1 * 100).toFixed(2)}% | ` +
    `${prediction.fighter_2}: ${(p2 * 100).toFixed(2)}% | ` +
    `${prediction.model_label || prediction.model} | ` +
    `test AUC ${fmtFixed(prediction.model_test_auc, 4)}`;
  analysisConfidenceValue.textContent = fmtPct(prediction.confidence);
  analysisFairOddsF1.textContent = fmtOdds(probabilityToAmerican(p1));
  analysisFairOddsF2.textContent = fmtOdds(probabilityToAmerican(p2));
  analysisMarketSignal.textContent = "Awaiting priced market scan";
  analysisTimestamp.textContent = `Updated ${new Date().toLocaleTimeString()}`;
}

function renderRecommendation(recommendation) {
  state.lastRecommendation = recommendation || null;
  const candidates = recommendation?.candidates || [];
  if (!candidates.length) {
    candidateOddsBody.innerHTML = `<tr><td colspan="5" class="empty-cell">No market comparison yet.</td></tr>`;
    recommendationCard.classList.add("hidden");
    recommendationEmpty.classList.remove("hidden");
    return;
  }

  candidateOddsBody.innerHTML = candidates
    .map(
      (candidate) => `
        <tr>
          <td>${escapeHtml(candidate.pick)} ${fmtOdds(candidate.american_odds)}</td>
          <td>${fmtPct(candidate.model_probability)}</td>
          <td>${fmtPct(candidate.implied_probability)}</td>
          <td>${fmtPct(candidate.edge)}</td>
          <td>${fmtSigned(candidate.expected_value_per_unit)}</td>
        </tr>
      `
    )
    .join("");

  const best = recommendation.best;
  recommendationEmpty.classList.add("hidden");
  recommendationCard.classList.remove("hidden");
  setPill(recommendationVerdict, best?.verdict || "pass");
  recommendationPick.textContent = best ? `${best.pick} ${fmtOdds(best.american_odds)}` : "Pass";
  recommendationSummary.textContent = recommendation.summary || "No recommendation.";
  recommendationEdge.textContent = best ? fmtPct(best.edge) : "-";
  recommendationEv.textContent = best ? fmtSigned(best.expected_value_per_unit) : "-";
  recommendationKelly.textContent = best ? fmtPct(best.kelly_fraction) : "-";
  recommendationStake.textContent = best ? `${fmtFixed(best.recommended_units, 2)}u` : "-";
  analysisMarketSignal.textContent = best
    ? `${best.verdict.toUpperCase()} on ${best.pick} ${fmtOdds(best.american_odds)}`
    : "No market edge";
}

function matchesSearch(value, query) {
  if (!query) return true;
  return normalized(value).includes(normalized(query));
}

function renderFilteredBets(bets) {
  if (!bets.length) {
    betsTableBody.innerHTML = `<tr><td colspan="11" class="empty-cell">No bets match the current filters.</td></tr>`;
    return;
  }
  betsTableBody.innerHTML = bets
    .map((bet) => {
      const result = String(bet.result || "open").toLowerCase();
      const actions =
        result === "open"
          ? `
            <div class="inline-actions">
              <select data-role="settle-select" data-bet-id="${bet.id}">
                <option value="win">win</option>
                <option value="loss">loss</option>
                <option value="push">push</option>
              </select>
              <button type="button" class="ghost" data-role="settle-btn" data-bet-id="${bet.id}">Settle</button>
            </div>
          `
          : "-";
      return `
        <tr>
          <td>${fmt(bet.id)}</td>
          <td>${escapeHtml(bet.event_name)}</td>
          <td>${escapeHtml(bet.matchup)}</td>
          <td>${escapeHtml(bet.pick)}</td>
          <td>${fmtOdds(bet.american_odds)}</td>
          <td>${fmtPct(bet.model_probability)}</td>
          <td>${fmtPct(bet.edge)}</td>
          <td>${fmtFixed(bet.stake, 2)}u</td>
          <td><span class="${setResultPill(result)}">${escapeHtml(result)}</span></td>
          <td>${fmtSigned(bet.realized_pnl)}u</td>
          <td>${actions}</td>
        </tr>
      `;
    })
    .join("");
}

function restorePredictionToForm(prediction) {
  document.getElementById("eventNameInput").value = prediction.event_name || "";
  document.getElementById("eventDateInput").value = prediction.event_date || "";
  fighterOneInput.value = prediction.fighter_1 || "";
  fighterTwoInput.value = prediction.fighter_2 || "";
  document.getElementById("weightClassInput").value = prediction.weight_class || "";
  document.getElementById("genderInput").value = prediction.gender || "";
  document.getElementById("scheduledRoundsInput").value = prediction.scheduled_rounds || 3;
  document.getElementById("titleBoutInput").checked = Boolean(prediction.is_title_bout);
  document.getElementById("notesInput").value = prediction.notes || "";
  pickSelect.value = prediction.recommendation_pick || "";
  syncPickOptions();
  betOddsInput.value = "";
  stakeInput.value = "";
  oddsFighterOneInput.value = prediction.odds_fighter_1 || "";
  oddsFighterTwoInput.value = prediction.odds_fighter_2 || "";
  saveDraft();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function renderFilteredPredictions(predictions) {
  if (!predictions.length) {
    predictionsTableBody.innerHTML = `<tr><td colspan="9" class="empty-cell">No predictions match the current filters.</td></tr>`;
    return;
  }
  predictionsTableBody.innerHTML = predictions
    .map(
      (prediction) => `
        <tr>
          <td>${fmt(prediction.id)}</td>
          <td>${fmtDateTime(prediction.created_at_utc)}</td>
          <td>${escapeHtml(prediction.event_name || "-")}</td>
          <td>${escapeHtml(prediction.fighter_1)} vs ${escapeHtml(prediction.fighter_2)}</td>
          <td>${escapeHtml(prediction.winner)}</td>
          <td>${fmtPct(prediction.confidence)}</td>
          <td>${escapeHtml(prediction.recommendation_pick || "-")} ${prediction.recommendation_verdict ? `(${escapeHtml(prediction.recommendation_verdict)})` : ""}</td>
          <td>${escapeHtml(prediction.notes || "-")}</td>
          <td><button type="button" class="ghost" data-role="reuse-prediction" data-prediction-id="${prediction.id}">Reuse</button></td>
        </tr>
      `
    )
    .join("");
}

function renderModelStatus(modelStatus) {
  const deployedModel = modelStatus?.deployed_model || {};
  heroModelLabel.textContent = deployedModel.label || deployedModel.run_id || "Unknown model";
  heroModelAuc.textContent = deployedModel.test_auc ? `AUC ${fmtFixed(deployedModel.test_auc, 4)}` : "-";

  modelSummary.textContent = deployedModel.label || deployedModel.run_id || "Unknown model";
  modelDetail.textContent =
    `Test AUC ${fmtFixed(deployedModel.test_auc, 4)} | ` +
    `Val AUC ${fmtFixed(deployedModel.val_auc, 4)} | ` +
    `${deployedModel.metrics_path || "-"}`;

  const catalog = modelStatus?.catalog || [];
  const currentValue = modelSelect.value;
  modelSelect.innerHTML = "";
  for (const item of catalog) {
    const option = document.createElement("option");
    option.value = item.key;
    option.textContent = `${item.model?.label || item.key}${item.is_deployed ? " (deployed)" : ""}`;
    modelSelect.appendChild(option);
  }
  if (catalog.some((item) => item.key === currentValue)) {
    modelSelect.value = currentValue;
  }
}

function renderTrackerStatus(trackerStatus) {
  const summary = trackerStatus?.summary || {};
  heroOpenBets.textContent = fmt(summary.open_bets);
  heroNetPnl.textContent = `${fmtSigned(summary.total_pnl_settled)}u`;
  heroRoi.textContent = summary.roi === null || summary.roi === undefined ? "ROI -" : `ROI ${fmtPct(summary.roi)}`;

  metricOpenBets.textContent = fmt(summary.open_bets);
  metricSettledBets.textContent = fmt(summary.settled_bets);
  metricWinRate.textContent = fmtPct(summary.win_rate);
  metricRoi.textContent = fmtPct(summary.roi);
  metricPnl.textContent = `${fmtSigned(summary.total_pnl_settled)}u`;
  metricAvgEdge.textContent = fmtPct(summary.avg_edge);
  metricPredictions.textContent = fmt(summary.total_predictions);
  metricRecommended.textContent = fmt(summary.recommended_predictions);

  const bets = trackerStatus?.bets || [];
  const filteredBets = bets.filter((bet) => {
    const result = String(bet.result || "open").toLowerCase();
    const searchBlob = `${bet.event_name} ${bet.matchup} ${bet.pick}`.trim();
    if (state.filters.betsMode === "open" && result !== "open") return false;
    if (state.filters.betsMode === "settled" && result === "open") return false;
    if (state.filters.betsMode === "recommended" && !bet.is_recommended) return false;
    return matchesSearch(searchBlob, state.filters.betsSearch);
  });
  renderFilteredBets(filteredBets);

  const predictions = trackerStatus?.predictions || [];
  const filteredPredictions = predictions.filter((prediction) => {
    const searchBlob = `${prediction.event_name} ${prediction.fighter_1} ${prediction.fighter_2} ${prediction.notes}`.trim();
    const verdict = String(prediction.recommendation_verdict || "").toLowerCase();
    if (state.filters.predictionsMode === "recommended" && !prediction.recommendation_pick) return false;
    if (state.filters.predictionsMode === "strong" && verdict !== "strong") return false;
    return matchesSearch(searchBlob, state.filters.predictionsSearch);
  });
  renderFilteredPredictions(filteredPredictions);
}

function renderDataStatus(dataStatus) {
  heroDataFreshness.textContent = dataStatus?.last_updated_utc ? fmtDateTime(dataStatus.last_updated_utc) : "No data yet";
  const assets = dataStatus?.assets || [];
  if (!assets.length) {
    dataAssetsGrid.innerHTML = `<div class="asset-card"><p>No tracked data assets found.</p></div>`;
    return;
  }
  dataAssetsGrid.innerHTML = assets
    .map(
      (asset) => `
        <article class="asset-card">
          <p class="meta-label">${escapeHtml(asset.label)}</p>
          <p>${asset.exists ? "Available" : "Missing"}</p>
          <p class="support-copy">Updated: ${fmtDateTime(asset.modified_at_utc)}</p>
          <p class="support-copy">Size: ${asset.size_mb === null || asset.size_mb === undefined ? "-" : `${asset.size_mb.toFixed(3)} MB`}</p>
          <p class="path-line">${escapeHtml(asset.path)}</p>
        </article>
      `
    )
    .join("");
}

function renderTrainStatus(trainStatus) {
  setPill(trainStateText, trainStatus?.state || "idle");
  trainLogTail.textContent = trainStatus?.log_tail || "No training activity yet.";
  const latestCandidate = trainStatus?.latest_candidate_model || null;
  latestCandidateText.textContent = latestCandidate ? `${latestCandidate.label} | AUC ${fmtFixed(latestCandidate.test_auc, 4)}` : "None yet";
}

function renderPipelineStatus(pipelineStatus) {
  setPill(pipelineStateText, pipelineStatus?.state || "idle");
  pipelineActionText.textContent = pipelineStatus?.action_label || "-";
  pipelineLogTail.textContent = pipelineStatus?.log_tail || "No pipeline activity yet.";
}

function syncButtons() {
  const trainRunning = Boolean(state.dashboard?.train_status?.running);
  const pipelineRunning = Boolean(state.dashboard?.pipeline_status?.running);

  deployModelBtn.disabled = trainRunning || pipelineRunning;
  reloadModelBtn.disabled = false;
  retrainModelBtn.disabled = trainRunning || pipelineRunning;
  stopTrainBtn.disabled = !trainRunning;

  maintenanceButtons.forEach((button) => {
    button.disabled = trainRunning || pipelineRunning;
  });
  stopPipelineBtn.disabled = !pipelineRunning;
}

function renderDashboard(dashboardState) {
  state.dashboard = dashboardState || {};
  renderModelStatus(dashboardState.model_status || {});
  renderTrackerStatus(dashboardState.tracker_status || {});
  renderDataStatus(dashboardState.data_status || {});
  renderTrainStatus(dashboardState.train_status || {});
  renderPipelineStatus(dashboardState.pipeline_status || {});
  syncButtons();
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  const body = await response.json();
  if (!body.ok) {
    throw new Error(body.error || "Request failed.");
  }
  return body;
}

async function analyze(savePrediction = false) {
  clearFlash(analysisFlash);
  try {
    analyzeBtn.disabled = true;
    savePredictionBtn.disabled = true;
    const payload = buildFormPayload();
    payload.save_prediction = savePrediction;
    const body = await fetchJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    renderPrediction(body.prediction);
    renderRecommendation(body.recommendation);
    if (body.saved_prediction) {
      state.lastSavedPrediction = body.saved_prediction;
      showFlash(analysisFlash, `Analysis saved as entry #${body.saved_prediction.id}.`, "ok");
    } else {
      showFlash(analysisFlash, "Analysis complete.", "ok");
    }
    if (body.tracker_status) {
      renderTrackerStatus(body.tracker_status);
    }
  } catch (error) {
    showFlash(analysisFlash, String(error), "error");
  } finally {
    analyzeBtn.disabled = false;
    savePredictionBtn.disabled = false;
  }
}

function applyRecommendationToBetSlip() {
  const best = state.lastRecommendation?.best;
  if (!best) {
    showFlash(analysisFlash, "Analyze a priced fight before using the recommendation.", "error");
    return;
  }
  pickSelect.value = best.pick;
  betOddsInput.value = best.american_odds;
  stakeInput.value = Number(best.recommended_units || 0).toFixed(2);
  showFlash(analysisFlash, "Recommendation copied into the bet slip.", "ok");
}

async function addBet() {
  clearFlash(analysisFlash);
  try {
    const payload = buildFormPayload();
    if (!payload.event_name) {
      throw new Error("Event name is required before logging a bet.");
    }
    if (!payload.pick) {
      throw new Error("Pick is required before logging a bet.");
    }
    if (!payload.american_odds) {
      throw new Error("Bet odds are required before logging a bet.");
    }
    if (!payload.stake) {
      throw new Error("Stake is required before logging a bet.");
    }

    if (state.lastPrediction) {
      if (normalized(payload.pick) === normalized(state.lastPrediction.fighter_1)) {
        payload.model_probability = state.lastPrediction.p_fighter_1;
      } else if (normalized(payload.pick) === normalized(state.lastPrediction.fighter_2)) {
        payload.model_probability = state.lastPrediction.p_fighter_2;
      }
    }
    if (state.lastSavedPrediction?.id) {
      payload.prediction_id = state.lastSavedPrediction.id;
    }
    const best = state.lastRecommendation?.best;
    payload.is_recommended = Boolean(best && normalized(best.pick) === normalized(payload.pick) && state.lastRecommendation?.recommended);

    addBetBtn.disabled = true;
    const body = await fetchJson("/api/bets/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    renderTrackerStatus(body.status);
    showFlash(analysisFlash, `Bet #${body.bet.id} logged.`, "ok");
  } catch (error) {
    showFlash(analysisFlash, String(error), "error");
  } finally {
    addBetBtn.disabled = false;
  }
}

async function settleBet(betId, result) {
  clearFlash(analysisFlash);
  try {
    const body = await fetchJson("/api/bets/settle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ bet_id: betId, result }),
    });
    renderTrackerStatus(body.status);
    showFlash(analysisFlash, `Bet #${betId} settled as ${result}.`, "ok");
  } catch (error) {
    showFlash(analysisFlash, String(error), "error");
  }
}

async function refreshDashboard() {
  try {
    const body = await fetchJson("/api/dashboard/state?tail=800");
    renderDashboard(body.state);
  } catch (error) {
    showFlash(maintenanceFlash, `Refresh failed: ${error}`, "error");
  }
}

async function runMaintenanceAction(action) {
  clearFlash(maintenanceFlash);
  if (action === "reset_data") {
    const ok = window.confirm(
      "Reset + Rescrape deletes the current raw fight file, sequences, and checkpoint before scraping again. Continue?"
    );
    if (!ok) return;
  }
  try {
    const body = await fetchJson("/api/pipeline/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }),
    });
    state.dashboard.pipeline_status = body.status;
    renderTrainStatus(state.dashboard.train_status || {});
    renderPipelineStatus(state.dashboard.pipeline_status || {});
    syncButtons();
    showFlash(maintenanceFlash, `Started ${action.replaceAll("_", " ")}.`, "ok");
  } catch (error) {
    showFlash(maintenanceFlash, String(error), "error");
  }
}

async function stopPipeline() {
  clearFlash(maintenanceFlash);
  try {
    const body = await fetchJson("/api/pipeline/stop", { method: "POST" });
    state.dashboard.pipeline_status = body.status;
    renderPipelineStatus(state.dashboard.pipeline_status || {});
    syncButtons();
    showFlash(maintenanceFlash, "Pipeline stop signal sent.", "ok");
  } catch (error) {
    showFlash(maintenanceFlash, String(error), "error");
  }
}

async function retrainModel() {
  clearFlash(modelFlash);
  try {
    const body = await fetchJson("/api/model/retrain", { method: "POST" });
    state.dashboard.train_status = body.status;
    renderTrainStatus(state.dashboard.train_status || {});
    syncButtons();
    showFlash(modelFlash, "Candidate retraining started.", "ok");
  } catch (error) {
    showFlash(modelFlash, String(error), "error");
  }
}

async function stopTraining() {
  clearFlash(modelFlash);
  try {
    const body = await fetchJson("/api/train/stop", { method: "POST" });
    state.dashboard.train_status = body.status;
    renderTrainStatus(state.dashboard.train_status || {});
    syncButtons();
    showFlash(modelFlash, "Training stop signal sent.", "ok");
  } catch (error) {
    showFlash(modelFlash, String(error), "error");
  }
}

async function deploySelectedModel() {
  clearFlash(modelFlash);
  try {
    const body = await fetchJson("/api/model/deploy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_key: modelSelect.value }),
    });
    state.dashboard.model_status = body.status;
    renderModelStatus(state.dashboard.model_status || {});
    showFlash(modelFlash, "Deployed model updated.", "ok");
  } catch (error) {
    showFlash(modelFlash, String(error), "error");
  }
}

async function reloadModel() {
  clearFlash(modelFlash);
  try {
    const body = await fetchJson("/api/model/reload", { method: "POST" });
    state.dashboard.model_status = body.status;
    renderModelStatus(state.dashboard.model_status || {});
    showFlash(modelFlash, "Predictor artifacts reloaded.", "ok");
  } catch (error) {
    showFlash(modelFlash, String(error), "error");
  }
}

swapFightersBtn.addEventListener("click", () => {
  const left = fighterOneInput.value;
  fighterOneInput.value = fighterTwoInput.value;
  fighterTwoInput.value = left;
  syncPickOptions();
});

fighterOneInput.addEventListener("input", syncPickOptions);
fighterTwoInput.addEventListener("input", syncPickOptions);
pickSelect.addEventListener("change", syncBetOddsFromPick);
oddsFighterOneInput.addEventListener("input", syncBetOddsFromPick);
oddsFighterTwoInput.addEventListener("input", syncBetOddsFromPick);
betsSearchInput.addEventListener("input", () => {
  state.filters.betsSearch = betsSearchInput.value;
  renderTrackerStatus(state.dashboard?.tracker_status || {});
});
betsFilterSelect.addEventListener("change", () => {
  state.filters.betsMode = betsFilterSelect.value;
  renderTrackerStatus(state.dashboard?.tracker_status || {});
});
predictionsSearchInput.addEventListener("input", () => {
  state.filters.predictionsSearch = predictionsSearchInput.value;
  renderTrackerStatus(state.dashboard?.tracker_status || {});
});
predictionsFilterSelect.addEventListener("change", () => {
  state.filters.predictionsMode = predictionsFilterSelect.value;
  renderTrackerStatus(state.dashboard?.tracker_status || {});
});

analyzeBtn.addEventListener("click", () => analyze(false));
savePredictionBtn.addEventListener("click", () => analyze(true));
useRecommendationBtn.addEventListener("click", applyRecommendationToBetSlip);
addBetBtn.addEventListener("click", addBet);

deployModelBtn.addEventListener("click", deploySelectedModel);
reloadModelBtn.addEventListener("click", reloadModel);
retrainModelBtn.addEventListener("click", retrainModel);
stopTrainBtn.addEventListener("click", stopTraining);
stopPipelineBtn.addEventListener("click", stopPipeline);
exportSnapshotBtn.addEventListener("click", () => {
  window.location.href = "/api/research/snapshot?download=1";
});

maintenanceButtons.forEach((button) => {
  button.addEventListener("click", () => runMaintenanceAction(button.dataset.action));
});

betsTableBody.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.dataset.role !== "settle-btn") return;
  const betId = Number(target.dataset.betId);
  const select = betsTableBody.querySelector(`select[data-role="settle-select"][data-bet-id="${betId}"]`);
  if (!(select instanceof HTMLSelectElement)) return;
  settleBet(betId, select.value);
});

predictionsTableBody.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.dataset.role !== "reuse-prediction") return;
  const predictionId = Number(target.dataset.predictionId);
  const predictions = state.dashboard?.tracker_status?.predictions || [];
  const prediction = predictions.find((row) => Number(row.id) === predictionId);
  if (!prediction) return;
  restorePredictionToForm(prediction);
  showFlash(analysisFlash, `Loaded saved analysis #${predictionId} back into the form.`, "ok");
});

analysisForm.addEventListener("input", saveDraft);
analysisForm.addEventListener("change", saveDraft);
analysisForm.addEventListener("keydown", (event) => {
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
    event.preventDefault();
    analyze(false);
  }
});

function applyDefaults() {
  const defaults = state.dashboard?.defaults || {};
  const eventDateInput = document.getElementById("eventDateInput");
  const scheduledRoundsInput = document.getElementById("scheduledRoundsInput");
  if (eventDateInput && !eventDateInput.value && defaults.event_date) {
    eventDateInput.value = defaults.event_date;
  }
  if (scheduledRoundsInput && !scheduledRoundsInput.value && defaults.scheduled_rounds) {
    scheduledRoundsInput.value = defaults.scheduled_rounds;
  }
  if (bankrollInput && !bankrollInput.value && defaults.bankroll_units) {
    bankrollInput.value = defaults.bankroll_units;
  }
}

renderDashboard(initialState);
applyDefaults();
restoreDraft();
syncPickOptions();
clearAnalysis();
setInterval(refreshDashboard, 4000);
