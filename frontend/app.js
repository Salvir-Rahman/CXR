// Neuro-Symbolic Chest X-Ray Triage Dashboard Controller
document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("fileInput");
  const dropPrompt = document.getElementById("dropPrompt");
  const previewContainer = document.getElementById("previewContainer");
  const previewImage = document.getElementById("previewImage");
  const clearBtn = document.getElementById("clearBtn");

  const tauSlider = document.getElementById("tauSlider");
  const tauDisplay = document.getElementById("tauDisplay");
  const triageBtn = document.getElementById("triageBtn");
  const btnText = document.getElementById("btnText");
  const btnSpinner = document.getElementById("btnSpinner");
  const errorAlert = document.getElementById("errorAlert");
  const errorMessage = document.getElementById("errorMessage");

  const healthStatus = document.getElementById("healthStatus");
  const healthStatusText = document.getElementById("healthStatusText");

  const placeholderView = document.getElementById("placeholderView");
  const resultsContent = document.getElementById("resultsContent");
  const decisionBanner = document.getElementById("decisionBanner");
  const decisionAction = document.getElementById("decisionAction");
  const decisionSub = document.getElementById("decisionSub");
  const scoreValue = document.getElementById("scoreValue");
  const tauValue = document.getElementById("tauValue");
  const latencyValue = document.getElementById("latencyValue");

  const escalationCard = document.getElementById("escalationCard");
  const escalationReasonsList = document.getElementById("escalationReasonsList");
  const supportingRules = document.getElementById("supportingRules");
  const violatedRules = document.getElementById("violatedRules");
  const findingsList = document.getElementById("findingsList");

  let selectedFile = null;

  // 1. Fetch System Health on Load
  async function fetchHealth() {
    try {
      const response = await fetch("/health");
      if (!response.ok) throw new Error("Health check returned HTTP " + response.status);
      const data = await response.json();
      healthStatus.classList.remove("loading");
      healthStatus.classList.add("healthy");
      const shortSha = (data.rules_sha256 || "").substring(0, 8);
      healthStatusText.textContent = `${data.model_backbone} • ${data.rules_loaded} Rules (${shortSha}) • ${data.device.toUpperCase()}`;
      healthStatus.title = `Full Rule Registry SHA-256: ${data.rules_sha256}\n${data.disclaimer}`;
    } catch (err) {
      healthStatus.classList.remove("loading");
      healthStatusText.textContent = "Backend offline or unreachable";
    }
  }

  fetchHealth();

  // 2. Drag & Drop and File Picker Handlers
  function handleFile(file) {
    if (!file || !file.type.startsWith("image/")) {
      showError("Please select a valid image file (PNG, JPEG, or JPG).");
      return;
    }
    selectedFile = file;
    hideError();

    const reader = new FileReader();
    reader.onload = (e) => {
      previewImage.src = e.target.result;
      dropPrompt.classList.add("hidden");
      previewContainer.classList.remove("hidden");
      triageBtn.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = "";
    previewImage.src = "";
    previewContainer.classList.add("hidden");
    dropPrompt.classList.remove("hidden");
    triageBtn.disabled = true;
    hideError();
  });

  // 3. Slider synchronization
  tauSlider.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    tauDisplay.textContent = val;
  });

  // 4. Run Triage Audit
  triageBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    hideError();
    setLoading(true);

    const tau = parseFloat(tauSlider.value);
    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch(`/predict?tau=${tau}`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        let errDetail = "Triage request failed with HTTP " + response.status;
        try {
          const errJson = await response.json();
          if (errJson.detail) errDetail = errJson.detail;
        } catch (_) {}
        throw new Error(errDetail);
      }

      const result = await response.json();
      renderResults(result);
    } catch (err) {
      showError(err.message || "An unexpected error occurred during triage.");
    } finally {
      setLoading(false);
    }
  });

  // 5. Render Audit Results
  function renderResults(data) {
    placeholderView.classList.add("hidden");
    resultsContent.classList.remove("hidden");

    // Primary Decision Banner
    decisionBanner.className = "decision-banner " + data.decision;
    if (data.decision === "auto") {
      decisionAction.textContent = "AUTO-REPORT NORMAL";
      decisionSub.textContent = "Autonomous clearance: posterior abnormality below threshold and 0 rule violations.";
    } else {
      decisionAction.textContent = "ESCALATE TO RADIOLOGIST";
      decisionSub.textContent = "Safety escalation triggered: elevated risk or guideline inconsistency detected.";
    }

    scoreValue.textContent = (data.study_abnormality_score * 100).toFixed(1) + "%";
    tauValue.textContent = parseFloat(data.threshold_tau).toFixed(2);
    latencyValue.textContent = data.inference_latency_ms.toFixed(1) + " ms";

    // Escalation Reasons
    if (data.escalation_reasons && data.escalation_reasons.length > 0) {
      escalationCard.classList.remove("hidden");
      escalationReasonsList.innerHTML = data.escalation_reasons
        .map((r) => `<li>${escapeHtml(r)}</li>`)
        .join("");
    } else {
      escalationCard.classList.add("hidden");
      escalationReasonsList.innerHTML = "";
    }

    // Supporting & Violated Rules
    renderRuleBadges(supportingRules, data.supporting_rules, "supporting", "None active");
    renderRuleBadges(violatedRules, data.violated_rules, "violated", "No violations detected");

    // 14 Pathology Findings (sorted descending by probability)
    const probs = data.finding_probabilities || {};
    const states = data.discretized_findings || {};

    const sortedFindings = Object.keys(probs)
      .map((k) => ({
        name: k,
        prob: probs[k],
        state: states[k] || "absent",
      }))
      .sort((a, b) => b.prob - a.prob);

    findingsList.innerHTML = sortedFindings
      .map((f) => {
        const pct = (f.prob * 100).toFixed(1);
        let barClass = "bar-fill";
        if (f.prob >= 0.5) barClass += " high";
        else if (f.prob >= 0.2) barClass += " medium";

        return `
          <div class="finding-row">
            <span class="finding-name" title="${escapeHtml(f.name)}">${escapeHtml(f.name)}</span>
            <div class="bar-track">
              <div class="${barClass}" style="width: ${Math.max(1, f.prob * 100)}%;"></div>
            </div>
            <span class="finding-pct">${pct}%</span>
            <span class="severity-tag ${f.state}">${f.state}</span>
          </div>
        `;
      })
      .join("");
  }

  function renderRuleBadges(container, rulesList, badgeType, emptyMsg) {
    if (!rulesList || rulesList.length === 0) {
      container.innerHTML = `<span class="badge-empty">${emptyMsg}</span>`;
      return;
    }
    container.innerHTML = rulesList
      .map((r) => `<span class="rule-badge ${badgeType}">${escapeHtml(r)}</span>`)
      .join("");
  }

  function setLoading(isLoading) {
    triageBtn.disabled = isLoading;
    if (isLoading) {
      btnText.textContent = "Analyzing Radiograph...";
      btnSpinner.classList.remove("hidden");
    } else {
      btnText.textContent = "Run Triage Audit";
      btnSpinner.classList.add("hidden");
    }
  }

  function showError(msg) {
    errorMessage.textContent = msg;
    errorAlert.classList.remove("hidden");
  }

  function hideError() {
    errorAlert.classList.add("hidden");
    errorMessage.textContent = "";
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
});
