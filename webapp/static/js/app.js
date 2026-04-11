// ViHateT5 Demo — Client-side JavaScript
(() => {
  "use strict";

  // ==================== DOM Elements ====================
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const modelSelect  = $("#model-select");
  const modelIdDisp  = $("#model-id-display");
  const inputText    = $("#input-text");
  const runBtn       = $("#run-btn");
  const runBtnText   = $("#run-btn-text");
  const runBtnSpin   = $("#run-btn-spinner");
  const resultPlace  = $("#result-placeholder");
  const resultContent= $("#result-content");
  const resultPrimary= $("#result-primary");
  const resultTime   = $("#result-time");
  const allTasksList = $("#all-tasks-list");

  // Batch
  const batchForm    = $("#batch-form");
  const batchFile    = $("#batch-file");
  const dropZone     = $("#drop-zone");
  const fileName     = $("#file-name");
  const batchRunBtn  = $("#batch-run-btn");
  const batchBtnText = $("#batch-btn-text");
  const batchBtnSpin = $("#batch-btn-spinner");
  const batchTextCol = $("#batch-text-col");

  // Theme
  const themeToggle  = $("#theme-toggle");

  // Sidebar
  const sidebar      = $("#sidebar");
  const sidebarToggle= $("#sidebar-toggle");
  const sidebarOverlay= $("#sidebar-overlay");

  // Model ID map (from template)
  const MODEL_IDS = {};
  modelSelect.querySelectorAll("option").forEach(opt => {
    MODEL_IDS[opt.value] = opt.value;
  });

  const LABEL_COLORS = {
    CLEAN: "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300 border-green-200",
    NONE:  "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300 border-green-200",
    OFFENSIVE: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300 border-yellow-200",
    TOXIC: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300 border-red-200",
    HATE:  "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300 border-red-200",
  };

  // ==================== Theme ====================
  function initTheme() {
    const saved = localStorage.getItem("theme");
    if (saved === "dark" || (!saved && window.matchMedia("(prefers-color-scheme: dark)").matches)) {
      document.documentElement.setAttribute("data-theme", "dark");
    }
  }
  initTheme();

  themeToggle.addEventListener("click", () => {
    const html = document.documentElement;
    const isDark = html.getAttribute("data-theme") === "dark";
    html.setAttribute("data-theme", isDark ? "light" : "dark");
    localStorage.setItem("theme", isDark ? "light" : "dark");
  });

  // ==================== Sidebar ====================
  sidebarToggle.addEventListener("click", () => {
    sidebar.classList.toggle("open");
    sidebarOverlay.classList.toggle("hidden");
  });
  sidebarOverlay.addEventListener("click", () => {
    sidebar.classList.remove("open");
    sidebarOverlay.classList.add("hidden");
  });

  // ==================== Model ID display ====================
  function updateModelId() {
    const label = modelSelect.value;
    // Fetch from options' actual HF ids via backend mapping
    modelIdDisp.textContent = label;
  }
  modelSelect.addEventListener("change", updateModelId);
  updateModelId();

  // ==================== Tabs ====================
  $$(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      $$(".tab-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      $$(".tab-content").forEach(s => s.classList.add("hidden"));
      $(`#tab-${btn.dataset.tab}`).classList.remove("hidden");
    });
  });

  // ==================== Sample buttons ====================
  $$(".sample-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      inputText.value = btn.dataset.text;
      inputText.focus();
    });
  });

  // ==================== Helper: get selected task ====================
  function getSelectedTask() {
    const checked = document.querySelector('input[name="task"]:checked');
    return checked ? checked.value : "vihsd";
  }

  // ==================== Highlight hate spans ====================
  function highlightSpans(text, spans) {
    if (!spans || spans.length === 0) return escapeHtml(text);
    // Merge & sort spans
    const sorted = [...spans].sort((a, b) => a.start - b.start);
    const merged = [];
    for (const sp of sorted) {
      if (merged.length && sp.start <= merged[merged.length - 1].end) {
        merged[merged.length - 1].end = Math.max(merged[merged.length - 1].end, sp.end);
      } else {
        merged.push({ ...sp });
      }
    }
    let html = "";
    let prev = 0;
    for (const sp of merged) {
      html += escapeHtml(text.slice(prev, sp.start));
      html += `<span class="hate-span">${escapeHtml(text.slice(sp.start, sp.end))}</span>`;
      prev = sp.end;
    }
    html += escapeHtml(text.slice(prev));
    return html;
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  // ==================== Inference ====================
  function setLoading(on) {
    runBtn.disabled = on;
    runBtnText.classList.toggle("hidden", on);
    runBtnSpin.classList.toggle("hidden", !on);
  }

  async function runInference() {
    const text = inputText.value.trim();
    if (!text) { inputText.focus(); return; }

    setLoading(true);
    resultPlace.classList.add("hidden");
    resultContent.classList.add("hidden");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          task: getSelectedTask(),
          model: modelSelect.value,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        showError(data.error || "Inference failed.");
        return;
      }
      displayResult(data, text);
    } catch (err) {
      showError("Network error: " + err.message);
    } finally {
      setLoading(false);
    }
  }

  runBtn.addEventListener("click", runInference);
  inputText.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runInference();
  });

  function showError(msg) {
    resultContent.classList.remove("hidden");
    resultPlace.classList.add("hidden");
    resultPrimary.className = "text-center p-6 rounded-xl bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 animate-fadeInUp";
    resultPrimary.innerHTML = `<p class="text-red-600 dark:text-red-400 font-medium">${escapeHtml(msg)}</p>`;
    resultTime.textContent = "";
    allTasksList.innerHTML = "";
    $("#result-all-tasks").classList.add("hidden");
  }

  function displayResult(data, text) {
    resultContent.classList.remove("hidden");
    resultPlace.classList.add("hidden");

    // Primary result
    if (data.task === "vihos") {
      const spans = data.spans || [];
      const spanHtml = highlightSpans(text, spans);
      resultPrimary.className = "p-6 rounded-xl bg-gray-50 dark:bg-gray-700/50 border border-gray-200 dark:border-gray-600 animate-fadeInUp";
      resultPrimary.innerHTML = `
        <p class="text-sm font-semibold mb-2">Hate Spans Detection</p>
        <p class="text-lg leading-relaxed">${spanHtml}</p>
        <p class="text-xs text-gray-400 mt-3">${spans.length} hate span(s) detected</p>
      `;
    } else {
      const label = data.label || "UNKNOWN";
      const colorCls = LABEL_COLORS[label] || "bg-gray-100 text-gray-800";
      resultPrimary.className = `text-center p-8 rounded-xl border-2 ${colorCls} animate-fadeInUp`;
      resultPrimary.innerHTML = `
        <p class="text-3xl font-black tracking-wider">${escapeHtml(label)}</p>
        <p class="text-xs mt-2 opacity-70">${escapeHtml(data.task_label)}</p>
      `;
    }

    // Time
    resultTime.textContent = `⏱️ ${data.inference_time}s · ${data.device.toUpperCase()}`;

    // All-task results
    const allTasks = data.all_tasks;
    if (allTasks) {
      $("#result-all-tasks").classList.remove("hidden");
      allTasksList.innerHTML = "";
      const taskNames = { vihsd: "ViHSD", victsd: "ViCTSD", vihos: "ViHOS" };
      for (const [tk, val] of Object.entries(allTasks)) {
        const li = document.createElement("div");
        li.className = "flex items-center gap-2 text-sm py-1";
        const labelStr = typeof val === "string" ? val : JSON.stringify(val);
        const colorCls = LABEL_COLORS[labelStr] || "";
        const badge = colorCls
          ? `<span class="px-2 py-0.5 rounded-full text-xs font-bold ${colorCls}">${escapeHtml(labelStr)}</span>`
          : `<span class="text-gray-500">${escapeHtml(labelStr)}</span>`;
        li.innerHTML = `<span class="font-medium w-16">${taskNames[tk]}:</span> ${badge}`;
        allTasksList.appendChild(li);
      }
    }
  }

  // ==================== Batch ====================
  // Drop zone
  dropZone.addEventListener("click", () => batchFile.click());
  dropZone.addEventListener("dragover", (e) => { e.preventDefault(); dropZone.classList.add("border-uit"); });
  dropZone.addEventListener("dragleave", () => dropZone.classList.remove("border-uit"));
  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("border-uit");
    if (e.dataTransfer.files.length) {
      batchFile.files = e.dataTransfer.files;
      onFileSelected();
    }
  });
  batchFile.addEventListener("change", onFileSelected);

  function onFileSelected() {
    if (batchFile.files.length) {
      fileName.textContent = batchFile.files[0].name;
      fileName.classList.remove("hidden");
      batchRunBtn.disabled = false;
    }
  }

  batchForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!batchFile.files.length) return;

    batchRunBtn.disabled = true;
    batchBtnText.classList.add("hidden");
    batchBtnSpin.classList.remove("hidden");

    const formData = new FormData();
    formData.append("file", batchFile.files[0]);
    formData.append("task", getSelectedTask());
    formData.append("model_label", modelSelect.value);
    formData.append("text_column", batchTextCol.value);

    try {
      const res = await fetch("/api/batch", { method: "POST", body: formData });
      if (!res.ok) {
        const err = await res.json();
        alert(err.error || "Batch inference failed.");
        return;
      }
      // Download the CSV response
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "predictions.csv";
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert("Network error: " + err.message);
    } finally {
      batchRunBtn.disabled = false;
      batchBtnText.classList.remove("hidden");
      batchBtnSpin.classList.add("hidden");
    }
  });

})();
