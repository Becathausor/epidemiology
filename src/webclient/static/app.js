const COLORS = { S: "#1f77b4", I: "#d62728", R: "#2ca02c", D: "#7f7f7f" };

function pingHealth() {
  const indicator = document.getElementById("health-indicator");
  fetch("/api/health")
    .then((res) => {
      if (!res.ok) throw new Error("unhealthy");
      indicator.textContent = "en ligne";
      indicator.className = "status-ok";
    })
    .catch(() => {
      indicator.textContent = "injoignable";
      indicator.className = "status-error";
    });
}

function wireModelToggle() {
  const select = document.getElementById("model");
  const vitalFields = document.getElementById("vital-fields");
  select.addEventListener("change", () => {
    vitalFields.classList.toggle("hidden", select.value !== "SIRDVital");
  });
}

function wireNoiseToggle() {
  const checkbox = document.getElementById("noise-enabled");
  const level = document.getElementById("noise-level");
  const seed = document.getElementById("noise-seed");
  checkbox.addEventListener("change", () => {
    level.disabled = !checkbox.checked;
    seed.disabled = !checkbox.checked;
  });
}

function readNumber(id) {
  return Number(document.getElementById(id).value);
}

function buildPayload() {
  const seedRaw = document.getElementById("noise-seed").value;
  return {
    model: document.getElementById("model").value,
    population: readNumber("population"),
    s0: readNumber("s0"),
    i0: readNumber("i0"),
    r0: readNumber("r0"),
    d0: readNumber("d0"),
    beta: readNumber("beta"),
    gamma: readNumber("gamma"),
    mu: readNumber("mu"),
    birth_rate: readNumber("birth_rate"),
    natural_death_rate: readNumber("natural_death_rate"),
    t_max: readNumber("t_max"),
    n_points: Math.round(readNumber("n_points")),
    noise: {
      enabled: document.getElementById("noise-enabled").checked,
      level: readNumber("noise-level"),
      seed: seedRaw === "" ? null : Math.round(Number(seedRaw)),
    },
  };
}

function showError(message) {
  const banner = document.getElementById("error-banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

function hideError() {
  document.getElementById("error-banner").classList.add("hidden");
}

function errorMessageFromDetail(detail) {
  if (Array.isArray(detail)) {
    return detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return String(detail);
}

function wireFormSubmit() {
  const form = document.getElementById("sim-form");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const payload = buildPayload();

    fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(async (res) => {
        const body = await res.json();
        if (!res.ok) {
          showError(errorMessageFromDetail(body.detail));
          return;
        }
        hideError();
        renderChart(body);
        renderLegend(body);
      })
      .catch(() => showError("Impossible de contacter l'API."));
  });
}

function renderChart(data) {
  const canvas = document.getElementById("chart");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const margin = { left: 60, right: 20, top: 20, bottom: 40 };
  const plotW = canvas.width - margin.left - margin.right;
  const plotH = canvas.height - margin.top - margin.bottom;

  const t = data.t;
  const xMin = t[0];
  const xMax = t[t.length - 1];

  const allValues = Object.values(data.compartments).flat();
  if (data.noisy_compartments) {
    allValues.push(...Object.values(data.noisy_compartments).flat());
  }
  const yMin = 0;
  const yMax = Math.max(...allValues, 1);

  const xScale = (v) => margin.left + ((v - xMin) / (xMax - xMin)) * plotW;
  const yScale = (v) => margin.top + plotH - ((v - yMin) / (yMax - yMin)) * plotH;

  // Axes
  ctx.strokeStyle = "#888";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  ctx.fillStyle = "#888";
  ctx.font = "11px sans-serif";
  ctx.textAlign = "center";
  const nTicks = 5;
  for (let i = 0; i <= nTicks; i++) {
    const tv = xMin + ((xMax - xMin) * i) / nTicks;
    const x = xScale(tv);
    ctx.beginPath();
    ctx.moveTo(x, margin.top + plotH);
    ctx.lineTo(x, margin.top + plotH + 4);
    ctx.stroke();
    ctx.fillText(tv.toFixed(0), x, margin.top + plotH + 16);
  }
  ctx.textAlign = "right";
  for (let i = 0; i <= nTicks; i++) {
    const yv = yMin + ((yMax - yMin) * i) / nTicks;
    const y = yScale(yv);
    ctx.beginPath();
    ctx.moveTo(margin.left - 4, y);
    ctx.lineTo(margin.left, y);
    ctx.stroke();
    ctx.fillText(yv.toFixed(0), margin.left - 8, y + 3);
  }

  // Trajectoires simulées (lignes)
  for (const [name, values] of Object.entries(data.compartments)) {
    ctx.strokeStyle = COLORS[name] || "#000";
    ctx.lineWidth = 2;
    ctx.beginPath();
    values.forEach((v, i) => {
      const x = xScale(t[i]);
      const y = yScale(v);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  // Observations bruitées (points)
  if (data.noisy_compartments) {
    for (const [name, values] of Object.entries(data.noisy_compartments)) {
      ctx.fillStyle = COLORS[name] || "#000";
      values.forEach((v, i) => {
        const x = xScale(t[i]);
        const y = yScale(v);
        ctx.beginPath();
        ctx.arc(x, y, 2.5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }
}

function renderLegend(data) {
  const legend = document.getElementById("legend");
  legend.innerHTML = "";

  if (data.noisy_compartments) {
    const caption = document.createElement("span");
    caption.className = "caption";
    caption.textContent = "trait plein = trajectoire simulée, points = observations bruitées";
    legend.appendChild(caption);
  }

  for (const name of Object.keys(data.compartments)) {
    const item = document.createElement("span");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = COLORS[name] || "#000";
    item.appendChild(swatch);
    item.appendChild(document.createTextNode(name));
    legend.appendChild(item);
  }
}

window.addEventListener("DOMContentLoaded", () => {
  pingHealth();
  wireModelToggle();
  wireNoiseToggle();
  wireFormSubmit();
});
