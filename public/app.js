// ---------------------------------------------------------------------------
// AutoLens — Frontend JS
// API_BASE: empty string = same origin (local Flask dev)
//           Set RENDER_BACKEND_URL in a config script, or update manually below
//           after you deploy the backend to Render.com
// ---------------------------------------------------------------------------
const API_BASE = window.RENDER_BACKEND_URL || "";

const form = document.querySelector("#car-form");
const brandSelect = document.querySelector("#brand");
const modelSelect = document.querySelector("#model");
const errorBox = document.querySelector("#form-error");
const submitButton = document.querySelector("#submit-button");
const resultsColumn = document.querySelector(".results-column");

const formatINR = (value) => new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
}).format(value);

function setOptions(select, values, preferredValue = "") {
  select.replaceChildren();
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    select.append(option);
  }
  if (preferredValue && values.includes(preferredValue)) select.value = preferredValue;
}

function addCell(row, text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  row.append(cell);
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}

function showResult(data) {
  document.querySelector("#empty-state").hidden = true;
  document.querySelector("#results").hidden = false;
  document.querySelector("#listings-section").hidden = false;
  document.querySelector("#result-car-name").textContent = `${brandSelect.value} ${modelSelect.value}`;
  document.querySelector("#result-price").textContent = formatINR(data.price);
  document.querySelector("#result-price-range").textContent =
    `Illustrative range · ${formatINR(data.price_range.low)} – ${formatINR(data.price_range.high)}`;
  document.querySelector("#result-segment").textContent = data.segment;
  document.querySelector("#result-cluster").textContent = `Market cluster ${data.cluster_id + 1}`;

  const screeningCard = document.querySelector("#screening-card");
  screeningCard.classList.toggle("flagged", data.screening_flag === 1);
  document.querySelector("#result-screening").textContent = data.screening_flag === 1 ? "Rule flagged" : "Rule not met";

  document.querySelector("#result-average").textContent = formatINR(data.average_listing_price);
  const difference = data.price_difference;
  const differenceLabel = difference >= 0
    ? `${formatINR(difference)} above listing average`
    : `${formatINR(Math.abs(difference))} below listing average`;
  document.querySelector("#result-difference").textContent = differenceLabel;

  const body = document.querySelector("#listings-body");
  body.replaceChildren();
  for (const listing of data.similar_cars) {
    const row = document.createElement("tr");
    addCell(row, listing.car_name, "listing-name");
    addCell(row, formatINR(listing.selling_price));
    addCell(row, `${listing.vehicle_age} yrs`);
    addCell(row, `${new Intl.NumberFormat("en-IN").format(listing.km_driven)} km`);
    addCell(row, `${listing.mileage.toFixed(1)} km/l`);
    addCell(row, listing.segment_name);
    addCell(row, listing.distance_similarity.toFixed(3), "similarity-value");
    body.append(row);
  }
}

async function loadMetadata() {
  const response = await fetch(`${API_BASE}/api/metadata`);
  if (!response.ok) throw new Error("Could not load vehicle options. Refresh the page to try again.");
  const metadata = await response.json();

  setOptions(brandSelect, metadata.brands, "Hyundai");
  function updateModels() {
    const models = metadata.brand_to_models[brandSelect.value] || [];
    setOptions(modelSelect, models);
    modelSelect.disabled = models.length === 0;
  }
  brandSelect.addEventListener("change", updateModels);
  updateModels();

  setOptions(document.querySelector("#seller_type"), metadata.seller_types, "Individual");
  setOptions(document.querySelector("#fuel_type"), metadata.fuel_types, "Petrol");
  setOptions(document.querySelector("#transmission_type"), metadata.transmission_types, "Manual");

  for (const [feature, bounds] of Object.entries(metadata.feature_ranges)) {
    const input = document.querySelector(`#${feature}`);
    if (!input) continue;
    input.min = bounds.min;
    input.max = bounds.max;
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorBox.hidden = true;
  submitButton.disabled = true;
  submitButton.querySelector("span:first-child").textContent = "Analyzing…";
  resultsColumn.setAttribute("aria-busy", "true");

  const payload = {
    brand: brandSelect.value,
    model: modelSelect.value,
    seller_type: document.querySelector("#seller_type").value,
    fuel_type: document.querySelector("#fuel_type").value,
    transmission_type: document.querySelector("#transmission_type").value,
  };
  for (const key of ["vehicle_age", "km_driven", "mileage", "engine", "max_power", "seats"]) {
    payload[key] = Number(document.querySelector(`#${key}`).value);
  }

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Could not analyze this vehicle.");
    showResult(data);
  } catch (error) {
    showError(error.message || "The analysis service could not be reached.");
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector("span:first-child").textContent = "Analyze this car";
    resultsColumn.setAttribute("aria-busy", "false");
  }
});

loadMetadata().catch((error) => showError(error.message));
