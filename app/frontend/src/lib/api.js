/* Client for the ELASTIQ pricing API (service-pricing-optimization).

   Base URL is configurable via VITE_PRICING_API_BASE_URL (see
   .env.example), read at build time by Vite. It defaults to the
   service's own local default port ONLY in dev mode (`npm run dev`) --
   a production build (`npm run build`, including the GitHub Pages
   deploy) never bakes a localhost URL into the bundle. With no base URL
   configured, calls fail fast with a clear "not configured" message
   instead of firing a request at the wrong origin.

   The app never hard-fails if the service is unreachable or
   unconfigured -- see App.jsx and DataView.jsx, which both fall back to
   client-side computation / the bundled demo snapshot with a clear
   error surfaced to the user, so the app keeps working with zero setup
   (including when hosted with no backend at all, e.g. GitHub Pages). */

const rawApiBase = import.meta.env.VITE_PRICING_API_BASE_URL || (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");
const API_BASE = rawApiBase.replace(/\/$/, "");

async function getJson(path) {
  if (!API_BASE) {
    throw new Error(
      "No pricing API configured for this deployment. Set VITE_PRICING_API_BASE_URL at build time to enable live data."
    );
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`);
  } catch {
    throw new Error(
      `Could not reach the pricing API at ${API_BASE}. Start it with: cd service-pricing-optimization && uvicorn app.main:app --reload`
    );
  }

  if (!response.ok) {
    let detail = "";
    try { detail = (await response.json()).detail || ""; } catch { /* ignore */ }
    throw new Error(`Pricing API returned ${response.status}${detail ? `: ${detail}` : ""}.`);
  }

  return response.json();
}

/* Fetches the live product catalog (input data: price, cost, elasticity...).
   Same row shape as DEMO_DATA -- the service aggregates its own freshly
   computed recommendations the same way scripts/export_demo_data.py
   does, so callers can swap it in directly. */
export async function fetchLiveProducts() {
  const products = await getJson("/products");

  if (!Array.isArray(products) || !products.length) {
    throw new Error("Pricing API returned no priceable products.");
  }

  return products;
}

/* Fetches one live pricing decision per product (aggregated across its
   stores). See app/frontend/src/lib/liveRecommendations.js for how these
   get applied to the client-computed rows for the "grid" technique. */
export async function fetchLiveProductRecommendations() {
  const recommendations = await getJson("/products/recommendations");

  if (!Array.isArray(recommendations)) {
    throw new Error("Pricing API returned an unexpected response shape.");
  }

  return recommendations;
}
