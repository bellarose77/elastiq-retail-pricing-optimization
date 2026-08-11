const CATEGORIES = ["Beverages", "Grocery", "Household", "Personal Care", "Seasonal", "Prepared Food"];
const REGIONS = ["North", "Central", "East", "West"];

export const PROBLEM_SIZES = [
  { id: "small", label: "Small", size: 50, note: "Fast functional test" },
  { id: "medium", label: "Medium", size: 250, note: "Method comparison test" },
  { id: "large", label: "Large", size: 1000, note: "Performance and scale test" },
  { id: "custom", label: "Custom", size: null, note: "5 to 2,000 units" },
];

export const SCENARIO_PROFILES = [
  { id: "balanced", label: "Balanced retail", note: "Mixed margins, elasticity, inventory and promotion evidence." },
  { id: "inventory", label: "Inventory constrained", note: "More items approach or reach their available stock." },
  { id: "promotion", label: "Promotion intensive", note: "More promotion candidates with varied incremental uplift." },
  { id: "volatile", label: "Volatile market", note: "Wider demand, competitor and market-signal variation." },
];

export function seededRandom(seed = 42) {
  let state = (Number(seed) || 42) >>> 0;
  return () => {
    state += 0x6D2B79F5;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function randomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    const value = new Uint32Array(1);
    globalThis.crypto.getRandomValues(value);
    return value[0] || 42;
  }
  return Math.floor(Math.random() * 4294967295) || 42;
}

const between = (rng, lo, hi) => lo + (hi - lo) * rng();
const choose = (rng, values) => values[Math.floor(rng() * values.length) % values.length];
const round = (value, digits = 2) => Number(value.toFixed(digits));

export function generateScenario({ size = 50, seed = 42, profile = "balanced" } = {}) {
  const count = Math.max(5, Math.min(2000, Math.floor(Number(size) || 50)));
  const rng = seededRandom(seed);
  const profileCfg = {
    balanced: { demandSpread: 1, inventoryLo: .8, inventoryHi: 1.8, promoRate: .24, ragSpread: .06, compSpread: .1 },
    inventory: { demandSpread: 1.05, inventoryLo: .42, inventoryHi: 1.05, promoRate: .2, ragSpread: .05, compSpread: .08 },
    promotion: { demandSpread: 1.1, inventoryLo: .75, inventoryHi: 1.6, promoRate: .62, ragSpread: .07, compSpread: .11 },
    volatile: { demandSpread: 1.7, inventoryLo: .55, inventoryHi: 2.1, promoRate: .34, ragSpread: .1, compSpread: .15 },
  }[profile] || null;
  const cfg = profileCfg || { demandSpread: 1, inventoryLo: .8, inventoryHi: 1.8, promoRate: .24, ragSpread: .06, compSpread: .1 };

  const categoryBase = {
    Beverages: { price: 7, demand: 95, elasticity: -1.55, margin: .45 },
    Grocery: { price: 6, demand: 120, elasticity: -1.35, margin: .38 },
    Household: { price: 17, demand: 44, elasticity: -1.1, margin: .5 },
    "Personal Care": { price: 14, demand: 52, elasticity: -1.7, margin: .58 },
    Seasonal: { price: 24, demand: 35, elasticity: -2.0, margin: .52 },
    "Prepared Food": { price: 9, demand: 78, elasticity: -1.25, margin: .42 },
  };

  return Array.from({ length: count }, (_, index) => {
    const category = CATEGORIES[index % CATEGORIES.length];
    const base = categoryBase[category];
    const priceNoise = Math.exp(between(rng, -.45, .45) * cfg.demandSpread);
    const currentPrice = Math.max(1.5, base.price * priceNoise);
    const margin = Math.max(.22, Math.min(.72, base.margin + between(rng, -.12, .12)));
    const unitCost = currentPrice * (1 - margin);
    const demandNoise = Math.exp(between(rng, -.65, .65) * cfg.demandSpread);
    const baselineQuantity = Math.max(4, base.demand * demandNoise * Math.pow(currentPrice / base.price, -.28));
    const elasticity = Math.min(-.55, base.elasticity + between(rng, -.42, .42) * cfg.demandSpread);
    const inventory = Math.max(3, baselineQuantity * between(rng, cfg.inventoryLo, cfg.inventoryHi));
    const promotionCandidate = rng() < cfg.promoRate;
    const promotionReliable = rng() > (profile === "promotion" ? .12 : .2);
    const uplift = promotionCandidate ? between(rng, .08, .56) : between(rng, 0, .16);
    const source = rng() < .82 ? "product_iv" : "pooled_iv";
    const storeNumber = (index % Math.max(5, Math.ceil(count / 18))) + 1;
    const productNumber = (index % Math.max(12, Math.ceil(count / 5))) + 1;
    return {
      itemId: `LIVE-${String(index + 1).padStart(4, "0")}`,
      productId: `SKU-${String(productNumber).padStart(4, "0")}`,
      storeId: `STORE-${String(storeNumber).padStart(3, "0")}`,
      category,
      region: choose(rng, REGIONS),
      currentPrice: round(currentPrice),
      unitCost: round(unitCost),
      competitorPrice: round(currentPrice * between(rng, 1 - cfg.compSpread, 1 + cfg.compSpread)),
      baselineQuantity: round(baselineQuantity, 3),
      baselineSource: "generated_live_forecast",
      inventory: round(inventory, 3),
      elasticity: round(elasticity, 4),
      elasticitySource: source,
      elasticityIsCausal: true,
      confidence: round(between(rng, .58, .96), 3),
      promotionFlag: promotionCandidate && rng() < .35 ? 1 : 0,
      promotionUpliftRate: round(uplift, 4),
      promotionModelReliable: promotionReliable,
      ragSignal: round(between(rng, -cfg.ragSpread, cfg.ragSpread), 4),
      marketEvidenceCount: Math.floor(between(rng, 1, 8)),
      scenarioProfile: profile,
      scenarioSeed: Number(seed) || 42,
    };
  });
}

export function scenarioStats(items) {
  if (!items.length) return { items: 0, categories: 0, stores: 0, avgPrice: 0, totalDemand: 0, inventoryPressure: 0 };
  const sum = (key) => items.reduce((total, item) => total + Number(item[key] || 0), 0);
  return {
    items: items.length,
    categories: new Set(items.map((item) => item.category)).size,
    stores: new Set(items.map((item) => item.storeId)).size,
    avgPrice: sum("currentPrice") / items.length,
    totalDemand: sum("baselineQuantity"),
    inventoryPressure: items.filter((item) => item.inventory <= item.baselineQuantity * 1.05).length / items.length,
  };
}
