import { isFin } from "./engine.js";

const quantile = (values, q) => {
  const sorted = values.filter(isFin).sort((a, b) => a - b);
  if (!sorted.length) return NaN;
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos), hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
};

const mean = (values) => {
  const valid = values.filter(isFin);
  return valid.length ? valid.reduce((total, value) => total + value, 0) / valid.length : NaN;
};

export function analyzeInput(items = []) {
  const numericFields = ["currentPrice", "unitCost", "competitorPrice", "inventory", "baselineQuantity", "elasticity", "promotionUpliftRate", "confidence", "ragSignal"];
  const missingByField = Object.fromEntries(numericFields.map((field) => [field, items.filter((item) => !isFin(Number(item[field]))).length]));
  const margins = items.map((item) => item.currentPrice > 0 ? (item.currentPrice - item.unitCost) / item.currentPrice : NaN);
  const pricePositions = items.map((item) => item.competitorPrice > 0 ? item.currentPrice / item.competitorPrice - 1 : NaN);
  const inventoryRatios = items.map((item) => item.baselineQuantity > 0 ? item.inventory / item.baselineQuantity : NaN);
  const causalRows = items.filter((item) => item.elasticityIsCausal || ["product_iv", "pooled_iv"].includes(item.elasticitySource)).length;
  const promoReady = items.filter((item) => item.promotionModelReliable !== false).length;
  const invalid = items.filter((item) => !(item.currentPrice > 0) || !(item.unitCost >= 0) || item.unitCost > item.currentPrice || !(item.baselineQuantity >= 0) || !(item.inventory >= 0) || !(item.elasticity < 0)).length;
  const duplicates = items.length - new Set(items.map((item) => item.itemId)).size;

  const categoryMap = {};
  items.forEach((item) => {
    const category = item.category || "Uncategorized";
    if (!categoryMap[category]) categoryMap[category] = { category, items: 0, prices: [], margins: [], demand: 0, inventoryRisk: 0, causal: 0, promoReady: 0 };
    const row = categoryMap[category];
    row.items += 1;
    row.prices.push(Number(item.currentPrice));
    row.margins.push(item.currentPrice > 0 ? (item.currentPrice - item.unitCost) / item.currentPrice : NaN);
    row.demand += Number(item.baselineQuantity) || 0;
    if (item.inventory <= item.baselineQuantity * 1.05) row.inventoryRisk += 1;
    if (item.elasticityIsCausal || ["product_iv", "pooled_iv"].includes(item.elasticitySource)) row.causal += 1;
    if (item.promotionModelReliable !== false) row.promoReady += 1;
  });
  const categories = Object.values(categoryMap).map((row) => ({
    category: row.category, items: row.items, avgPrice: mean(row.prices), avgMargin: mean(row.margins),
    totalDemand: row.demand, inventoryRiskRate: row.items ? row.inventoryRisk / row.items : 0,
    causalRate: row.items ? row.causal / row.items : 0, promotionReadyRate: row.items ? row.promoReady / row.items : 0,
  })).sort((a, b) => b.totalDemand - a.totalDemand);

  const prices = items.map((item) => Number(item.currentPrice));
  const demand = items.map((item) => Number(item.baselineQuantity));
  const elasticities = items.map((item) => Number(item.elasticity));
  const completenessCells = items.length * numericFields.length;
  const missingCells = Object.values(missingByField).reduce((total, value) => total + value, 0);
  return {
    rows: items.length,
    categories,
    categoryCount: categories.length,
    storeCount: new Set(items.map((item) => item.storeId).filter(Boolean)).size,
    productCount: new Set(items.map((item) => item.productId).filter(Boolean)).size,
    causalRate: items.length ? causalRows / items.length : 0,
    promotionReadyRate: items.length ? promoReady / items.length : 0,
    inventoryRiskRate: items.length ? inventoryRatios.filter((ratio) => isFin(ratio) && ratio <= 1.05).length / items.length : 0,
    invalidRows: invalid,
    duplicateIds: Math.max(0, duplicates),
    completenessRate: completenessCells ? 1 - missingCells / completenessCells : 1,
    missingByField,
    avgPrice: mean(prices), avgMargin: mean(margins), avgCompetitorPosition: mean(pricePositions),
    totalDemand: demand.filter(isFin).reduce((total, value) => total + value, 0),
    priceRange: [quantile(prices, 0), quantile(prices, 1)], priceP50: quantile(prices, .5),
    demandRange: [quantile(demand, 0), quantile(demand, 1)], demandP50: quantile(demand, .5),
    elasticityRange: [quantile(elasticities, 0), quantile(elasticities, 1)], elasticityP50: quantile(elasticities, .5),
  };
}

