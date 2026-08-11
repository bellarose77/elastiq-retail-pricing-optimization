import { isFin } from "./engine.js";

export const fmtMoney = (x, d = 0) =>
  isFin(x) ? "$" + x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";
export const fmtPrice = (x) => (isFin(x) ? "$" + x.toFixed(2) : "—");
export const fmtPct = (x, d = 1) => (isFin(x) ? (x >= 0 ? "+" : "") + (x * 100).toFixed(d) + "%" : "—");
export const fmtPctPlain = (x, d = 0) => (isFin(x) ? (x * 100).toFixed(d) + "%" : "—");
export const fmtNum = (x, d = 0) =>
  isFin(x) ? x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }) : "—";

export const ACTION_LABEL = {
  increase_price: "Increase",
  decrease_price: "Decrease",
  hold_price: "Hold",
  review: "Review",
};
