"use client";

import { useState, useEffect, useCallback } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const API = "http://localhost:8001";

interface BacktestMetrics {
  total_return: number;
  sharpe_ratio: number;
  win_rate: number;
  max_drawdown: number;
  trade_count: number;
}

interface BacktestTrade {
  entry_time: string;
  exit_time: string;
  direction: string;
  entry_price: number;
  exit_price: number;
  pnl: number;
  pnl_pct: number;
  reason: string;
}

interface ModelResult {
  model: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface BacktestResults {
  cumulative_pnl_curve: number[];
  metrics: BacktestMetrics;
  trades: BacktestTrade[];
  per_model?: ModelResult[];
}

function StatCard({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color: string;
}) {
  const colorMap: Record<string, string> = {
    emerald: "text-emerald-400",
    red: "text-red-400",
    amber: "text-amber-400",
    white: "text-white",
    cyan: "text-cyan-400",
  };
  return (
    <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className={`text-2xl font-mono font-bold ${colorMap[color] ?? "text-white"}`}>
        {value}
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const [data, setData] = useState<BacktestResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const res = await fetch(`${API}/backtest/results`);
      if (!res.ok) {
        setError("Backtest results not available");
        return;
      }
      const json = await res.json();
      setData(json);
      setError(null);
    } catch {
      setError("Backend unavailable");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  if (loading) {
    return (
      <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
        <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
          Loading analytics data...
        </div>
      </main>
    );
  }

  if (error || !data) {
    return (
      <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold text-white mb-6">Analytics</h1>
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-8 text-center">
          <p className="text-gray-500 text-sm mb-2">
            {error ?? "No backtest data available"}
          </p>
          <p className="text-gray-600 text-xs">
            Run the backtest first: cd backend && python -m ml.backtest
          </p>
        </div>
      </main>
    );
  }

  const metrics = data.metrics;
  const pnlCurve = data.cumulative_pnl_curve.map((val, idx) => ({
    index: idx,
    pnl: val,
  }));

  const totalReturnColor = metrics.total_return >= 0 ? "emerald" : "red";
  const sharpeColor = metrics.sharpe_ratio >= 1 ? "emerald" : metrics.sharpe_ratio >= 0.5 ? "amber" : "red";

  return (
    <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Analytics</h1>
        <p className="text-xs text-gray-500 mt-1">Backtest results and performance metrics</p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
        <StatCard
          label="Total Return"
          value={`${metrics.total_return >= 0 ? "+" : ""}${(metrics.total_return * 100).toFixed(2)}%`}
          color={totalReturnColor}
        />
        <StatCard
          label="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          color={sharpeColor}
        />
        <StatCard
          label="Win Rate"
          value={`${(metrics.win_rate * 100).toFixed(1)}%`}
          color={metrics.win_rate >= 0.5 ? "emerald" : "red"}
        />
        <StatCard
          label="Max Drawdown"
          value={`${(metrics.max_drawdown * 100).toFixed(2)}%`}
          color="red"
        />
        <StatCard
          label="Trade Count"
          value={String(metrics.trade_count)}
          color="white"
        />
      </div>

      {/* Cumulative PnL Chart */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 mb-6">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
          Cumulative PnL
        </h2>
        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={pnlCurve}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(55,65,81,0.3)" />
              <XAxis
                dataKey="index"
                stroke="#6b7280"
                fontSize={10}
                tickLine={false}
              />
              <YAxis
                stroke="#6b7280"
                fontSize={10}
                tickLine={false}
                tickFormatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border: "1px solid #374151",
                  borderRadius: "0.5rem",
                  fontSize: "12px",
                }}
                labelStyle={{ color: "#9ca3af" }}
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                formatter={(value: any) => [`${(Number(value) * 100).toFixed(2)}%`, "PnL"]}
              />
              <Line
                type="monotone"
                dataKey="pnl"
                stroke="#f59e0b"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: "#f59e0b" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison */}
      {data.per_model && data.per_model.length > 0 && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 mb-6">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Model Comparison
          </h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 px-3 text-xs text-gray-500 font-medium">Model</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">Accuracy</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">Precision</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">Recall</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">F1</th>
                </tr>
              </thead>
              <tbody>
                {data.per_model.map((m) => (
                  <tr key={m.model} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 px-3 text-white font-medium">{m.model}</td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      {(m.accuracy * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      {(m.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      {(m.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-amber-400">
                      {(m.f1 * 100).toFixed(1)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Recent Trades Table */}
      {data.trades && data.trades.length > 0 && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Recent Trades
          </h2>
          <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-gray-900">
                <tr className="border-b border-gray-800">
                  <th className="text-left py-2 px-3 text-xs text-gray-500 font-medium">Direction</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">Entry</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">Exit</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">PnL</th>
                  <th className="text-right py-2 px-3 text-xs text-gray-500 font-medium">PnL %</th>
                  <th className="text-left py-2 px-3 text-xs text-gray-500 font-medium">Reason</th>
                </tr>
              </thead>
              <tbody>
                {data.trades.slice().reverse().map((t, i) => (
                  <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                    <td className="py-2 px-3">
                      <span
                        className={`px-2 py-0.5 rounded text-xs font-semibold ${
                          t.direction === "BUY" || t.direction === "LONG"
                            ? "bg-emerald-500/15 text-emerald-400"
                            : "bg-red-500/15 text-red-400"
                        }`}
                      >
                        {t.direction}
                      </span>
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      ${t.entry_price.toFixed(2)}
                    </td>
                    <td className="py-2 px-3 text-right font-mono text-gray-300">
                      ${t.exit_price.toFixed(2)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right font-mono ${
                        t.pnl >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {t.pnl >= 0 ? "+" : ""}
                      ${t.pnl.toFixed(2)}
                    </td>
                    <td
                      className={`py-2 px-3 text-right font-mono ${
                        t.pnl_pct >= 0 ? "text-emerald-400" : "text-red-400"
                      }`}
                    >
                      {t.pnl_pct >= 0 ? "+" : ""}
                      {(t.pnl_pct * 100).toFixed(2)}%
                    </td>
                    <td className="py-2 px-3 text-gray-500 text-xs truncate max-w-[200px]">
                      {t.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </main>
  );
}
