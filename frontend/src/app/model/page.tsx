"use client";

import { useState, useEffect, useCallback } from "react";
import ShapChart from "@/components/ShapChart";
import type { PredictionResponse } from "@/lib/types";

const API = "http://localhost:8001";

interface ModelInfo {
  name: string;
  version: string;
  accuracy: number;
  feature_count: number;
  feature_importance?: Record<string, number>;
}

export default function ModelPage() {
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const [predRes, modelRes] = await Promise.all([
        fetch(`${API}/predict`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
        fetch(`${API}/predict/model`).then((r) => (r.ok ? r.json() : null)).catch(() => null),
      ]);

      setPrediction(predRes);
      setModelInfo(modelRes);
      setError(predRes || modelRes ? null : "Backend unavailable");
    } catch {
      setError("Failed to fetch model data");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const id = setInterval(fetchData, 10000);
    return () => clearInterval(id);
  }, [fetchData]);

  if (loading) {
    return (
      <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
        <div className="flex items-center justify-center h-64 text-gray-600 text-sm">
          Loading model data...
        </div>
      </main>
    );
  }

  const pred = prediction?.prediction;
  const shap = pred?.shap_explanation;

  // Sort feature importance if available
  const sortedImportance = modelInfo?.feature_importance
    ? Object.entries(modelInfo.feature_importance)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 15)
    : [];

  const maxImportance =
    sortedImportance.length > 0
      ? Math.max(...sortedImportance.map(([, v]) => v))
      : 0;

  return (
    <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white">Model</h1>
        <p className="text-xs text-gray-500 mt-1">
          Prediction details and SHAP explanations
        </p>
      </div>

      {error && !prediction && !modelInfo && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-8 text-center">
          <p className="text-gray-500 text-sm mb-2">{error}</p>
          <p className="text-gray-600 text-xs">
            Start the backend: cd backend && uvicorn main:app --port 8001
          </p>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-6">
        {/* Current Prediction */}
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Current Prediction
          </h2>
          {pred ? (
            <div className="space-y-4">
              {/* Direction + Confidence */}
              <div className="flex items-center gap-4">
                <div
                  className={`w-16 h-16 rounded-xl flex items-center justify-center text-2xl font-bold ${
                    pred.direction === "UP"
                      ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/30"
                      : "bg-red-500/10 text-red-400 border border-red-500/30"
                  }`}
                >
                  {pred.direction === "UP" ? "\u2191" : "\u2193"}
                </div>
                <div>
                  <div className="text-xl font-bold text-white">{pred.direction}</div>
                  <div className="text-sm text-gray-400">
                    {(pred.confidence * 100).toFixed(1)}% confidence
                  </div>
                </div>
              </div>

              {/* Probability bar */}
              <div>
                <div className="flex justify-between mb-1 text-xs text-gray-500">
                  <span>P(DOWN)</span>
                  <span>P(UP)</span>
                </div>
                <div className="h-3 bg-gray-800 rounded-full overflow-hidden flex">
                  <div
                    className="h-full bg-red-500 transition-all duration-500"
                    style={{
                      width: `${(1 - pred.probability_up) * 100}%`,
                    }}
                  />
                  <div
                    className="h-full bg-emerald-500 transition-all duration-500"
                    style={{ width: `${pred.probability_up * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1 text-xs font-mono">
                  <span className="text-red-400">
                    {((1 - pred.probability_up) * 100).toFixed(1)}%
                  </span>
                  <span className="text-emerald-400">
                    {(pred.probability_up * 100).toFixed(1)}%
                  </span>
                </div>
              </div>

              {/* Symbol */}
              <div className="pt-2 border-t border-gray-800 text-xs text-gray-500">
                Symbol: <span className="text-white font-mono">{prediction?.symbol}</span>
              </div>
            </div>
          ) : (
            <div className="text-center py-8 text-gray-600 text-sm">
              No prediction available. The model will generate predictions when
              market data is loaded.
            </div>
          )}
        </div>

        {/* Model Info */}
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Model Info
          </h2>
          {modelInfo ? (
            <div className="space-y-3">
              <InfoRow label="Name" value={modelInfo.name} />
              <InfoRow label="Version" value={modelInfo.version} />
              <InfoRow
                label="Accuracy"
                value={`${(modelInfo.accuracy * 100).toFixed(1)}%`}
                color={
                  modelInfo.accuracy >= 0.6
                    ? "text-emerald-400"
                    : "text-amber-400"
                }
              />
              <InfoRow
                label="Features"
                value={String(modelInfo.feature_count)}
              />
            </div>
          ) : (
            <div className="text-center py-8 text-gray-600 text-sm">
              Model info unavailable. Check the /predict/model endpoint.
            </div>
          )}
        </div>
      </div>

      {/* SHAP Explanation */}
      <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5 mb-6">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
          SHAP Feature Explanation
        </h2>
        <ShapChart shapExplanation={shap ?? null} maxFeatures={10} />
      </div>

      {/* Feature Importance */}
      {sortedImportance.length > 0 && (
        <div className="bg-gray-900 rounded-2xl border border-gray-800 p-5">
          <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-4">
            Feature Importance (Global)
          </h2>
          <div className="space-y-2">
            {sortedImportance.map(([feature, importance]) => {
              const barWidth =
                maxImportance > 0 ? (importance / maxImportance) * 100 : 0;
              return (
                <div key={feature}>
                  <div className="flex items-center justify-between mb-0.5">
                    <span className="text-xs font-mono text-gray-300 truncate">
                      {feature}
                    </span>
                    <span className="text-xs font-mono text-amber-400 shrink-0 ml-2">
                      {importance.toFixed(4)}
                    </span>
                  </div>
                  <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-amber-500/70 rounded-full transition-all duration-500"
                      style={{ width: `${Math.max(barWidth, 2)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </main>
  );
}

function InfoRow({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800/50">
      <span className="text-xs text-gray-500">{label}</span>
      <span className={`text-sm font-mono ${color ?? "text-white"}`}>
        {value}
      </span>
    </div>
  );
}
