"use client";

import { useMemo } from "react";

interface ShapEntry {
  feature: string;
  value: number;
  impact: string;
}

interface ShapChartProps {
  shapExplanation: Record<string, { value: number; impact: string }> | null | undefined;
  maxFeatures?: number;
  compact?: boolean;
}

export default function ShapChart({
  shapExplanation,
  maxFeatures = 10,
  compact = false,
}: ShapChartProps) {
  const sortedFeatures: ShapEntry[] = useMemo(() => {
    if (!shapExplanation) return [];

    return Object.entries(shapExplanation)
      .map(([feature, data]) => ({
        feature,
        value: data.value,
        impact: data.impact,
      }))
      .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
      .slice(0, maxFeatures);
  }, [shapExplanation, maxFeatures]);

  if (sortedFeatures.length === 0) {
    return (
      <div className="text-center py-6 text-gray-600 text-sm">
        No SHAP data available. Run a prediction first.
      </div>
    );
  }

  const maxAbsValue = Math.max(...sortedFeatures.map((f) => Math.abs(f.value)));

  return (
    <div className={compact ? "space-y-1.5" : "space-y-2.5"}>
      {sortedFeatures.map((entry) => {
        const absVal = Math.abs(entry.value);
        const barWidth = maxAbsValue > 0 ? (absVal / maxAbsValue) * 100 : 0;
        const isPositive = entry.value > 0;

        return (
          <div key={entry.feature} className="group">
            <div className="flex items-center justify-between mb-0.5">
              <span
                className={`font-mono truncate ${
                  compact ? "text-[10px] text-gray-400" : "text-xs text-gray-300"
                }`}
                title={entry.feature}
              >
                {entry.feature}
              </span>
              <span
                className={`font-mono shrink-0 ml-2 ${
                  compact ? "text-[10px]" : "text-xs"
                } ${isPositive ? "text-emerald-400" : "text-red-400"}`}
              >
                {isPositive ? "+" : ""}
                {entry.value.toFixed(4)}
              </span>
            </div>
            <div
              className={`w-full bg-gray-800 rounded-full overflow-hidden ${
                compact ? "h-1.5" : "h-2.5"
              }`}
            >
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  isPositive
                    ? "bg-emerald-500"
                    : "bg-red-500"
                }`}
                style={{ width: `${Math.max(barWidth, 2)}%` }}
              />
            </div>
          </div>
        );
      })}
      {!compact && (
        <div className="flex items-center justify-between pt-2 text-[10px] text-gray-600">
          <span>Red = pushes DOWN</span>
          <span>Green = pushes UP</span>
        </div>
      )}
    </div>
  );
}
