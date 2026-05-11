"use client";

import { useEffect, useRef, useState, useCallback } from "react";

interface Candle {
  open_time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const API = "http://localhost:8001";

export default function PriceChart() {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<ReturnType<typeof import("lightweight-charts").createChart> | null>(null);
  const seriesRef = useRef<ReturnType<ReturnType<typeof import("lightweight-charts").createChart>["addSeries"]> | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchCandles = useCallback(async () => {
    try {
      const res = await fetch(`${API}/market/candles?limit=100`);
      if (!res.ok) {
        setError("Failed to load candle data");
        return null;
      }
      const data = await res.json();
      return data.candles as Candle[];
    } catch {
      setError("Backend unavailable");
      return null;
    }
  }, []);

  useEffect(() => {
    let mounted = true;
    let intervalId: ReturnType<typeof setInterval>;

    const init = async () => {
      if (!containerRef.current) return;

      const { createChart, LineSeries } = await import("lightweight-charts");

      if (!mounted || !containerRef.current) return;

      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height: 300,
        layout: {
          background: { color: "transparent" },
          textColor: "#9ca3af",
          fontSize: 11,
        },
        grid: {
          vertLines: { color: "rgba(55, 65, 81, 0.3)" },
          horzLines: { color: "rgba(55, 65, 81, 0.3)" },
        },
        crosshair: {
          vertLine: { color: "rgba(245, 158, 11, 0.3)" },
          horzLine: { color: "rgba(245, 158, 11, 0.3)" },
        },
        rightPriceScale: {
          borderColor: "rgba(55, 65, 81, 0.5)",
        },
        timeScale: {
          borderColor: "rgba(55, 65, 81, 0.5)",
          timeVisible: true,
          secondsVisible: false,
        },
      });

      const series = chart.addSeries(LineSeries, {
        color: "#f59e0b",
        lineWidth: 2,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: "#f59e0b",
        crosshairMarkerBackgroundColor: "#1f2937",
      });

      chartRef.current = chart;
      seriesRef.current = series;

      // Responsive resize
      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          chart.applyOptions({ width: entry.contentRect.width });
        }
      });
      resizeObserver.observe(containerRef.current);

      // Load initial data
      const candles = await fetchCandles();
      if (candles && mounted) {
        const lineData = candles.map((c) => ({
          time: Math.floor(c.open_time) as import("lightweight-charts").UTCTimestamp,
          value: c.close,
        }));
        series.setData(lineData);
        chart.timeScale().fitContent();
        setLoading(false);
        setError(null);
      } else if (mounted) {
        setLoading(false);
      }

      // Poll for updates
      intervalId = setInterval(async () => {
        const fresh = await fetchCandles();
        if (fresh && mounted && seriesRef.current) {
          const lineData = fresh.map((c) => ({
            time: Math.floor(c.open_time) as import("lightweight-charts").UTCTimestamp,
            value: c.close,
          }));
          seriesRef.current.setData(lineData);
        }
      }, 10000);

      return () => {
        resizeObserver.disconnect();
      };
    };

    init();

    return () => {
      mounted = false;
      clearInterval(intervalId);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
      }
    };
  }, [fetchCandles]);

  return (
    <div className="mb-6 bg-gray-900 rounded-2xl border border-gray-800 p-5">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
          Price Chart
        </h2>
        <span className="text-[10px] text-gray-600">
          Auto-updates every 10s
        </span>
      </div>

      {loading && !error && (
        <div className="h-[300px] flex items-center justify-center text-sm text-gray-600">
          Loading chart data...
        </div>
      )}

      {error && (
        <div className="h-[300px] flex items-center justify-center text-sm text-gray-600">
          {error} -- chart will load when backend is available
        </div>
      )}

      <div
        ref={containerRef}
        className={loading || error ? "h-0 overflow-hidden" : ""}
      />
    </div>
  );
}
