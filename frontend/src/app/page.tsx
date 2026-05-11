"use client";

import { useState } from "react";
import { useVAPMData } from "@/lib/api";
import HeroBanner from "@/components/HeroBanner";
import PriceChart from "@/components/PriceChart";
import ShapChart from "@/components/ShapChart";
import Pipeline from "@/components/Pipeline";
import RejectionShowcase from "@/components/RejectionShowcase";
import EncryptCard from "@/components/EncryptCard";
import DWalletCard from "@/components/DWalletCard";
import ActivityFeed from "@/components/ActivityFeed";

interface TradeResult {
  verdict: string;
  risk_passed: boolean;
  rejection_reason: string | null;
  trade: { direction: string; price: number; confidence: number; message: string; message_hash: string; position_bps: number };
  onchain: { tx_hash: string | null; explorer: string | null };
}

function TradeButton() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<TradeResult | null>(null);

  const submitTrade = async () => {
    setLoading(true);
    setResult(null);
    try {
      const res = await fetch("http://localhost:8001/trade/submit-demo", { method: "POST" });
      if (res.ok) setResult(await res.json());
    } catch { /* ignore */ }
    setLoading(false);
  };

  return (
    <div className="mb-6 p-5 bg-gray-900 rounded-2xl border border-gray-800">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Execute Trade</h2>
        <button
          onClick={submitTrade}
          disabled={loading}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            loading
              ? "bg-gray-700 text-gray-400 cursor-wait"
              : "bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 ring-1 ring-amber-500/30"
          }`}
        >
          {loading ? "Submitting..." : "Submit Trade Proposal"}
        </button>
      </div>
      {result && (
        <div className={`p-4 rounded-lg border ${
          result.risk_passed
            ? "bg-emerald-500/5 border-emerald-500/20"
            : "bg-red-500/5 border-red-500/20"
        }`}>
          <div className="flex items-center gap-2 mb-2">
            <span className={`text-lg font-bold ${result.risk_passed ? "text-emerald-400" : "text-red-400"}`}>
              {result.verdict}
            </span>
            <span className="text-xs text-gray-500">{result.trade.message}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div><span className="text-gray-500">Direction:</span> <span className="text-white">{result.trade.direction}</span></div>
            <div><span className="text-gray-500">Confidence:</span> <span className="text-white">{(result.trade.confidence * 100).toFixed(0)}%</span></div>
            <div><span className="text-gray-500">Position:</span> <span className="text-white">{result.trade.position_bps}bps</span></div>
            <div><span className="text-gray-500">Price:</span> <span className="text-white">${result.trade.price.toFixed(2)}</span></div>
          </div>
          {result.rejection_reason && (
            <p className="mt-2 text-xs text-red-400">Reason: {result.rejection_reason}</p>
          )}
          {result.onchain.explorer && (
            <a href={result.onchain.explorer} target="_blank" rel="noopener noreferrer"
               className="mt-2 inline-block text-xs text-violet-400 hover:text-violet-300 underline decoration-dotted">
              View on Solana Explorer
            </a>
          )}
        </div>
      )}
    </div>
  );
}

export default function Dashboard() {
  const data = useVAPMData();

  const shap = data.prediction?.prediction?.shap_explanation;

  return (
    <main className="min-h-screen p-4 md:p-6 max-w-7xl mx-auto">
      {/* Zone 1: Hero */}
      <HeroBanner
        agent={data.agent}
        prediction={data.prediction}
        encrypt={data.encrypt}
        dwallet={data.dwallet}
      />

      {/* Zone 1.5: Price Chart */}
      <PriceChart />

      {/* Zone 1.75: SHAP Summary (compact top 3 features) */}
      {shap && Object.keys(shap).length > 0 && (
        <div className="mb-6 p-5 bg-gray-900 rounded-2xl border border-gray-800">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              Top Prediction Drivers
            </h2>
            <a
              href="/model"
              className="text-[10px] text-amber-400 hover:text-amber-300 underline decoration-dotted"
            >
              View all features
            </a>
          </div>
          <ShapChart shapExplanation={shap} maxFeatures={3} compact />
        </div>
      )}

      {/* Zone 2: Pipeline */}
      <Pipeline
        agent={data.agent}
        prediction={data.prediction}
        encrypt={data.encrypt}
        dwallet={data.dwallet}
        executor={data.executor}
      />

      {/* Zone 3: Rejection Showcase */}
      <RejectionShowcase
        executor={data.executor}
        dwallet={data.dwallet}
      />

      {/* Zone 4: Integration Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-6">
        <EncryptCard
          encrypt={data.encrypt}
          prediction={data.prediction}
          live={data.live}
        />
        <DWalletCard
          dwallet={data.dwallet}
          executor={data.executor}
          live={data.live}
        />
      </div>

      {/* Zone 5: Execute Trade */}
      <TradeButton />

      {/* Zone 6: Activity Feed */}
      <ActivityFeed executor={data.executor} />

      {/* On-chain verification links */}
      {data.live?.agent && (
        <div className="mt-6 p-4 bg-gray-900 rounded-2xl border border-gray-800">
          <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Verify on Solana Explorer
          </h3>
          <div className="flex flex-wrap gap-3 text-xs">
            <a href={data.live.program_explorer ?? "#"} target="_blank" rel="noopener noreferrer"
               className="text-gray-400 hover:text-white underline decoration-dotted">
              Program
            </a>
            <a href={data.live.agent.explorer} target="_blank" rel="noopener noreferrer"
               className="text-violet-400 hover:text-violet-300 underline decoration-dotted">
              Agent PDA
            </a>
            <a href={data.live.agent.dwallet_explorer} target="_blank" rel="noopener noreferrer"
               className="text-violet-400 hover:text-violet-300 underline decoration-dotted">
              dWallet
            </a>
            {data.live.trades.map((t, i) => (
              <a key={i} href={t.explorer} target="_blank" rel="noopener noreferrer"
                 className={`underline decoration-dotted ${t.verdict_code === 1 ? "text-emerald-400" : t.verdict_code === 2 ? "text-red-400" : "text-gray-400"}`}>
                Trade #{t.index} ({t.verdict})
              </a>
            ))}
          </div>
        </div>
      )}

      {/* Connection status */}
      {!data.connected && (
        <div className="fixed bottom-4 right-4 bg-red-500/90 text-white text-xs px-3 py-2 rounded-lg">
          Backend disconnected -- start with: cd backend && uvicorn main:app --port 8001
        </div>
      )}
    </main>
  );
}
