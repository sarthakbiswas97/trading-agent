"use client";

import { useState, useCallback, useRef } from "react";
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

type ExecutionStep = "idle" | "encrypting" | "risk_check" | "ika_approval" | "complete" | "rejected";

const STEP_LABELS: Record<Exclude<ExecutionStep, "idle">, string> = {
  encrypting: "Encrypting trade parameters...",
  risk_check: "Running FHE risk comparison...",
  ika_approval: "Requesting dWallet signature...",
  complete: "Trade finalized",
  rejected: "Trade rejected",
};

const STEP_ORDER: Exclude<ExecutionStep, "idle" | "complete" | "rejected">[] = [
  "encrypting",
  "risk_check",
  "ika_approval",
];

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function StepIndicator({ step, currentStep, passed }: { step: string; currentStep: ExecutionStep; passed: boolean | null }) {
  const stepIndex = STEP_ORDER.indexOf(step as typeof STEP_ORDER[number]);
  const currentIndex = STEP_ORDER.indexOf(currentStep as typeof STEP_ORDER[number]);
  const isTerminal = currentStep === "complete" || currentStep === "rejected";

  let status: "pending" | "active" | "done" | "failed";
  if (isTerminal) {
    if (currentStep === "rejected" && step === "risk_check") {
      status = "failed";
    } else if (currentStep === "rejected" && stepIndex > 1) {
      status = "pending";
    } else {
      status = passed === false && step === "ika_approval" ? "pending" : "done";
    }
  } else if (stepIndex < currentIndex) {
    status = "done";
  } else if (stepIndex === currentIndex) {
    status = "active";
  } else {
    status = "pending";
  }

  const colorMap = {
    pending: "border-gray-700 bg-gray-800/50 text-gray-500",
    active: "border-amber-500/40 bg-amber-500/10 text-amber-400",
    done: "border-emerald-500/40 bg-emerald-500/10 text-emerald-400",
    failed: "border-red-500/40 bg-red-500/10 text-red-400",
  };

  const dotMap = {
    pending: "bg-gray-600",
    active: "bg-amber-400 animate-pulse",
    done: "bg-emerald-400",
    failed: "bg-red-400",
  };

  const label = STEP_LABELS[step as keyof typeof STEP_LABELS] ?? step;

  return (
    <div className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-all duration-300 ${colorMap[status]}`}>
      <span className={`block w-2 h-2 rounded-full shrink-0 ${dotMap[status]}`} />
      <span className="text-sm font-medium">{label}</span>
    </div>
  );
}

function TradeButton() {
  const [step, setStep] = useState<ExecutionStep>("idle");
  const [result, setResult] = useState<TradeResult | null>(null);
  const abortRef = useRef(false);

  const submitTrade = useCallback(async () => {
    abortRef.current = false;
    setResult(null);
    setStep("encrypting");

    // Fire the API call immediately, progress through steps with delays
    const fetchPromise = fetch("http://localhost:8001/trade/submit-demo", { method: "POST" })
      .then(async (res) => (res.ok ? ((await res.json()) as TradeResult) : null))
      .catch(() => null);

    // Step 1: encrypting (show for at least 1.2s)
    await delay(1200);
    if (abortRef.current) return;
    setStep("risk_check");

    // Step 2: risk check (show for at least 1.4s)
    await delay(1400);
    if (abortRef.current) return;
    setStep("ika_approval");

    // Step 3: ika approval -- wait for API response
    const data = await fetchPromise;
    // Small extra delay so the step is visible
    await delay(800);
    if (abortRef.current) return;

    if (data) {
      setResult(data);
      setStep(data.risk_passed ? "complete" : "rejected");
    } else {
      setStep("rejected");
    }
  }, []);

  const isRunning = step !== "idle" && step !== "complete" && step !== "rejected";

  return (
    <div className="mb-6 p-5 bg-gray-900 rounded-2xl border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">Execute Trade</h2>
        <button
          onClick={submitTrade}
          disabled={isRunning}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            isRunning
              ? "bg-gray-700 text-gray-400 cursor-wait"
              : "bg-amber-500/20 text-amber-400 hover:bg-amber-500/30 ring-1 ring-amber-500/30"
          }`}
        >
          {isRunning ? "Processing..." : "Submit Trade Proposal"}
        </button>
      </div>

      {/* Step-by-step progression */}
      {step !== "idle" && (
        <div className="space-y-2 mb-4">
          {STEP_ORDER.map((s) => (
            <StepIndicator
              key={s}
              step={s}
              currentStep={step}
              passed={result ? result.risk_passed : null}
            />
          ))}

          {/* Terminal step */}
          {(step === "complete" || step === "rejected") && (
            <div
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border transition-all duration-300 ${
                step === "complete"
                  ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-400"
                  : "border-red-500/40 bg-red-500/10 text-red-400"
              }`}
            >
              <span
                className={`block w-2 h-2 rounded-full shrink-0 ${
                  step === "complete" ? "bg-emerald-400" : "bg-red-400"
                }`}
              />
              <span className="text-sm font-bold">
                {step === "complete" ? "Trade finalized" : "Trade rejected"}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Result details */}
      {result && (step === "complete" || step === "rejected") && (
        <div
          className={`p-4 rounded-lg border ${
            result.risk_passed
              ? "bg-emerald-500/5 border-emerald-500/20"
              : "bg-red-500/5 border-red-500/20"
          }`}
        >
          <div className="flex items-center gap-2 mb-2">
            <span
              className={`text-lg font-bold ${
                result.risk_passed ? "text-emerald-400" : "text-red-400"
              }`}
            >
              {result.verdict}
            </span>
            <span className="text-xs text-gray-500">{result.trade.message}</span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
            <div>
              <span className="text-gray-500">Direction:</span>{" "}
              <span className="text-white">{result.trade.direction}</span>
            </div>
            <div>
              <span className="text-gray-500">Confidence:</span>{" "}
              <span className="text-white">
                {(result.trade.confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div>
              <span className="text-gray-500">Position:</span>{" "}
              <span className="text-white">{result.trade.position_bps}bps</span>
            </div>
            <div>
              <span className="text-gray-500">Price:</span>{" "}
              <span className="text-white">${result.trade.price.toFixed(2)}</span>
            </div>
          </div>
          {result.rejection_reason && (
            <p className="mt-2 text-xs text-red-400">
              Reason: {result.rejection_reason}
            </p>
          )}
          {result.onchain.tx_hash && (
            <p className="mt-2 text-xs text-gray-400">
              tx: <span className="text-violet-400 font-mono">{result.onchain.tx_hash}</span>
            </p>
          )}
          {result.onchain.explorer && (
            <a
              href={result.onchain.explorer}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-2 inline-block text-xs text-violet-400 hover:text-violet-300 underline decoration-dotted"
            >
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
