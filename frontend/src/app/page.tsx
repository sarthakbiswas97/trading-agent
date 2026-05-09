"use client";

import { useEffect, useState, useCallback } from "react";

const API_BASE = "http://localhost:8001";

interface AgentStatus {
  agent_name: string;
  status: string;
  latest_price: number;
  symbol: string;
}

interface DWalletStatus {
  dwallet: {
    enabled: boolean;
    initialized: boolean;
    dwallet_address: string | null;
    ika_program: string | null;
    cpi_authority: string | null;
  };
  risk_limits: {
    max_position_bps: number;
    max_daily_loss_bps: number;
    max_drawdown_bps: number;
  };
}

interface EncryptStatus {
  encrypt: {
    enabled: boolean;
    initialized: boolean;
    encrypt_program: string;
    encrypted_values_count: number;
  };
  encrypted_decisions: Array<{
    ciphertext_account: string;
    fhe_type: number;
    type_name: string;
  }>;
}

interface TradeResult {
  success: boolean;
  action: string;
  amount: number;
  price: number;
  reason: string;
  pnl: number | null;
}

interface ExecutorStatus {
  running: boolean;
  has_position: boolean;
  position: Record<string, unknown> | null;
  capital: { current: number; base: number; peak: number };
  risk: {
    current_drawdown_pct: number;
    max_drawdown_pct: number;
    throttle_factor: number;
    trading_enabled: boolean;
  };
  trades_today: number;
  daily_pnl_pct: number;
  recent_trades: TradeResult[];
}

function StatusBadge({ active, label }: { active: boolean; label: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${
        active
          ? "bg-emerald-500/15 text-emerald-400 ring-1 ring-emerald-500/30"
          : "bg-gray-700/50 text-gray-400 ring-1 ring-gray-600/30"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          active ? "bg-emerald-400 animate-pulse" : "bg-gray-500"
        }`}
      />
      {label}
    </span>
  );
}

function Card({
  title,
  children,
  badge,
}: {
  title: string;
  children: React.ReactNode;
  badge?: React.ReactNode;
}) {
  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-gray-300 uppercase tracking-wider">
          {title}
        </h2>
        {badge}
      </div>
      {children}
    </div>
  );
}

function MetricRow({
  label,
  value,
  sub,
  warn,
}: {
  label: string;
  value: string;
  sub?: string;
  warn?: boolean;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-800/50 last:border-0">
      <span className="text-sm text-gray-400">{label}</span>
      <div className="text-right">
        <span
          className={`text-sm font-mono font-medium ${
            warn ? "text-red-400" : "text-gray-100"
          }`}
        >
          {value}
        </span>
        {sub && <span className="block text-xs text-gray-500">{sub}</span>}
      </div>
    </div>
  );
}

function RiskBar({
  label,
  current,
  max,
}: {
  label: string;
  current: number;
  max: number;
}) {
  const pct = Math.min((current / max) * 100, 100);
  const danger = pct > 80;
  const warning = pct > 50;

  return (
    <div className="py-2">
      <div className="flex items-center justify-between mb-1.5">
        <span className="text-xs text-gray-400">{label}</span>
        <span
          className={`text-xs font-mono ${
            danger
              ? "text-red-400"
              : warning
                ? "text-amber-400"
                : "text-gray-300"
          }`}
        >
          {current.toFixed(1)}% / {max.toFixed(1)}%
        </span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            danger
              ? "bg-red-500"
              : warning
                ? "bg-amber-500"
                : "bg-emerald-500"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export default function Dashboard() {
  const [agent, setAgent] = useState<AgentStatus | null>(null);
  const [dwallet, setDwallet] = useState<DWalletStatus | null>(null);
  const [encrypt, setEncrypt] = useState<EncryptStatus | null>(null);
  const [executor, setExecutor] = useState<ExecutorStatus | null>(null);
  const [connected, setConnected] = useState(false);
  const [lastUpdate, setLastUpdate] = useState("");

  const fetchData = useCallback(async () => {
    try {
      const [agentRes, dwalletRes, encryptRes, executorRes] =
        await Promise.allSettled([
          fetch(`${API_BASE}/agent/status`),
          fetch(`${API_BASE}/agent/dwallet`),
          fetch(`${API_BASE}/agent/encrypt`),
          fetch(`${API_BASE}/trades/status`),
        ]);

      if (agentRes.status === "fulfilled" && agentRes.value.ok)
        setAgent(await agentRes.value.json());
      if (dwalletRes.status === "fulfilled" && dwalletRes.value.ok)
        setDwallet(await dwalletRes.value.json());
      if (encryptRes.status === "fulfilled" && encryptRes.value.ok)
        setEncrypt(await encryptRes.value.json());
      if (executorRes.status === "fulfilled" && executorRes.value.ok)
        setExecutor(await executorRes.value.json());

      setConnected(true);
      setLastUpdate(new Date().toLocaleTimeString());
    } catch {
      setConnected(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, [fetchData]);

  return (
    <main className="min-h-screen p-6 max-w-7xl mx-auto">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <h1 className="text-2xl font-bold text-white">
              VAPM
              <span className="text-gray-500 font-normal ml-2 text-lg">
                Verifiable AI Portfolio Manager
              </span>
            </h1>
            <p className="text-sm text-gray-500 mt-1">
              Encrypted strategy + Cryptographic risk guardrails on Solana
            </p>
          </div>
          <div className="flex items-center gap-3">
            <StatusBadge
              active={connected}
              label={connected ? "Connected" : "Disconnected"}
            />
            {lastUpdate && (
              <span className="text-xs text-gray-600">{lastUpdate}</span>
            )}
          </div>
        </div>

        <div className="flex gap-2 mt-4 flex-wrap">
          <StatusBadge
            active={!!encrypt?.encrypt?.initialized}
            label="Encrypt FHE"
          />
          <StatusBadge
            active={!!dwallet?.dwallet?.initialized}
            label="Ika dWallet"
          />
          <StatusBadge
            active={!!agent}
            label={`SOL/USDC @ $${agent?.latest_price?.toFixed(2) ?? "---"}`}
          />
        </div>
      </header>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        {/* On-Chain Risk Limits */}
        <Card
          title="On-Chain Risk Limits"
          badge={
            <span className="text-xs text-violet-400 bg-violet-500/10 px-2 py-0.5 rounded-full ring-1 ring-violet-500/30">
              Ika Enforced
            </span>
          }
        >
          {dwallet ? (
            <>
              <RiskBar
                label="Position Size"
                current={executor?.has_position ? 5.0 : 0}
                max={dwallet.risk_limits.max_position_bps / 100}
              />
              <RiskBar
                label="Daily Loss"
                current={Math.abs(executor?.daily_pnl_pct ?? 0) * 100}
                max={dwallet.risk_limits.max_daily_loss_bps / 100}
              />
              <RiskBar
                label="Drawdown"
                current={(executor?.risk?.current_drawdown_pct ?? 0) * 100}
                max={dwallet.risk_limits.max_drawdown_bps / 100}
              />
              <div className="mt-3 p-2.5 bg-gray-800/50 rounded-lg">
                <p className="text-xs text-gray-500">
                  Limits stored on-chain in AgentState PDA. The dWallet cannot
                  sign trades that violate these limits.
                </p>
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>

        {/* Encrypted Decisions */}
        <Card
          title="Encrypted Decisions"
          badge={
            <span className="text-xs text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded-full ring-1 ring-cyan-500/30">
              FHE Private
            </span>
          }
        >
          {encrypt ? (
            <>
              <MetricRow
                label="Encrypted Values"
                value={String(encrypt.encrypt.encrypted_values_count)}
              />
              <MetricRow
                label="Encrypt Program"
                value={
                  encrypt.encrypt.encrypt_program
                    ? `${encrypt.encrypt.encrypt_program.slice(0, 8)}...`
                    : "---"
                }
                sub="Solana Devnet"
              />
              <div className="mt-3 space-y-1.5 max-h-32 overflow-y-auto">
                {encrypt.encrypted_decisions.map((d, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-2 bg-gray-800/30 rounded text-xs"
                  >
                    <code className="text-cyan-400">
                      {d.ciphertext_account.slice(0, 12)}...
                    </code>
                    <span className="text-gray-500">{d.type_name}</span>
                  </div>
                ))}
                {encrypt.encrypted_decisions.length === 0 && (
                  <p className="text-xs text-gray-600">
                    No encrypted decisions yet. Signals will be encrypted before
                    on-chain storage.
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>

        {/* Agent Status */}
        <Card title="Agent Status">
          {agent && executor ? (
            <>
              <MetricRow label="Agent" value={agent.agent_name} />
              <MetricRow
                label="Capital"
                value={`$${executor.capital.current.toLocaleString()}`}
                sub={`Peak: $${executor.capital.peak.toLocaleString()}`}
              />
              <MetricRow
                label="Trades Today"
                value={String(executor.trades_today)}
              />
              <MetricRow
                label="Daily PnL"
                value={`${(executor.daily_pnl_pct * 100).toFixed(2)}%`}
                warn={executor.daily_pnl_pct < -0.02}
              />
              <MetricRow
                label="Trading"
                value={executor.risk.trading_enabled ? "Enabled" : "Halted"}
                warn={!executor.risk.trading_enabled}
              />
            </>
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>

        {/* dWallet Custody */}
        <Card title="dWallet Custody">
          {dwallet ? (
            <>
              <MetricRow
                label="Status"
                value={
                  dwallet.dwallet.initialized ? "Active" : "Fallback Mode"
                }
              />
              <MetricRow
                label="Ika Program"
                value={
                  dwallet.dwallet.ika_program
                    ? `${dwallet.dwallet.ika_program.slice(0, 8)}...`
                    : "---"
                }
              />
              <MetricRow
                label="CPI Authority"
                value={
                  dwallet.dwallet.cpi_authority
                    ? `${dwallet.dwallet.cpi_authority.slice(0, 8)}...`
                    : "---"
                }
              />
              <MetricRow
                label="dWallet Address"
                value={
                  dwallet.dwallet.dwallet_address
                    ? `${dwallet.dwallet.dwallet_address.slice(0, 8)}...`
                    : "Not configured"
                }
              />
            </>
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>

        {/* Position */}
        <Card title="Current Position">
          {executor ? (
            executor.has_position && executor.position ? (
              <>
                <MetricRow
                  label="Side"
                  value={String(executor.position.side ?? "---")}
                />
                <MetricRow
                  label="Size"
                  value={`${Number(executor.position.size ?? 0).toFixed(4)} SOL`}
                />
                <MetricRow
                  label="Entry"
                  value={`$${Number(executor.position.entry_price ?? 0).toFixed(2)}`}
                />
                <MetricRow
                  label="Unrealized PnL"
                  value={`$${Number(executor.position.unrealized_pnl ?? 0).toFixed(2)}`}
                  warn={Number(executor.position.unrealized_pnl ?? 0) < 0}
                />
              </>
            ) : (
              <p className="text-sm text-gray-500 py-4 text-center">
                No open position
              </p>
            )
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>

        {/* Recent Trades */}
        <Card title="Recent Trades">
          {executor ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {executor.recent_trades.length > 0 ? (
                executor.recent_trades
                  .slice()
                  .reverse()
                  .map((trade, i) => (
                    <div
                      key={i}
                      className={`p-2.5 rounded-lg border text-xs ${
                        trade.success
                          ? trade.action === "BUY"
                            ? "border-emerald-800/50 bg-emerald-900/10"
                            : "border-amber-800/50 bg-amber-900/10"
                          : "border-red-800/50 bg-red-900/10"
                      }`}
                    >
                      <div className="flex justify-between items-center">
                        <span
                          className={`font-semibold ${
                            !trade.success
                              ? "text-red-400"
                              : trade.action === "BUY"
                                ? "text-emerald-400"
                                : "text-amber-400"
                          }`}
                        >
                          {trade.success ? trade.action : "REJECTED"}
                        </span>
                        {trade.success && (
                          <span className="text-gray-400">
                            {trade.amount.toFixed(4)} SOL @ $
                            {trade.price.toFixed(2)}
                          </span>
                        )}
                      </div>
                      <p className="text-gray-500 mt-1 truncate">
                        {trade.reason}
                      </p>
                      {trade.pnl !== null && (
                        <p
                          className={`mt-0.5 ${
                            trade.pnl >= 0 ? "text-emerald-500" : "text-red-500"
                          }`}
                        >
                          PnL: ${trade.pnl.toFixed(2)}
                        </p>
                      )}
                    </div>
                  ))
              ) : (
                <p className="text-sm text-gray-500 py-4 text-center">
                  No trades yet
                </p>
              )}
            </div>
          ) : (
            <p className="text-sm text-gray-500">Loading...</p>
          )}
        </Card>
      </div>

      {/* Architecture Footer */}
      <footer className="mt-8 p-5 bg-gray-900 border border-gray-800 rounded-xl">
        <h3 className="text-sm font-semibold text-gray-300 uppercase tracking-wider mb-3">
          Architecture
        </h3>
        <pre className="font-mono text-xs text-gray-500 leading-relaxed overflow-x-auto">
          {`Market Data (Birdeye/Jupiter) -> Feature Engine -> XGBoost ML -> Strategy
                                                                         |
                                                              [Encrypt FHE] confidence + risk
                                                              scores encrypted (anti-frontrun)
                                                                         |
                                                                         v
                                                       +--On-Chain Risk Check (Anchor)--+
                                                       |  pos <= max?  loss <= max?      |
                                                       |  drawdown <= max?               |
                                                       +------+----------------+---------+
                                                              |                |
                                                            PASS             FAIL
                                                              |                |
                                                       [Ika dWallet]     [REJECTED]
                                                       MPC signs trade   trade blocked
                                                              |
                                                       Jupiter Aggregator -> SOL/USDC swap`}
        </pre>
      </footer>
    </main>
  );
}
