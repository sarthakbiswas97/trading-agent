"use client";

export default function PresentationPage() {
  return (
    <>
      <style>{`nav { display: none !important; }`}</style>
      <div className="bg-gray-950 text-gray-100 min-h-screen">
        {/* Slide 1: Title */}
        <section className="py-20 px-6 flex flex-col items-center justify-center min-h-screen text-center">
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight mb-6">
            <span className="text-amber-400">VAPM</span>{" "}
            <span className="text-gray-400">&mdash;</span> Verifiable AI
            Portfolio Manager
          </h1>
          <p className="text-lg md:text-xl text-gray-300 max-w-3xl mb-4">
            Encrypted Risk Enforcement + Distributed Custody on Solana
          </p>
          <p className="text-lg text-violet-400 font-medium mb-8">
            Frontier Hackathon &mdash; Encrypt + Ika Track
          </p>
          <p className="text-base text-gray-500 mb-10">
            Built by Sarthak Biswas
          </p>
          <div className="flex items-center gap-6 text-sm">
            <a
              href="https://vapm-agent-frontend.vercel.app"
              target="_blank"
              rel="noopener noreferrer"
              className="px-5 py-2.5 rounded-lg bg-amber-500/10 border border-amber-500/30 text-amber-400 hover:bg-amber-500/20 transition-colors"
            >
              Frontend
            </a>
            <a
              href="https://github.com/sarthakbiswas97/vapm-agent"
              target="_blank"
              rel="noopener noreferrer"
              className="px-5 py-2.5 rounded-lg bg-violet-500/10 border border-violet-500/30 text-violet-400 hover:bg-violet-500/20 transition-colors"
            >
              GitHub
            </a>
          </div>
        </section>

        {/* Slide 2: The Problem */}
        <section className="py-20 px-6 border-t border-gray-800">
          <div className="max-w-4xl mx-auto">
            <h2 className="text-4xl font-bold mb-4 text-center">
              The Problem
            </h2>
            <p className="text-lg text-gray-300 text-center mb-12">
              AI trading agents have three vulnerabilities
            </p>
            <div className="space-y-8">
              <div className="flex gap-5 items-start">
                <span className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 font-bold">
                  1
                </span>
                <div>
                  <p className="text-lg text-gray-300">
                    Risk parameters are public on-chain &mdash;{" "}
                    <span className="text-red-400">
                      front-runners exploit them
                    </span>
                  </p>
                </div>
              </div>
              <div className="flex gap-5 items-start">
                <span className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 font-bold">
                  2
                </span>
                <div>
                  <p className="text-lg text-gray-300">
                    Guardrails are code, not cryptography &mdash;{" "}
                    <span className="text-red-400">they can be bypassed</span>
                  </p>
                </div>
              </div>
              <div className="flex gap-5 items-start">
                <span className="flex-shrink-0 w-10 h-10 rounded-full bg-red-500/10 border border-red-500/30 flex items-center justify-center text-red-400 font-bold">
                  3
                </span>
                <div>
                  <p className="text-lg text-gray-300">
                    Wallet is a single private key &mdash;{" "}
                    <span className="text-red-400">
                      one compromise, everything gone
                    </span>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Slide 3: The Solution */}
        <section className="py-20 px-6 border-t border-gray-800">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl font-bold mb-4 text-center">
              The Solution
            </h2>
            <p className="text-lg text-gray-300 text-center mb-12">
              Two cryptographic gates, both required
            </p>
            <div className="grid md:grid-cols-2 gap-8 mb-12">
              <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-8">
                <h3 className="text-2xl font-bold text-amber-400 mb-4">
                  Encrypt FHE
                </h3>
                <p className="text-lg text-gray-300">
                  Trade params + risk limits encrypted. Comparison happens in FHE
                  &mdash; only boolean pass/fail revealed.
                </p>
              </div>
              <div className="rounded-xl border border-violet-500/30 bg-violet-500/5 p-8">
                <h3 className="text-2xl font-bold text-violet-400 mb-4">
                  Ika dWallet
                </h3>
                <p className="text-lg text-gray-300">
                  Signing key is 2PC-MPC. No single party holds it. Program must
                  approve via CPI before key signs.
                </p>
              </div>
            </div>
            <p className="text-center text-lg text-gray-500 italic">
              Remove either and the system breaks.
            </p>
          </div>
        </section>

        {/* Slide 4: How It Works */}
        <section className="py-20 px-6 border-t border-gray-800">
          <div className="max-w-5xl mx-auto">
            <h2 className="text-4xl font-bold mb-12 text-center">
              How It Works
            </h2>
            <div className="flex flex-wrap items-center justify-center gap-3 mb-16 text-sm md:text-base">
              {[
                { label: "Market Data", color: "gray" },
                { label: "XGBoost ML", color: "gray" },
                { label: "Encrypt FHE", color: "amber" },
                { label: "On-chain Risk Check", color: "amber" },
                { label: "Ika dWallet", color: "violet" },
                { label: "Jupiter Swap", color: "gray" },
              ].map((step, i, arr) => (
                <div key={step.label} className="flex items-center gap-3">
                  <span
                    className={`px-4 py-2 rounded-lg border ${
                      step.color === "amber"
                        ? "border-amber-500/40 bg-amber-500/10 text-amber-400"
                        : step.color === "violet"
                          ? "border-violet-500/40 bg-violet-500/10 text-violet-400"
                          : "border-gray-700 bg-gray-800/50 text-gray-300"
                    }`}
                  >
                    {step.label}
                  </span>
                  {i < arr.length - 1 && (
                    <span className="text-gray-600">&rarr;</span>
                  )}
                </div>
              ))}
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="py-3 pr-6 text-gray-500 font-medium">
                      Component
                    </th>
                    <th className="py-3 text-gray-500 font-medium">
                      Deployed Address
                    </th>
                  </tr>
                </thead>
                <tbody className="font-mono text-gray-300">
                  <tr className="border-b border-gray-800/50">
                    <td className="py-3 pr-6 text-amber-400">VAPM Program</td>
                    <td className="py-3">6xDo2...</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-3 pr-6 text-violet-400">Ika dWallet</td>
                    <td className="py-3">87W5...</td>
                  </tr>
                  <tr className="border-b border-gray-800/50">
                    <td className="py-3 pr-6 text-amber-400">Encrypt</td>
                    <td className="py-3">4ebf...</td>
                  </tr>
                  <tr>
                    <td className="py-3 pr-6 text-violet-400">dWallet</td>
                    <td className="py-3">7ruu...</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        {/* Slide 5: AI + Explainability */}
        <section className="py-20 px-6 border-t border-gray-800">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-12">
              AI + Explainability
            </h2>
            <div className="space-y-6 text-lg text-gray-300">
              <p>
                <span className="text-amber-400 font-semibold">
                  XGBoost + SHAP
                </span>{" "}
                &mdash; every prediction is auditable
              </p>
              <p>
                <span className="text-violet-400 font-semibold">
                  14 technical indicators
                </span>
                , 129K training samples
              </p>
              <p>Backtested with realistic costs</p>
            </div>
          </div>
        </section>

        {/* Slide 6: Try It */}
        <section className="py-20 px-6 border-t border-gray-800">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-4xl font-bold mb-8">Try It</h2>
            <p className="text-lg text-gray-300 mb-4">
              Everything deployed on Solana Devnet
            </p>
            <a
              href="https://vapm-agent-frontend.vercel.app"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-block text-lg text-amber-400 underline underline-offset-4 hover:text-amber-300 transition-colors mb-12"
            >
              Dashboard: vapm-agent-frontend.vercel.app
            </a>
            <p className="text-xl text-gray-400 italic mt-8">
              &ldquo;That&rsquo;s not software saying no. That&rsquo;s
              cryptography saying no.&rdquo;
            </p>
          </div>
        </section>
      </div>
    </>
  );
}
