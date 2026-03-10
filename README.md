# VAPM - Verifiable AI Portfolio Manager

> An autonomous AI trading agent with on-chain verifiable decisions using ERC-8004 concepts.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

VAPM is a trustless AI trading system that combines machine learning prediction with blockchain-based verification. Every trading decision is logged, hashed, and stored on-chain, enabling anyone to verify the agent's behavior.

### Key Features

- **ML-Powered Trading**: XGBoost model predicting ETH/USDC price movements
- **Explainable Decisions**: SHAP values reveal why each trade was made
- **On-Chain Verification**: Decision hashes stored on-chain for transparency
- **Risk Management**: Hard limits on position size, daily loss, and drawdown
- **EIP-712 Signed Intents**: Cryptographically secure trade authorization

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    VAPM System                           │
├─────────────────────────────────────────────────────────┤
│  Market Data → Feature Engine → ML Model → Strategy     │
│                                              ↓           │
│  Blockchain ← Trade Executor ← Risk Guardian ←          │
│       ↓                                                  │
│  Validation Registry (on-chain decision proofs)         │
└─────────────────────────────────────────────────────────┘
```

## Tech Stack

| Layer | Technology |
|-------|------------|
| ML | XGBoost, SHAP, scikit-learn |
| Backend | Python 3.11, FastAPI |
| Database | PostgreSQL, Redis |
| Blockchain | Solidity 0.8, Hardhat |
| Frontend | Next.js 14, Tailwind |

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- MetaMask wallet with Base Sepolia ETH

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/vapm.git
cd vapm
cp .env.example .env
# Edit .env with your keys
```

### 2. Start Infrastructure

```bash
docker-compose up -d postgres redis
```

### 3. Install Dependencies

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Contracts
cd ../contracts
npm install

# Frontend
cd ../frontend
npm install
```

### 4. Deploy Contracts

```bash
cd contracts
npm run deploy:baseSepolia
```

### 5. Run the Agent

```bash
cd backend
python -m uvicorn main:app --reload
```

### 6. Start Dashboard

```bash
cd frontend
npm run dev
```

## Smart Contracts

### AgentRegistry.sol
ERC-721 based identity registry for AI agents. Each agent gets an NFT representing its on-chain identity and reputation score.

### ValidationRegistry.sol
Stores SHA256 hashes of decision records. Enables verification that the agent made decisions as claimed.

### TradeExecutor.sol
Executes EIP-712 signed trade intents via Uniswap V3. Validates signatures and enforces basic limits.

## Decision Transparency

Every trade generates a Decision Record:

```json
{
  "timestamp": "2026-03-20T14:02:00Z",
  "market_state": {
    "price": 3245.50,
    "rsi": 28.4,
    "volatility": 0.032
  },
  "model_output": {
    "probability_up": 0.72,
    "shap_values": {"rsi": 0.12, "volume": 0.08}
  },
  "strategy_decision": {
    "action": "BUY",
    "reason": "RSI oversold + high confidence"
  },
  "risk_validation": {
    "checks_passed": true,
    "risk_score": 0.31
  }
}
```

The hash of this record is stored on-chain, enabling verification.

## Risk Management

Hard limits that cannot be bypassed:

| Limit | Value |
|-------|-------|
| Max Position Size | 5% of capital |
| Max Daily Loss | 3% of capital |
| Max Drawdown | 10% of capital |
| Min Trade Interval | 60 seconds |

## API Endpoints

```
GET  /health              # System health
GET  /agent/status        # Agent state, PnL
GET  /decisions           # Decision history
GET  /decisions/{id}      # Decision with SHAP
GET  /trades              # Trade history
GET  /risk/snapshot       # Current risk state
POST /agent/start         # Start trading
POST /agent/stop          # Stop trading
```

## Project Structure

```
vapm/
├── contracts/           # Solidity smart contracts
├── backend/            # Python FastAPI backend
│   ├── models/         # Pydantic data models
│   ├── services/       # Business logic
│   └── api/            # REST endpoints
├── ml/                 # ML training & inference
├── frontend/           # Next.js dashboard
└── docs/               # Documentation
```

## Performance Metrics

- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Validation Score**: Decision verification success rate

## Roadmap

- [ ] Core trading pipeline
- [ ] ML model training
- [ ] Risk guardian
- [ ] Smart contract deployment
- [ ] Decision logging
- [ ] Dashboard UI
- [ ] Demo video

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built for the AI Trading Agents with ERC-8004 Hackathon.
