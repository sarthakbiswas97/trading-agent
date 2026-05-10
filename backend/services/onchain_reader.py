"""
On-Chain Reader - Reads REAL account data from Solana devnet.

Parses AgentState and TradeProposal PDAs using exact byte offsets
from the deployed Anchor program.
"""

from __future__ import annotations

import base64
import logging
import struct
import time
from dataclasses import dataclass

import aiohttp
from solders.pubkey import Pubkey

from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

EXPLORER_BASE = "https://explorer.solana.com/address"


def explorer_url(pubkey: str) -> str:
    return f"{EXPLORER_BASE}/{pubkey}?cluster=devnet"


@dataclass
class AgentOnChain:
    pda: str
    authority: str
    name: str
    decision_count: int
    dwallet: str
    enc_max_position: str
    enc_max_daily_loss: str
    enc_max_drawdown: str
    trades_approved: int
    trades_rejected: int


@dataclass
class TradeOnChain:
    pda: str
    agent: str
    proposer: str
    index: int
    enc_position: str
    enc_pnl: str
    enc_drawdown: str
    fhe_pos_ok: str
    fhe_pnl_ok: str
    fhe_dd_ok: str
    verdict: int  # 0=Pending, 1=Approved, 2=Rejected
    message_hash: str
    timestamp: int


class OnChainReader:
    """Reads real account data from Solana devnet."""

    def __init__(self) -> None:
        self._cache: dict[str, tuple[float, dict]] = {}
        self._cache_ttl = 10.0  # seconds

    async def _rpc_get_account(self, pubkey: str) -> bytes | None:
        """Fetch raw account data from devnet RPC."""
        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getAccountInfo",
            "params": [pubkey, {"encoding": "base64", "commitment": "confirmed"}],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                settings.solana_rpc_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                result = await resp.json()
                value = result.get("result", {}).get("value")
                if not value:
                    return None
                return base64.b64decode(value["data"][0])

    def _parse_pubkey(self, data: bytes, offset: int) -> str:
        """Parse a 32-byte pubkey from raw data."""
        return str(Pubkey.from_bytes(data[offset : offset + 32]))

    def _parse_agent_state(self, data: bytes, pda: str) -> AgentOnChain | None:
        """Parse AgentState from raw Anchor account data."""
        if len(data) < 200:
            return None

        # Anchor discriminator: 8 bytes
        offset = 8

        authority = self._parse_pubkey(data, offset)
        offset += 32

        # String: 4 bytes len + content
        name_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        name = data[offset : offset + min(name_len, 32)].decode("utf-8", errors="replace")
        offset += 32  # fixed 32 bytes allocated for name

        decision_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8

        dwallet = self._parse_pubkey(data, offset)
        offset += 32

        enc_max_position = self._parse_pubkey(data, offset)
        offset += 32
        enc_max_daily_loss = self._parse_pubkey(data, offset)
        offset += 32
        enc_max_drawdown = self._parse_pubkey(data, offset)
        offset += 32

        trades_approved = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        trades_rejected = struct.unpack_from("<I", data, offset)[0]

        return AgentOnChain(
            pda=pda,
            authority=authority,
            name=name,
            decision_count=decision_count,
            dwallet=dwallet,
            enc_max_position=enc_max_position,
            enc_max_daily_loss=enc_max_daily_loss,
            enc_max_drawdown=enc_max_drawdown,
            trades_approved=trades_approved,
            trades_rejected=trades_rejected,
        )

    def _parse_trade_proposal(self, data: bytes, pda: str) -> TradeOnChain | None:
        """Parse TradeProposal from raw Anchor account data."""
        if len(data) < 300:
            return None

        offset = 8  # discriminator
        agent = self._parse_pubkey(data, offset); offset += 32
        proposer = self._parse_pubkey(data, offset); offset += 32
        index = struct.unpack_from("<Q", data, offset)[0]; offset += 8

        enc_position = self._parse_pubkey(data, offset); offset += 32
        enc_pnl = self._parse_pubkey(data, offset); offset += 32
        enc_drawdown = self._parse_pubkey(data, offset); offset += 32

        fhe_pos_ok = self._parse_pubkey(data, offset); offset += 32
        fhe_pnl_ok = self._parse_pubkey(data, offset); offset += 32
        fhe_dd_ok = self._parse_pubkey(data, offset); offset += 32

        verdict = data[offset]; offset += 1
        message_hash = data[offset : offset + 32].hex(); offset += 32
        timestamp = struct.unpack_from("<q", data, offset)[0]

        return TradeOnChain(
            pda=pda, agent=agent, proposer=proposer, index=index,
            enc_position=enc_position, enc_pnl=enc_pnl, enc_drawdown=enc_drawdown,
            fhe_pos_ok=fhe_pos_ok, fhe_pnl_ok=fhe_pnl_ok, fhe_dd_ok=fhe_dd_ok,
            verdict=verdict, message_hash=message_hash, timestamp=timestamp,
        )

    async def get_live_data(self, authority_pubkey: str) -> dict:
        """Get all on-chain data for a given agent authority."""
        cache_key = f"live:{authority_pubkey}"
        now = time.time()
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return data

        program_id = settings.decision_program_id
        result: dict = {
            "program_id": program_id,
            "program_explorer": explorer_url(program_id) if program_id else None,
            "agent": None,
            "trades": [],
            "ika_program": "87W54kGYFQ1rgWqMeu4XTPHWXWmXSQCcjm8vCTfiq1oY",
            "encrypt_program": "4ebfzWdKnrnGseuQpezXdG8yCdHqwQ1SSBHD3bWArND8",
        }

        if not program_id or not authority_pubkey:
            return result

        # Derive and fetch AgentState PDA
        try:
            program_pubkey = Pubkey.from_string(program_id)
            authority = Pubkey.from_string(authority_pubkey)
            agent_pda, _ = Pubkey.find_program_address(
                [b"agent", bytes(authority)], program_pubkey
            )
            agent_pda_str = str(agent_pda)

            data = await self._rpc_get_account(agent_pda_str)
            if data:
                agent = self._parse_agent_state(data, agent_pda_str)
                if agent:
                    result["agent"] = {
                        "pda": agent.pda,
                        "explorer": explorer_url(agent.pda),
                        "authority": agent.authority,
                        "name": agent.name,
                        "decision_count": agent.decision_count,
                        "dwallet": agent.dwallet,
                        "dwallet_explorer": explorer_url(agent.dwallet),
                        "enc_max_position": agent.enc_max_position,
                        "enc_max_daily_loss": agent.enc_max_daily_loss,
                        "enc_max_drawdown": agent.enc_max_drawdown,
                        "trades_approved": agent.trades_approved,
                        "trades_rejected": agent.trades_rejected,
                    }

                    # Fetch trade proposals (up to last 5)
                    for i in range(min(agent.decision_count, 5)):
                        trade_pda, _ = Pubkey.find_program_address(
                            [b"t", bytes(agent_pda), struct.pack("<Q", i)],
                            program_pubkey,
                        )
                        trade_data = await self._rpc_get_account(str(trade_pda))
                        if trade_data:
                            trade = self._parse_trade_proposal(trade_data, str(trade_pda))
                            if trade:
                                verdict_label = {0: "Pending", 1: "Approved", 2: "Rejected"}.get(trade.verdict, "Unknown")
                                result["trades"].append({
                                    "pda": trade.pda,
                                    "explorer": explorer_url(trade.pda),
                                    "index": trade.index,
                                    "verdict": verdict_label,
                                    "verdict_code": trade.verdict,
                                    "message_hash": trade.message_hash,
                                    "enc_position": trade.enc_position,
                                    "enc_pnl": trade.enc_pnl,
                                    "enc_drawdown": trade.enc_drawdown,
                                    "fhe_pos_ok": trade.fhe_pos_ok,
                                    "fhe_pnl_ok": trade.fhe_pnl_ok,
                                    "fhe_dd_ok": trade.fhe_dd_ok,
                                    "timestamp": trade.timestamp,
                                })
        except Exception as e:
            logger.error("[OnChainReader] Error: %s", e)

        self._cache[cache_key] = (now, result)
        return result


# Global singleton
onchain_reader = OnChainReader()
