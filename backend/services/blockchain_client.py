"""
Blockchain Client Service - Web3 integration for ERC-8004 operations.

Handles:
- Agent registration and identity (ERC-721 NFT)
- Decision hash logging and verification (ValidationRegistry)
- EIP-712 trade intent signing
- Trade execution via TradeExecutor contract
"""

import json
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from eth_account import Account
from eth_account.messages import encode_typed_data
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware

from config import get_settings


@dataclass
class AgentInfo:
    """On-chain agent information."""
    token_id: int
    name: str
    metadata_uri: str
    reputation_score: int
    registered_at: int
    active: bool


@dataclass
class ValidationRecord:
    """On-chain validation record."""
    decision_hash: str
    model_confidence: int
    risk_score: int
    timestamp: int
    executed: bool


@dataclass
class TxResult:
    """Transaction result."""
    success: bool
    tx_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    error: Optional[str] = None


class BlockchainClient:
    """
    Web3 client for ERC-8004 verifiable AI agent operations.

    Provides methods for:
    - Agent identity management (register, get info)
    - Decision validation (log, verify, mark executed)
    - Trade intent signing and execution
    """

    def __init__(self):
        self.settings = get_settings()
        self._w3: Optional[Web3] = None
        self._account: Optional[Account] = None
        self._contracts: dict = {}
        self._initialized = False

    @property
    def is_enabled(self) -> bool:
        """Check if blockchain features are enabled."""
        return self.settings.blockchain_enabled

    @property
    def address(self) -> str:
        """Get agent wallet address."""
        if not self._account:
            return ""
        return self._account.address

    async def initialize(self) -> bool:
        """
        Initialize Web3 connection and load contracts.

        Returns:
            True if initialization successful, False otherwise.
        """
        if not self.is_enabled:
            print("[Blockchain] Disabled - skipping initialization")
            return False

        if not self.settings.private_key:
            print("[Blockchain] No private key configured")
            return False

        try:
            # Connect to RPC
            self._w3 = Web3(Web3.HTTPProvider(self.settings.rpc_url))

            # Add PoA middleware for Base Sepolia
            self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)

            if not self._w3.is_connected():
                print(f"[Blockchain] Failed to connect to {self.settings.rpc_url}")
                return False

            # Load account from private key
            private_key = self.settings.private_key
            if not private_key.startswith("0x"):
                private_key = f"0x{private_key}"
            self._account = Account.from_key(private_key)

            # Load contract ABIs
            self._load_contracts()

            self._initialized = True
            chain_id = self._w3.eth.chain_id
            balance = self._w3.eth.get_balance(self._account.address)
            balance_eth = self._w3.from_wei(balance, "ether")

            print(f"[Blockchain] Connected to chain {chain_id}")
            print(f"[Blockchain] Agent address: {self._account.address}")
            print(f"[Blockchain] Balance: {balance_eth:.6f} ETH")

            return True

        except Exception as e:
            print(f"[Blockchain] Initialization failed: {e}")
            return False

    def _load_contracts(self):
        """Load contract ABIs and create contract instances."""
        abi_dir = Path(__file__).parent.parent / "contract_abis"

        # AgentRegistry
        if self.settings.agent_registry_address:
            with open(abi_dir / "AgentRegistry.json") as f:
                abi = json.load(f)["abi"]
            self._contracts["agent_registry"] = self._w3.eth.contract(
                address=Web3.to_checksum_address(self.settings.agent_registry_address),
                abi=abi,
            )

        # ValidationRegistry
        if self.settings.validation_registry_address:
            with open(abi_dir / "ValidationRegistry.json") as f:
                abi = json.load(f)["abi"]
            self._contracts["validation_registry"] = self._w3.eth.contract(
                address=Web3.to_checksum_address(self.settings.validation_registry_address),
                abi=abi,
            )

        # TradeExecutor
        if self.settings.trade_executor_address:
            with open(abi_dir / "TradeExecutor.json") as f:
                abi = json.load(f)["abi"]
            self._contracts["trade_executor"] = self._w3.eth.contract(
                address=Web3.to_checksum_address(self.settings.trade_executor_address),
                abi=abi,
            )

    def _send_transaction(self, tx_func, *args) -> TxResult:
        """
        Build and send a transaction.

        Args:
            tx_func: Contract function to call
            *args: Function arguments

        Returns:
            TxResult with transaction details
        """
        try:
            # Build transaction
            tx = tx_func(*args).build_transaction({
                "from": self._account.address,
                "nonce": self._w3.eth.get_transaction_count(self._account.address),
                "gas": 500000,  # Estimate in production
                "gasPrice": self._w3.eth.gas_price,
                "chainId": self.settings.chain_id,
            })

            # Sign and send
            signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)

            # Wait for receipt
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)

            return TxResult(
                success=receipt["status"] == 1,
                tx_hash=receipt["transactionHash"].hex(),
                block_number=receipt["blockNumber"],
                gas_used=receipt["gasUsed"],
            )

        except Exception as e:
            return TxResult(
                success=False,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error=str(e),
            )

    # ─────────────────────────────────────────────────────────────
    # AGENT IDENTITY (ERC-8004 / ERC-721)
    # ─────────────────────────────────────────────────────────────

    async def register_agent(self, name: str, metadata_uri: str) -> TxResult:
        """
        Register this agent on-chain (mint NFT identity).

        Args:
            name: Human-readable agent name
            metadata_uri: IPFS or HTTP URI for agent metadata

        Returns:
            TxResult with registration transaction details
        """
        if not self._initialized:
            return TxResult(False, None, None, None, "Not initialized")

        contract = self._contracts.get("agent_registry")
        if not contract:
            return TxResult(False, None, None, None, "AgentRegistry not configured")

        # Check if already registered
        has_agent = contract.functions.hasAgent(self._account.address).call()
        if has_agent:
            token_id = contract.functions.agentOf(self._account.address).call()
            return TxResult(
                success=True,
                tx_hash=None,
                block_number=None,
                gas_used=None,
                error=f"Already registered as token {token_id}",
            )

        return self._send_transaction(
            contract.functions.registerAgent,
            name,
            metadata_uri,
        )

    async def get_agent_info(self) -> Optional[AgentInfo]:
        """
        Get this agent's on-chain information.

        Returns:
            AgentInfo or None if not registered
        """
        if not self._initialized:
            return None

        contract = self._contracts.get("agent_registry")
        if not contract:
            return None

        # Check if registered
        has_agent = contract.functions.hasAgent(self._account.address).call()
        if not has_agent:
            return None

        token_id = contract.functions.agentOf(self._account.address).call()
        info = contract.functions.getAgent(token_id).call()

        return AgentInfo(
            token_id=token_id,
            name=info[0],
            metadata_uri=info[1],
            reputation_score=info[2],
            registered_at=info[3],
            active=info[4],
        )

    # ─────────────────────────────────────────────────────────────
    # DECISION VALIDATION (ValidationRegistry)
    # ─────────────────────────────────────────────────────────────

    async def log_decision(
        self,
        decision_id: str,
        decision_hash: str,
        confidence: int,
        risk_score: int,
    ) -> TxResult:
        """
        Log a decision hash on-chain for verification.

        Args:
            decision_id: Unique decision identifier
            decision_hash: SHA256 hash of decision JSON (0x prefixed)
            confidence: Model confidence scaled 0-1000
            risk_score: Risk score scaled 0-1000

        Returns:
            TxResult with transaction details
        """
        if not self._initialized:
            return TxResult(False, None, None, None, "Not initialized")

        contract = self._contracts.get("validation_registry")
        if not contract:
            return TxResult(False, None, None, None, "ValidationRegistry not configured")

        # Convert hash string to bytes32
        hash_bytes = bytes.fromhex(decision_hash[2:]) if decision_hash.startswith("0x") else bytes.fromhex(decision_hash)

        return self._send_transaction(
            contract.functions.logDecision,
            decision_id,
            hash_bytes,
            confidence,
            risk_score,
        )

    async def verify_decision(self, decision_id: str, expected_hash: str) -> bool:
        """
        Verify a decision hash matches on-chain record.

        Args:
            decision_id: Decision identifier
            expected_hash: Expected SHA256 hash

        Returns:
            True if hash matches, False otherwise
        """
        if not self._initialized:
            return False

        contract = self._contracts.get("validation_registry")
        if not contract:
            return False

        hash_bytes = bytes.fromhex(expected_hash[2:]) if expected_hash.startswith("0x") else bytes.fromhex(expected_hash)

        return contract.functions.verifyDecision(
            self._account.address,
            decision_id,
            hash_bytes,
        ).call()

    async def get_validation_record(self, decision_id: str) -> Optional[ValidationRecord]:
        """
        Get validation record for a decision.

        Args:
            decision_id: Decision identifier

        Returns:
            ValidationRecord or None if not found
        """
        if not self._initialized:
            return None

        contract = self._contracts.get("validation_registry")
        if not contract:
            return None

        record = contract.functions.getRecord(self._account.address, decision_id).call()

        if record[3] == 0:  # timestamp == 0 means not found
            return None

        return ValidationRecord(
            decision_hash="0x" + record[0].hex(),
            model_confidence=record[1],
            risk_score=record[2],
            timestamp=record[3],
            executed=record[4],
        )

    async def mark_executed(self, decision_id: str) -> TxResult:
        """
        Mark a decision as executed on-chain.

        Args:
            decision_id: Decision identifier

        Returns:
            TxResult with transaction details
        """
        if not self._initialized:
            return TxResult(False, None, None, None, "Not initialized")

        contract = self._contracts.get("validation_registry")
        if not contract:
            return TxResult(False, None, None, None, "ValidationRegistry not configured")

        return self._send_transaction(
            contract.functions.markExecuted,
            decision_id,
        )

    async def get_decision_count(self) -> int:
        """Get total number of decisions logged for this agent."""
        if not self._initialized:
            return 0

        contract = self._contracts.get("validation_registry")
        if not contract:
            return 0

        return contract.functions.getDecisionCount(self._account.address).call()

    # ─────────────────────────────────────────────────────────────
    # TRADE EXECUTION (TradeExecutor with EIP-712)
    # ─────────────────────────────────────────────────────────────

    async def get_nonce(self) -> int:
        """
        Get current nonce for trade intents.

        Returns:
            Current nonce value
        """
        if not self._initialized:
            return 0

        contract = self._contracts.get("trade_executor")
        if not contract:
            return 0

        return contract.functions.getNonce(self._account.address).call()

    def sign_trade_intent(
        self,
        asset: str,
        action: str,
        amount: int,
        max_slippage_bps: int,
        deadline: int,
        decision_hash: str,
        nonce: int,
    ) -> str:
        """
        Sign a trade intent using EIP-712.

        Args:
            asset: Asset to trade (e.g., "ETH")
            action: "BUY" or "SELL"
            amount: Amount in wei/smallest unit
            max_slippage_bps: Max slippage in basis points
            deadline: Unix timestamp deadline
            decision_hash: Hash of the decision record
            nonce: Current nonce from contract

        Returns:
            Hex-encoded signature
        """
        if not self._initialized or not self._account:
            return ""

        trade_executor_address = self.settings.trade_executor_address
        if not trade_executor_address:
            return ""

        # EIP-712 typed data
        typed_data = {
            "types": {
                "EIP712Domain": [
                    {"name": "name", "type": "string"},
                    {"name": "version", "type": "string"},
                    {"name": "chainId", "type": "uint256"},
                    {"name": "verifyingContract", "type": "address"},
                ],
                "TradeIntent": [
                    {"name": "agent", "type": "address"},
                    {"name": "asset", "type": "string"},
                    {"name": "action", "type": "string"},
                    {"name": "amount", "type": "uint256"},
                    {"name": "maxSlippageBps", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "decisionHash", "type": "bytes32"},
                    {"name": "nonce", "type": "uint256"},
                ],
            },
            "primaryType": "TradeIntent",
            "domain": {
                "name": "VAPM Trade Executor",
                "version": "1",
                "chainId": self.settings.chain_id,
                "verifyingContract": Web3.to_checksum_address(trade_executor_address),
            },
            "message": {
                "agent": self._account.address,
                "asset": asset,
                "action": action,
                "amount": amount,
                "maxSlippageBps": max_slippage_bps,
                "deadline": deadline,
                "decisionHash": bytes.fromhex(decision_hash[2:]) if decision_hash.startswith("0x") else bytes.fromhex(decision_hash),
                "nonce": nonce,
            },
        }

        # Sign using eth_account
        signed = self._account.sign_typed_data(
            typed_data["domain"],
            typed_data["types"],
            typed_data["message"],
        )

        return signed.signature.hex()

    async def submit_trade_intent(
        self,
        asset: str,
        action: str,
        amount: int,
        max_slippage_bps: int,
        deadline: int,
        decision_hash: str,
        signature: str,
    ) -> TxResult:
        """
        Submit a signed trade intent to the TradeExecutor contract.

        Args:
            asset: Asset to trade
            action: "BUY" or "SELL"
            amount: Amount in wei
            max_slippage_bps: Max slippage in basis points
            deadline: Unix timestamp deadline
            decision_hash: Hash of the decision record
            signature: EIP-712 signature

        Returns:
            TxResult with transaction details
        """
        if not self._initialized:
            return TxResult(False, None, None, None, "Not initialized")

        contract = self._contracts.get("trade_executor")
        if not contract:
            return TxResult(False, None, None, None, "TradeExecutor not configured")

        # Get current nonce
        nonce = await self.get_nonce()

        # Build intent tuple
        hash_bytes = bytes.fromhex(decision_hash[2:]) if decision_hash.startswith("0x") else bytes.fromhex(decision_hash)
        intent = (
            self._account.address,
            asset,
            action,
            amount,
            max_slippage_bps,
            deadline,
            hash_bytes,
            nonce,
        )

        # Convert signature to bytes
        sig_bytes = bytes.fromhex(signature[2:]) if signature.startswith("0x") else bytes.fromhex(signature)

        return self._send_transaction(
            contract.functions.executeTrade,
            intent,
            sig_bytes,
        )

    # ─────────────────────────────────────────────────────────────
    # UTILITY METHODS
    # ─────────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Get blockchain client status."""
        if not self._initialized:
            return {
                "enabled": self.is_enabled,
                "initialized": False,
                "address": None,
                "balance": None,
                "chain_id": None,
            }

        balance = self._w3.eth.get_balance(self._account.address)
        balance_eth = float(self._w3.from_wei(balance, "ether"))

        return {
            "enabled": True,
            "initialized": True,
            "address": self._account.address,
            "balance": balance_eth,
            "chain_id": self._w3.eth.chain_id,
            "contracts": {
                "agent_registry": self.settings.agent_registry_address or None,
                "validation_registry": self.settings.validation_registry_address or None,
                "trade_executor": self.settings.trade_executor_address or None,
            },
        }


# Global singleton
blockchain_client = BlockchainClient()
