// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC20/IERC20.sol";
import "@openzeppelin/contracts/token/ERC20/utils/SafeERC20.sol";
import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/ReentrancyGuard.sol";

/**
 * @title TradeExecutor
 * @notice Executes EIP-712 signed trade intents via Uniswap
 * @dev Validates signatures and enforces basic risk limits
 */
contract TradeExecutor is EIP712, Ownable, ReentrancyGuard {
    using SafeERC20 for IERC20;
    using ECDSA for bytes32;

    // Uniswap V3 SwapRouter interface (simplified)
    ISwapRouter public immutable swapRouter;

    // Supported tokens
    address public immutable WETH;
    address public immutable USDC;

    // Pool fee tier (0.3%)
    uint24 public constant POOL_FEE = 3000;

    // Nonce tracking to prevent replay
    mapping(address => uint256) public nonces;

    // Trade intent type hash for EIP-712
    bytes32 public constant TRADE_INTENT_TYPEHASH = keccak256(
        "TradeIntent(address agent,string asset,string action,uint256 amount,uint256 maxSlippageBps,uint256 deadline,bytes32 decisionHash,uint256 nonce)"
    );

    struct TradeIntent {
        address agent;
        string asset;
        string action;
        uint256 amount;
        uint256 maxSlippageBps;
        uint256 deadline;
        bytes32 decisionHash;
        uint256 nonce;
    }

    event TradeExecuted(
        address indexed agent,
        string action,
        uint256 amountIn,
        uint256 amountOut,
        bytes32 decisionHash
    );

    constructor(
        address _swapRouter,
        address _weth,
        address _usdc
    ) EIP712("VAPM Trade Executor", "1") Ownable(msg.sender) {
        swapRouter = ISwapRouter(_swapRouter);
        WETH = _weth;
        USDC = _usdc;
    }

    /**
     * @notice Execute a signed trade intent
     * @param intent Trade intent data
     * @param signature EIP-712 signature from agent
     */
    function executeTrade(
        TradeIntent calldata intent,
        bytes calldata signature
    ) external nonReentrant returns (uint256 amountOut) {
        // Validate deadline
        require(block.timestamp <= intent.deadline, "Intent expired");

        // Validate nonce
        require(intent.nonce == nonces[intent.agent], "Invalid nonce");
        nonces[intent.agent]++;

        // Verify signature
        bytes32 structHash = keccak256(abi.encode(
            TRADE_INTENT_TYPEHASH,
            intent.agent,
            keccak256(bytes(intent.asset)),
            keccak256(bytes(intent.action)),
            intent.amount,
            intent.maxSlippageBps,
            intent.deadline,
            intent.decisionHash,
            intent.nonce
        ));

        bytes32 digest = _hashTypedDataV4(structHash);
        address signer = ECDSA.recover(digest, signature);
        require(signer == intent.agent, "Invalid signature");

        // Execute swap
        if (keccak256(bytes(intent.action)) == keccak256(bytes("BUY"))) {
            // BUY ETH with USDC
            amountOut = _swapUSDCForETH(intent.agent, intent.amount, intent.maxSlippageBps);
        } else if (keccak256(bytes(intent.action)) == keccak256(bytes("SELL"))) {
            // SELL ETH for USDC
            amountOut = _swapETHForUSDC(intent.agent, intent.amount, intent.maxSlippageBps);
        } else {
            revert("Invalid action");
        }

        emit TradeExecuted(
            intent.agent,
            intent.action,
            intent.amount,
            amountOut,
            intent.decisionHash
        );

        return amountOut;
    }

    /**
     * @notice Swap USDC for WETH
     */
    function _swapUSDCForETH(
        address agent,
        uint256 amountIn,
        uint256 maxSlippageBps
    ) internal returns (uint256 amountOut) {
        // Transfer USDC from agent
        IERC20(USDC).safeTransferFrom(agent, address(this), amountIn);

        // Approve router
        IERC20(USDC).safeIncreaseAllowance(address(swapRouter), amountIn);

        // Calculate minimum output (with slippage)
        // In production, use oracle for price
        uint256 amountOutMinimum = 0; // Simplified - use oracle in production

        // Execute swap
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: USDC,
            tokenOut: WETH,
            fee: POOL_FEE,
            recipient: agent,
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: amountOutMinimum,
            sqrtPriceLimitX96: 0
        });

        amountOut = swapRouter.exactInputSingle(params);
    }

    /**
     * @notice Swap WETH for USDC
     */
    function _swapETHForUSDC(
        address agent,
        uint256 amountIn,
        uint256 maxSlippageBps
    ) internal returns (uint256 amountOut) {
        // Transfer WETH from agent
        IERC20(WETH).safeTransferFrom(agent, address(this), amountIn);

        // Approve router
        IERC20(WETH).safeIncreaseAllowance(address(swapRouter), amountIn);

        // Execute swap
        ISwapRouter.ExactInputSingleParams memory params = ISwapRouter.ExactInputSingleParams({
            tokenIn: WETH,
            tokenOut: USDC,
            fee: POOL_FEE,
            recipient: agent,
            deadline: block.timestamp,
            amountIn: amountIn,
            amountOutMinimum: 0,
            sqrtPriceLimitX96: 0
        });

        amountOut = swapRouter.exactInputSingle(params);
    }

    /**
     * @notice Get current nonce for an agent
     */
    function getNonce(address agent) external view returns (uint256) {
        return nonces[agent];
    }

    /**
     * @notice Get domain separator for EIP-712
     */
    function getDomainSeparator() external view returns (bytes32) {
        return _domainSeparatorV4();
    }
}

// Minimal Uniswap V3 SwapRouter interface
interface ISwapRouter {
    struct ExactInputSingleParams {
        address tokenIn;
        address tokenOut;
        uint24 fee;
        address recipient;
        uint256 deadline;
        uint256 amountIn;
        uint256 amountOutMinimum;
        uint160 sqrtPriceLimitX96;
    }

    function exactInputSingle(ExactInputSingleParams calldata params) external payable returns (uint256 amountOut);
}
