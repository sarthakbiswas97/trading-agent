// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title ValidationRegistry
 * @notice Stores decision hashes for verifiable AI trading
 * @dev Provides on-chain proof that decisions were made transparently
 */
contract ValidationRegistry {
    struct ValidationRecord {
        bytes32 decisionHash;      // SHA256 hash of full decision JSON
        uint256 modelConfidence;   // Scaled 0-1000 (0.001 precision)
        uint256 riskScore;         // Scaled 0-1000
        uint256 timestamp;
        bool executed;
    }

    // agent address => decision ID => ValidationRecord
    mapping(address => mapping(string => ValidationRecord)) public records;

    // agent address => array of decision IDs
    mapping(address => string[]) public agentDecisions;

    // Total records count
    uint256 public totalRecords;

    event DecisionLogged(
        address indexed agent,
        string indexed decisionId,
        bytes32 decisionHash,
        uint256 timestamp
    );

    event DecisionExecuted(
        address indexed agent,
        string indexed decisionId,
        bytes32 txHash
    );

    /**
     * @notice Log a trading decision for verification
     * @param decisionId Unique decision identifier
     * @param decisionHash SHA256 hash of the decision record JSON
     * @param modelConfidence ML model confidence (0-1000)
     * @param riskScore Risk assessment score (0-1000)
     */
    function logDecision(
        string calldata decisionId,
        bytes32 decisionHash,
        uint256 modelConfidence,
        uint256 riskScore
    ) external {
        require(bytes(decisionId).length > 0, "Decision ID required");
        require(records[msg.sender][decisionId].timestamp == 0, "Already logged");
        require(modelConfidence <= 1000, "Confidence out of range");
        require(riskScore <= 1000, "Risk score out of range");

        records[msg.sender][decisionId] = ValidationRecord({
            decisionHash: decisionHash,
            modelConfidence: modelConfidence,
            riskScore: riskScore,
            timestamp: block.timestamp,
            executed: false
        });

        agentDecisions[msg.sender].push(decisionId);
        totalRecords++;

        emit DecisionLogged(msg.sender, decisionId, decisionHash, block.timestamp);
    }

    /**
     * @notice Mark a decision as executed (called after trade)
     * @param decisionId Decision identifier
     */
    function markExecuted(string calldata decisionId) external {
        require(records[msg.sender][decisionId].timestamp > 0, "Decision not found");
        require(!records[msg.sender][decisionId].executed, "Already executed");

        records[msg.sender][decisionId].executed = true;

        emit DecisionExecuted(msg.sender, decisionId, bytes32(0));
    }

    /**
     * @notice Verify a decision hash matches
     * @param agent Agent address
     * @param decisionId Decision identifier
     * @param expectedHash Expected decision hash
     */
    function verifyDecision(
        address agent,
        string calldata decisionId,
        bytes32 expectedHash
    ) external view returns (bool) {
        ValidationRecord memory record = records[agent][decisionId];
        return record.decisionHash == expectedHash && record.timestamp > 0;
    }

    /**
     * @notice Get validation record
     * @param agent Agent address
     * @param decisionId Decision identifier
     */
    function getRecord(
        address agent,
        string calldata decisionId
    ) external view returns (ValidationRecord memory) {
        return records[agent][decisionId];
    }

    /**
     * @notice Get all decision IDs for an agent
     * @param agent Agent address
     */
    function getAgentDecisions(address agent) external view returns (string[] memory) {
        return agentDecisions[agent];
    }

    /**
     * @notice Get decision count for an agent
     * @param agent Agent address
     */
    function getDecisionCount(address agent) external view returns (uint256) {
        return agentDecisions[agent].length;
    }
}
