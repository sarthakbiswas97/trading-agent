// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title AgentRegistry
 * @notice ERC-8004 inspired agent identity registry
 * @dev Each AI agent is represented as an NFT with metadata
 */
contract AgentRegistry is ERC721, Ownable {
    uint256 private _tokenIdCounter;

    struct AgentInfo {
        string name;
        string metadataUri;
        uint256 reputationScore;
        uint256 registeredAt;
        bool active;
    }

    // tokenId => AgentInfo
    mapping(uint256 => AgentInfo) public agents;

    // owner address => tokenId (one agent per address for simplicity)
    mapping(address => uint256) public agentOf;

    event AgentRegistered(uint256 indexed tokenId, address indexed owner, string name);
    event AgentUpdated(uint256 indexed tokenId, string metadataUri);
    event ReputationUpdated(uint256 indexed tokenId, uint256 newScore);

    constructor() ERC721("VAPM Agent", "VAPM") {}

    /**
     * @notice Register a new AI agent
     * @param name Human-readable agent name
     * @param metadataUri IPFS or HTTP URI for agent metadata
     */
    function registerAgent(string memory name, string memory metadataUri) external returns (uint256) {
        require(agentOf[msg.sender] == 0, "Already registered");
        require(bytes(name).length > 0, "Name required");

        _tokenIdCounter++;
        uint256 tokenId = _tokenIdCounter;

        _safeMint(msg.sender, tokenId);

        agents[tokenId] = AgentInfo({
            name: name,
            metadataUri: metadataUri,
            reputationScore: 50, // Start at neutral reputation
            registeredAt: block.timestamp,
            active: true
        });

        agentOf[msg.sender] = tokenId;

        emit AgentRegistered(tokenId, msg.sender, name);
        return tokenId;
    }

    /**
     * @notice Update agent metadata
     * @param tokenId Agent token ID
     * @param metadataUri New metadata URI
     */
    function updateMetadata(uint256 tokenId, string memory metadataUri) external {
        require(ownerOf(tokenId) == msg.sender, "Not agent owner");
        agents[tokenId].metadataUri = metadataUri;
        emit AgentUpdated(tokenId, metadataUri);
    }

    /**
     * @notice Update agent reputation (only by authorized validators)
     * @param tokenId Agent token ID
     * @param newScore New reputation score (0-100)
     */
    function updateReputation(uint256 tokenId, uint256 newScore) external onlyOwner {
        require(newScore <= 100, "Score must be <= 100");
        agents[tokenId].reputationScore = newScore;
        emit ReputationUpdated(tokenId, newScore);
    }

    /**
     * @notice Get agent info
     * @param tokenId Agent token ID
     */
    function getAgent(uint256 tokenId) external view returns (AgentInfo memory) {
        require(tokenId > 0 && tokenId <= _tokenIdCounter, "Invalid tokenId");
        return agents[tokenId];
    }

    /**
     * @notice Check if address has registered agent
     * @param addr Address to check
     */
    function hasAgent(address addr) external view returns (bool) {
        return agentOf[addr] != 0;
    }
}
