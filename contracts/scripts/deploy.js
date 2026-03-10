const hre = require("hardhat");

async function main() {
  const [deployer] = await hre.ethers.getSigners();
  console.log("Deploying contracts with account:", deployer.address);
  console.log("Account balance:", (await hre.ethers.provider.getBalance(deployer.address)).toString());

  // Base Sepolia addresses (update these for your target network)
  const UNISWAP_ROUTER = "0x94cC0AaC535CCDB3C01d6787D6413C739ae12bc4"; // Base Sepolia SwapRouter
  const WETH = "0x4200000000000000000000000000000000000006"; // Base WETH
  const USDC = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"; // Base Sepolia USDC

  // Deploy AgentRegistry
  console.log("\nDeploying AgentRegistry...");
  const AgentRegistry = await hre.ethers.getContractFactory("AgentRegistry");
  const agentRegistry = await AgentRegistry.deploy();
  await agentRegistry.waitForDeployment();
  const agentRegistryAddress = await agentRegistry.getAddress();
  console.log("AgentRegistry deployed to:", agentRegistryAddress);

  // Deploy ValidationRegistry
  console.log("\nDeploying ValidationRegistry...");
  const ValidationRegistry = await hre.ethers.getContractFactory("ValidationRegistry");
  const validationRegistry = await ValidationRegistry.deploy();
  await validationRegistry.waitForDeployment();
  const validationRegistryAddress = await validationRegistry.getAddress();
  console.log("ValidationRegistry deployed to:", validationRegistryAddress);

  // Deploy TradeExecutor
  console.log("\nDeploying TradeExecutor...");
  const TradeExecutor = await hre.ethers.getContractFactory("TradeExecutor");
  const tradeExecutor = await TradeExecutor.deploy(UNISWAP_ROUTER, WETH, USDC);
  await tradeExecutor.waitForDeployment();
  const tradeExecutorAddress = await tradeExecutor.getAddress();
  console.log("TradeExecutor deployed to:", tradeExecutorAddress);

  // Summary
  console.log("\n========================================");
  console.log("Deployment Summary");
  console.log("========================================");
  console.log("Network:", hre.network.name);
  console.log("AgentRegistry:", agentRegistryAddress);
  console.log("ValidationRegistry:", validationRegistryAddress);
  console.log("TradeExecutor:", tradeExecutorAddress);
  console.log("========================================");

  // Save deployment addresses
  const fs = require("fs");
  const deployments = {
    network: hre.network.name,
    chainId: hre.network.config.chainId,
    timestamp: new Date().toISOString(),
    contracts: {
      AgentRegistry: agentRegistryAddress,
      ValidationRegistry: validationRegistryAddress,
      TradeExecutor: tradeExecutorAddress,
    },
    externalContracts: {
      SwapRouter: UNISWAP_ROUTER,
      WETH: WETH,
      USDC: USDC,
    },
  };

  fs.writeFileSync(
    `./deployments-${hre.network.name}.json`,
    JSON.stringify(deployments, null, 2)
  );
  console.log(`\nDeployment addresses saved to deployments-${hre.network.name}.json`);
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  });
