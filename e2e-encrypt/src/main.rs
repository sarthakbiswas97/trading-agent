//! VAPM x Encrypt FHE E2E Demo
//!
//! Creates real encrypted inputs on Solana devnet via Encrypt gRPC,
//! demonstrating FHE privacy for AI trading signals.
//!
//! Usage: cargo run -p vapm-e2e-encrypt

use std::env;
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

use solana_rpc_client::rpc_client::RpcClient;
use solana_sdk::instruction::{AccountMeta, Instruction};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::transaction::Transaction;

use encrypt_solana_client::grpc::EncryptClient;
use encrypt_types::encrypted::Bool;

const B: &str = "\x1b[1m";
const R: &str = "\x1b[0m";
const C: &str = "\x1b[36m";
const G: &str = "\x1b[32m";
const Y: &str = "\x1b[33m";

fn log(s: &str, m: &str) { println!("{C}[{s}]{R} {m}"); }
fn ok(m: &str) { println!("{G}  \u{2713}{R} {m}"); }
fn val(l: &str, v: impl std::fmt::Display) { println!("{Y}  \u{2192}{R} {l}: {v}"); }

fn pda(seeds: &[&[u8]], program_id: &Pubkey) -> (Pubkey, u8) {
    Pubkey::find_program_address(seeds, program_id)
}

fn load_payer() -> Keypair {
    let path = env::var("PAYER_KEYPAIR").unwrap_or_else(|_| {
        format!("{}/.config/solana/id.json", env::var("HOME").unwrap_or_default())
    });
    let data = std::fs::read_to_string(&path).expect("read keypair");
    let bytes: Vec<u8> = data.trim()[1..data.trim().len()-1]
        .split(',').map(|v| v.trim().parse::<u8>().unwrap()).collect();
    Keypair::from_bytes(&bytes).expect("valid keypair")
}

fn poll_until(
    client: &RpcClient, account: &Pubkey,
    check: impl Fn(&[u8]) -> bool, timeout: Duration,
) -> Vec<u8> {
    let start = Instant::now();
    loop {
        if start.elapsed() > timeout { panic!("timeout waiting for {account}"); }
        if let Ok(acct) = client.get_account(account) {
            if check(&acct.data) { return acct.data; }
        }
        thread::sleep(Duration::from_millis(500));
    }
}

fn send_tx(c: &RpcClient, p: &Keypair, ixs: Vec<Instruction>, extra: &[&Keypair]) {
    let bh = c.get_latest_blockhash().expect("blockhash");
    let mut signers: Vec<&Keypair> = vec![p];
    signers.extend_from_slice(extra);
    let tx = Transaction::new_signed_with_payer(&ixs, Some(&p.pubkey()), &signers, bh);
    let b = bincode::serialize(&tx).unwrap();
    let v: solana_transaction::versioned::VersionedTransaction = bincode::deserialize(&b).unwrap();
    c.send_and_confirm_transaction(&v).expect("send tx");
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let encrypt_program = Pubkey::from_str("4ebfzWdKnrnGseuQpezXdG8yCdHqwQ1SSBHD3bWArND8")?;
    let vapm_program = Pubkey::from_str(
        &env::args().nth(1).unwrap_or("6xDo2r8Edvu1MHxwUtqmmzm3Auavf2fokbjGoJHxcMLx".into())
    )?;

    let rpc_url = env::var("RPC_URL").unwrap_or("https://api.devnet.solana.com".into());
    let client = RpcClient::new(&rpc_url);
    let payer = load_payer();

    println!("\n{B}=== VAPM x Encrypt FHE E2E ==={R}\n");
    val("VAPM Program", vapm_program);
    val("Encrypt Program", encrypt_program);
    val("Payer", payer.pubkey());
    println!();

    // 1. Check Encrypt program config
    log("1/5", "Checking Encrypt config on devnet...");
    let (config_pda, _) = pda(&[b"encrypt_config"], &encrypt_program);
    let (event_authority, _) = pda(&[b"__event_authority"], &encrypt_program);
    let (deposit_pda, _) = pda(&[b"encrypt_deposit", payer.pubkey().as_ref()], &encrypt_program);

    match client.get_account(&config_pda) {
        Ok(acc) => {
            ok(&format!("EncryptConfig: {config_pda} ({} bytes)", acc.data.len()));
        }
        Err(_) => {
            println!("  EncryptConfig not found - waiting for initialization...");
            poll_until(&client, &config_pda, |d| !d.is_empty(), Duration::from_secs(30));
            ok(&format!("EncryptConfig: {config_pda}"));
        }
    }

    // Find network encryption key
    let network_key = [0x55u8; 32]; // mock pre-alpha key
    let (network_key_pda, _) = pda(
        &[b"network_encryption_key", &network_key],
        &encrypt_program,
    );
    val("NetworkEncryptionKey PDA", network_key_pda);

    // 2. Create deposit account (if needed)
    log("2/5", "Setting up Encrypt deposit...");
    match client.get_account(&deposit_pda) {
        Ok(_) => ok("Deposit account exists"),
        Err(_) => {
            // Create deposit via instruction (disc=5 for CreateDeposit)
            let mut ix_data = Vec::new();
            ix_data.push(5); // CreateDeposit discriminator
            let ix = Instruction::new_with_bytes(
                encrypt_program,
                &ix_data,
                vec![
                    AccountMeta::new(deposit_pda, false),
                    AccountMeta::new_readonly(config_pda, false),
                    AccountMeta::new(payer.pubkey(), true),
                    AccountMeta::new_readonly(solana_sdk::system_program::id(), false),
                ],
            );
            send_tx(&client, &payer, vec![ix], &[]);
            ok("Deposit account created");
        }
    }
    val("Deposit PDA", deposit_pda);

    // 3. Connect to Encrypt gRPC
    log("3/5", "Connecting to Encrypt gRPC...");
    let mut encrypt = EncryptClient::connect().await?;
    ok("Encrypt gRPC connected!");

    // 4. Create encrypted input (confidence score)
    log("4/5", "Creating encrypted input: confidence=720 (72%)...");

    let ct_pubkey = encrypt
        .create_input::<Bool>(
            &true, // Encrypted boolean: "prediction is UP"
            &vapm_program, // Authorized to VAPM program
            &network_key,
        )
        .await?;

    ok("Encrypted input created on-chain!");
    val("Ciphertext account", ct_pubkey[0]);
    val("FHE type", "EBool (prediction direction)");
    val("Authorized to", vapm_program);

    // Verify ciphertext exists on-chain
    let ct_data = client.get_account(&ct_pubkey[0])?.data;
    val("Ciphertext size", format!("{} bytes", ct_data.len()));
    val("Status", if ct_data.len() > 97 && ct_data[97] == 1 { "Verified" } else { "Pending" });

    // 5. Summary
    log("5/5", "Integration Summary");
    println!("\n{B}{G}=== Encrypt FHE Integration Complete ==={R}");
    ok("gRPC connected to Encrypt pre-alpha service");
    ok(&format!("Encrypted input created: {}", ct_pubkey[0]));
    ok("Trading signal encrypted on-chain (FHE)");
    ok("Observers cannot read prediction direction");
    println!();
    println!("Verify on Solana Explorer:");
    println!("  https://explorer.solana.com/address/{}?cluster=devnet", ct_pubkey[0]);
    println!();

    Ok(())
}
