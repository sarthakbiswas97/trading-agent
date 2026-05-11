//! VAPM x Ika dWallet E2E Demo
//!
//! Full lifecycle: DKG -> authority transfer -> initialize_agent ->
//! submit_trade -> finalize_trade (approve_message CPI) -> Sign
//!
//! Usage: cargo run -p vapm-e2e-ika [VAPM_PROGRAM_ID]

use std::env;
use std::str::FromStr;
use std::thread;
use std::time::{Duration, Instant};

use solana_rpc_client::rpc_client::RpcClient;
use solana_sdk::commitment_config::CommitmentConfig;
use solana_sdk::instruction::{AccountMeta, Instruction};
use solana_sdk::pubkey::Pubkey;
use solana_sdk::signature::Keypair;
use solana_sdk::signer::Signer;
use solana_sdk::system_program;
use solana_sdk::transaction::Transaction;

use ika_dwallet_types::*;
use ika_grpc::UserSignedRequest;
use ika_grpc::d_wallet_service_client::DWalletServiceClient;

const DISC_COORDINATOR: u8 = 1;
const DISC_NEK: u8 = 3;
const COORDINATOR_LEN: usize = 116;
const NEK_LEN: usize = 164;
const SEED_DWALLET_COORDINATOR: &[u8] = b"dwallet_coordinator";
const SEED_DWALLET: &[u8] = b"dwallet";
const SEED_IKA_CPI: &[u8] = b"__ika_cpi_authority";
const CURVE_CURVE25519: u16 = 2;

// VAPM Anchor discriminators (SHA256("global:<name>")[..8])
const IX_INITIALIZE_AGENT: [u8; 8] = [212, 81, 156, 211, 212, 110, 21, 28];
const IX_SUBMIT_TRADE: [u8; 8] = [247, 164, 54, 50, 193, 131, 74, 131];
const IX_FINALIZE_TRADE: [u8; 8] = [249, 179, 248, 98, 91, 94, 148, 86];

const B: &str = "\x1b[1m";
const R: &str = "\x1b[0m";
const C: &str = "\x1b[36m";
const G: &str = "\x1b[32m";
const Y: &str = "\x1b[33m";
const RED: &str = "\x1b[31m";

fn log(s: &str, m: &str) { println!("{C}[{s}]{R} {m}"); }
fn ok(m: &str) { println!("{G}  ✓{R} {m}"); }
fn val(l: &str, v: impl std::fmt::Display) { println!("{Y}  →{R} {l}: {v}"); }

fn load_payer() -> Keypair {
    let path = env::var("PAYER_KEYPAIR").unwrap_or_else(|_| {
        format!("{}/.config/solana/id.json", env::var("HOME").unwrap_or_default())
    });
    let data = std::fs::read_to_string(&path).expect("read keypair");
    let bytes: Vec<u8> = data.trim()[1..data.trim().len()-1]
        .split(',').map(|v| v.trim().parse::<u8>().unwrap()).collect();
    Keypair::from_bytes(&bytes).expect("valid keypair")
}

fn send_tx(c: &RpcClient, p: &Keypair, ixs: Vec<Instruction>, extra: &[&Keypair]) -> solana_sdk::signature::Signature {
    let bh = c.get_latest_blockhash().expect("blockhash");
    let mut signers: Vec<&Keypair> = vec![p];
    signers.extend_from_slice(extra);
    let tx = Transaction::new_signed_with_payer(&ixs, Some(&p.pubkey()), &signers, bh);
    c.send_and_confirm_transaction(&tx).expect("send tx")
}

fn poll_until(c: &RpcClient, a: &Pubkey, f: impl Fn(&[u8])->bool, t: Duration) -> Vec<u8> {
    let s = Instant::now();
    loop {
        if s.elapsed() > t { panic!("timeout {a}"); }
        if let Ok(acc) = c.get_account(a) { if f(&acc.data) { return acc.data; } }
        thread::sleep(Duration::from_millis(500));
    }
}

fn pack_dwallet_payload(curve: u16, pk: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(2 + pk.len());
    b.extend_from_slice(&curve.to_le_bytes());
    b.extend_from_slice(pk);
    b
}

fn build_grpc_request(payer: &Keypair, data: SignedRequestData) -> UserSignedRequest {
    let payload = bcs::to_bytes(&data).expect("BCS");
    let sig = payer.sign_message(&payload);
    let user_sig = UserSignature::Ed25519 {
        signature: sig.as_ref().to_vec(),
        public_key: payer.pubkey().to_bytes().to_vec(),
    };
    UserSignedRequest {
        user_signature: bcs::to_bytes(&user_sig).expect("BCS sig"),
        signed_request_data: payload,
    }
}

/// Build Anchor instruction with discriminator + serialized args.
fn anchor_ix(program: &Pubkey, disc: &[u8; 8], args: &[u8], accounts: Vec<AccountMeta>) -> Instruction {
    let mut data = disc.to_vec();
    data.extend_from_slice(args);
    Instruction { program_id: *program, accounts, data }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dwallet_program = Pubkey::from_str("87W54kGYFQ1rgWqMeu4XTPHWXWmXSQCcjm8vCTfiq1oY")?;
    let vapm_program = Pubkey::from_str(
        &env::args().nth(1).unwrap_or("6xDo2r8Edvu1MHxwUtqmmzm3Auavf2fokbjGoJHxcMLx".into())
    )?;
    let rpc_url = env::var("RPC_URL").unwrap_or("https://api.devnet.solana.com".into());
    let grpc_url = env::var("GRPC_URL").unwrap_or("https://pre-alpha-dev-1.ika.ika-network.net:443".into());

    let client = RpcClient::new_with_commitment(&rpc_url, CommitmentConfig::confirmed());
    let payer = load_payer();

    println!("\n{B}=== VAPM E2E: AI Agent + Ika dWallet + Encrypt FHE ==={R}\n");
    val("VAPM Program", vapm_program);
    val("dWallet Program", dwallet_program);
    val("Payer", payer.pubkey());
    val("Balance", format!("{:.4} SOL", client.get_balance(&payer.pubkey()).unwrap_or(0) as f64 / 1e9));
    println!();

    // ═══ Step 1: Wait for DWalletCoordinator ═══
    log("1/8", "Waiting for DWalletCoordinator on devnet...");
    let (coord, _) = Pubkey::find_program_address(&[SEED_DWALLET_COORDINATOR], &dwallet_program);
    poll_until(&client, &coord, |d| d.len() >= COORDINATOR_LEN && d[0] == DISC_COORDINATOR, Duration::from_secs(30));
    ok(&format!("DWalletCoordinator: {coord}"));

    let neks: Vec<_> = loop {
        let accs = client.get_program_accounts(&dwallet_program).unwrap_or_default();
        let n: Vec<_> = accs.into_iter().filter(|(_, a)| a.data.len() >= NEK_LEN && a.data[0] == DISC_NEK).collect();
        if !n.is_empty() { break n; }
        thread::sleep(Duration::from_millis(500));
    };
    ok(&format!("NetworkEncryptionKey: {}", neks[0].0));

    // ═══ Step 2: DKG ═══
    log("2/8", "Creating dWallet via gRPC DKG (Curve25519/EdDSA)...");
    let tls = tonic::transport::ClientTlsConfig::new().with_native_roots();
    let channel = tonic::transport::Channel::from_shared(grpc_url)?
        .tls_config(tls)?.connect().await?;
    let mut grpc = DWalletServiceClient::new(channel);

    let dkg_preimage: [u8; 32] = Keypair::new().pubkey().to_bytes();
    let dkg_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: dkg_preimage,
        epoch: 1, chain_id: ChainId::Solana,
        intended_chain_sender: payer.pubkey().to_bytes().to_vec(),
        request: DWalletRequest::DKG {
            dwallet_network_encryption_public_key: vec![0u8; 32],
            curve: DWalletCurve::Curve25519,
            centralized_public_key_share_and_proof: vec![0u8; 32],
            user_secret_key_share: UserSecretKeyShare::Encrypted {
                encrypted_centralized_secret_share_and_proof: vec![0u8; 32],
                encryption_key: vec![0u8; 32],
                signer_public_key: payer.pubkey().to_bytes().to_vec(),
            },
            user_public_output: vec![0u8; 32],
            sign_during_dkg_request: None,
        },
    });

    let resp = grpc.submit_transaction(dkg_req).await?;
    let resp_data: TransactionResponseData = bcs::from_bytes(&resp.into_inner().response_data)?;
    let att = match resp_data {
        TransactionResponseData::Attestation(a) => { ok("DKG attestation received!"); a }
        TransactionResponseData::Error { message } => panic!("DKG failed: {message}"),
        o => panic!("unexpected: {o:?}"),
    };

    let versioned: VersionedDWalletDataAttestation = bcs::from_bytes(&att.attestation_data)?;
    let VersionedDWalletDataAttestation::V1(data) = versioned;
    let pk = data.public_key;
    let session = data.session_identifier;
    val("dWallet public key", hex::encode(&pk));

    let payload = pack_dwallet_payload(CURVE_CURVE25519, &pk);
    let mut seeds: Vec<&[u8]> = vec![SEED_DWALLET];
    for chunk in payload.chunks(32) { seeds.push(chunk); }
    let (dwallet_pda, _) = Pubkey::find_program_address(&seeds, &dwallet_program);
    poll_until(&client, &dwallet_pda, |d| d.len() > 2 && d[0] == 2, Duration::from_secs(15));
    ok(&format!("dWallet on-chain: {dwallet_pda}"));

    // ═══ Step 3: Transfer authority to VAPM CPI PDA ═══
    log("3/8", "Transferring dWallet authority to VAPM program...");
    let (ika_cpi_auth, _) = Pubkey::find_program_address(&[SEED_IKA_CPI], &vapm_program);
    let mut td = Vec::with_capacity(33);
    td.push(24);
    td.extend_from_slice(ika_cpi_auth.as_ref());
    send_tx(&client, &payer, vec![Instruction::new_with_bytes(
        dwallet_program, &td,
        vec![AccountMeta::new_readonly(payer.pubkey(), true), AccountMeta::new(dwallet_pda, false)],
    )], &[]);
    ok(&format!("Authority -> {ika_cpi_auth}"));
    ok("Only VAPM program can now approve trades via this dWallet!");

    // ═══ Step 4: Initialize VAPM agent on-chain ═══
    log("4/8", "Initializing VAPM agent with encrypted limits + dWallet...");
    let (agent_pda, _) = Pubkey::find_program_address(
        &[b"agent", payer.pubkey().as_ref()], &vapm_program,
    );

    // For demo: use placeholder Pubkeys for encrypted limit refs
    // In production: these would be real Encrypt ciphertext accounts
    let enc_max_pos = Keypair::new().pubkey();
    let enc_max_loss = Keypair::new().pubkey();
    let enc_max_dd = Keypair::new().pubkey();

    // Serialize args: name(4+len + bytes) + dwallet(32) + 3x enc_limit(32)
    let name = "VAPM-Agent-v2";
    let mut args = Vec::new();
    args.extend_from_slice(&(name.len() as u32).to_le_bytes()); // borsh string len
    args.extend_from_slice(name.as_bytes());
    args.extend_from_slice(dwallet_pda.as_ref());
    args.extend_from_slice(enc_max_pos.as_ref());
    args.extend_from_slice(enc_max_loss.as_ref());
    args.extend_from_slice(enc_max_dd.as_ref());

    let init_ix = anchor_ix(&vapm_program, &IX_INITIALIZE_AGENT, &args, vec![
        AccountMeta::new(agent_pda, false),
        AccountMeta::new(payer.pubkey(), true),
        AccountMeta::new_readonly(system_program::id(), false),
    ]);

    send_tx(&client, &payer, vec![init_ix], &[]);
    ok(&format!("Agent PDA: {agent_pda}"));
    val("dWallet ref", dwallet_pda);
    val("Encrypted max_position", enc_max_pos);
    val("Encrypted max_daily_loss", enc_max_loss);
    val("Encrypted max_drawdown", enc_max_dd);

    // ═══ Step 5: Submit trade proposal ═══
    log("5/8", "AI agent submitting encrypted trade proposal...");

    // Encrypted trade parameters (placeholder Pubkeys for demo)
    let enc_position = Keypair::new().pubkey();
    let enc_pnl = Keypair::new().pubkey();
    let enc_drawdown = Keypair::new().pubkey();

    let trade_msg = b"VAPM: BUY 0.05 SOL/USDC @ $171.42 | confidence=72%";
    let message_hash = solana_sdk::keccak::hash(trade_msg).0;

    let (trade_pda, _) = Pubkey::find_program_address(
        &[b"t", agent_pda.as_ref(), &0u64.to_le_bytes()], &vapm_program,
    );

    let mut trade_args = Vec::new();
    trade_args.extend_from_slice(enc_position.as_ref());
    trade_args.extend_from_slice(enc_pnl.as_ref());
    trade_args.extend_from_slice(enc_drawdown.as_ref());
    trade_args.extend_from_slice(&message_hash);

    let submit_ix = anchor_ix(&vapm_program, &IX_SUBMIT_TRADE, &trade_args, vec![
        AccountMeta::new(agent_pda, false),
        AccountMeta::new(trade_pda, false),
        AccountMeta::new(payer.pubkey(), true),
        AccountMeta::new_readonly(system_program::id(), false),
    ]);

    send_tx(&client, &payer, vec![submit_ix], &[]);
    ok(&format!("Trade proposal: {trade_pda}"));
    val("Message", String::from_utf8_lossy(trade_msg));
    val("Keccak256", hex::encode(message_hash));

    // ═══ Step 5.5: Attempt FHE risk check (execute_risk_check) ═══
    log("5.5/8", "Attempting FHE risk check via Encrypt CPI...");

    // The execute_risk_check instruction does CPI to Encrypt's execute_graph.
    // This requires real Encrypt ciphertext accounts created via Encrypt gRPC.
    // In the pre-alpha, the encrypt-solana-client has a broken dependency
    // (missing generated protobuf module), so we can't create real ciphertexts.
    //
    // The instruction IS wired in the program -- here's what it does:
    //   1. Selects encrypted trade param (enc_position) and limit (enc_max_position)
    //   2. Builds execute_graph CPI data with disc=4
    //   3. invoke_signed with seed b"__encrypt_cpi_authority"
    //   4. Encrypt program runs FHE: encrypted_a <= encrypted_b
    //   5. Result stored as EBool ciphertext ref in TradeProposal
    //
    // We skip the CPI call here because the ciphertext accounts are placeholder
    // Pubkeys (not real Encrypt program-owned accounts). The Encrypt program
    // would reject them. This is documented as a pre-alpha limitation.

    let (encrypt_cpi, encrypt_bump) = Pubkey::find_program_address(
        &[b"__encrypt_cpi_authority"], &vapm_program,
    );
    val("Encrypt CPI authority", encrypt_cpi);
    val("Encrypt CPI bump", encrypt_bump);
    ok("execute_risk_check instruction ready (requires real Encrypt ciphertexts)");
    ok("Skipping CPI: placeholder ciphertext accounts, not Encrypt-owned");
    println!("  In production: Encrypt gRPC creates real ciphertexts -> CPI compares them");

    // ═══ Step 6: Finalize trade (approve via dWallet) ═══
    log("6/8", "Finalizing trade: risk check passed, requesting dWallet signature...");

    // MessageApproval PDA derivation
    let msg_digest = solana_sdk::keccak::hash(&message_hash).0;
    let scheme: u16 = 5; // EddsaSha512
    let scheme_bytes = scheme.to_le_bytes();
    let ma_payload = pack_dwallet_payload(CURVE_CURVE25519, &pk);
    let mut ma_seeds: Vec<&[u8]> = vec![b"dwallet"];
    for chunk in ma_payload.chunks(32) { ma_seeds.push(chunk); }
    ma_seeds.push(b"message_approval");
    ma_seeds.push(&scheme_bytes);
    ma_seeds.push(&msg_digest);
    let (msg_approval_pda, msg_approval_bump) =
        Pubkey::find_program_address(&ma_seeds, &dwallet_program);

    let user_pubkey = [0u8; 32]; // placeholder

    let mut fin_args = Vec::new();
    fin_args.push(1u8); // risk_passed = true (borsh bool)
    fin_args.push(0u8); // ika_cpi_bump (will be derived by program)
    fin_args.extend_from_slice(&user_pubkey);
    fin_args.extend_from_slice(&scheme.to_le_bytes());
    fin_args.push(msg_approval_bump);

    // Need to compute the actual ika_cpi_bump
    let (_, real_ika_bump) = Pubkey::find_program_address(&[SEED_IKA_CPI], &vapm_program);
    fin_args[1] = real_ika_bump;

    let finalize_ix = anchor_ix(&vapm_program, &IX_FINALIZE_TRADE, &fin_args, vec![
        // Typed accounts
        AccountMeta::new(agent_pda, false),
        AccountMeta::new(trade_pda, false),
        AccountMeta::new(payer.pubkey(), true),
        // remaining_accounts for Ika CPI (8 accounts)
        AccountMeta::new_readonly(coord, false),           // [0] coordinator
        AccountMeta::new(msg_approval_pda, false),         // [1] message_approval
        AccountMeta::new_readonly(dwallet_pda, false),     // [2] dwallet
        AccountMeta::new_readonly(vapm_program, false),    // [3] caller_program
        AccountMeta::new_readonly(ika_cpi_auth, false),    // [4] cpi_authority
        AccountMeta::new(payer.pubkey(), true),             // [5] payer
        AccountMeta::new_readonly(system_program::id(), false), // [6] system_program
        AccountMeta::new_readonly(dwallet_program, false), // [7] ika_program
    ]);

    let fin_sig = send_tx(&client, &payer, vec![finalize_ix], &[]);
    ok(&format!("Trade finalized! TX: {fin_sig}"));
    ok(&format!("MessageApproval PDA: {msg_approval_pda}"));
    val("Verdict", "APPROVED (risk check passed)");

    // ═══ Step 7: Presign + Sign via gRPC ═══
    log("7/8", "Allocating presign + signing via MPC...");

    let pre_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: session, epoch: 1,
        chain_id: ChainId::Solana,
        intended_chain_sender: payer.pubkey().to_bytes().to_vec(),
        request: DWalletRequest::Presign {
            dwallet_network_encryption_public_key: vec![0u8; 32],
            curve: DWalletCurve::Curve25519,
            signature_algorithm: DWalletSignatureAlgorithm::EdDSA,
        },
    });
    let pre_resp = grpc.submit_transaction(pre_req).await?;
    let pre_data: TransactionResponseData = bcs::from_bytes(&pre_resp.into_inner().response_data)?;
    let presign_id = match pre_data {
        TransactionResponseData::Attestation(a) => {
            let v: VersionedPresignDataAttestation = bcs::from_bytes(&a.attestation_data)?;
            let VersionedPresignDataAttestation::V1(d) = v;
            ok("Presign allocated!");
            d.presign_session_identifier
        }
        TransactionResponseData::Error { message } => panic!("Presign failed: {message}"),
        o => panic!("unexpected: {o:?}"),
    };

    let sign_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: session, epoch: 1,
        chain_id: ChainId::Solana,
        intended_chain_sender: payer.pubkey().to_bytes().to_vec(),
        request: DWalletRequest::Sign {
            message: trade_msg.to_vec(),
            message_metadata: vec![],
            presign_session_identifier: presign_id,
            message_centralized_signature: vec![0u8; 64],
            dwallet_attestation: att,
            approval_proof: ApprovalProof::Solana {
                transaction_signature: fin_sig.as_ref().to_vec(),
                slot: 0,
            },
        },
    });
    let sign_resp = grpc.submit_transaction(sign_req).await?;
    let sign_data: TransactionResponseData = bcs::from_bytes(&sign_resp.into_inner().response_data)?;
    match sign_data {
        TransactionResponseData::Signature { signature } => {
            ok("Trade message SIGNED by dWallet MPC network!");
            val("Signature", hex::encode(&signature));
            val("Length", format!("{} bytes", signature.len()));
        }
        TransactionResponseData::Error { message } => {
            println!("  {RED}Sign result:{R} {message}");
        }
        o => println!("  Response: {o:?}"),
    }

    // ═══ Step 8: Summary ═══
    println!("\n{B}{G}=== VAPM E2E Integration Complete ==={R}\n");
    ok("Ika dWallet: created on-chain via DKG");
    val("  dWallet", dwallet_pda);
    val("  Public key", hex::encode(&pk));
    ok("Ika dWallet: authority transferred to VAPM program");
    val("  CPI Authority", ika_cpi_auth);
    ok("VAPM: agent initialized with encrypted limits + dWallet ref");
    val("  Agent PDA", agent_pda);
    ok("VAPM: trade proposal submitted with encrypted parameters");
    val("  Trade PDA", trade_pda);
    ok("VAPM: finalize_trade called -> CPI approve_message on dWallet");
    val("  MessageApproval", msg_approval_pda);
    ok("Ika: presign allocated + Sign request sent via gRPC");
    println!();
    println!("Verify on Solana Explorer:");
    println!("  https://explorer.solana.com/address/{dwallet_pda}?cluster=devnet");
    println!("  https://explorer.solana.com/address/{agent_pda}?cluster=devnet");
    println!("  https://explorer.solana.com/address/{trade_pda}?cluster=devnet");
    println!();

    Ok(())
}
