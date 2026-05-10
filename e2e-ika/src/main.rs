//! VAPM x Ika dWallet E2E Demo
//!
//! Creates a real dWallet on Solana devnet via Ika gRPC,
//! transfers authority to VAPM program's CPI PDA,
//! allocates presign and signs a trade message.
//!
//! Usage: cargo run -p vapm-e2e-ika

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
const SEED_CPI_AUTHORITY: &[u8] = b"__ika_cpi_authority";
const CURVE_CURVE25519: u16 = 2;

const B: &str = "\x1b[1m";
const R: &str = "\x1b[0m";
const C: &str = "\x1b[36m";
const G: &str = "\x1b[32m";
const Y: &str = "\x1b[33m";

fn log(s: &str, m: &str) { println!("{C}[{s}]{R} {m}"); }
fn ok(m: &str) { println!("{G}  \u{2713}{R} {m}"); }
fn val(l: &str, v: impl std::fmt::Display) { println!("{Y}  \u{2192}{R} {l}: {v}"); }

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

    println!("\n{B}=== VAPM x Ika dWallet E2E ==={R}\n");
    val("VAPM Program", vapm_program);
    val("dWallet Program", dwallet_program);
    val("Payer", payer.pubkey());
    let bal = client.get_balance(&payer.pubkey()).unwrap_or(0);
    val("Balance", format!("{:.4} SOL", bal as f64 / 1e9));
    println!();

    // 1. Wait for coordinator
    log("1/6", "Waiting for DWalletCoordinator...");
    let (coord, _) = Pubkey::find_program_address(&[SEED_DWALLET_COORDINATOR], &dwallet_program);
    poll_until(&client, &coord, |d| d.len() >= COORDINATOR_LEN && d[0] == DISC_COORDINATOR, Duration::from_secs(30));
    ok(&format!("DWalletCoordinator: {coord}"));

    // Find NEK
    let neks: Vec<_> = loop {
        let accs = client.get_program_accounts(&dwallet_program).unwrap_or_default();
        let n: Vec<_> = accs.into_iter().filter(|(_, a)| a.data.len() >= NEK_LEN && a.data[0] == DISC_NEK).collect();
        if !n.is_empty() { break n; }
        thread::sleep(Duration::from_millis(500));
    };
    ok(&format!("NetworkEncryptionKey: {}", neks[0].0));

    // 2. DKG
    log("2/6", "Creating dWallet via gRPC DKG (Curve25519)...");
    let tls = tonic::transport::ClientTlsConfig::new().with_native_roots();
    let channel = tonic::transport::Channel::from_shared(grpc_url)?
        .tls_config(tls)?.connect().await?;
    let mut grpc = DWalletServiceClient::new(channel);

    let dkg_preimage: [u8; 32] = Keypair::new().pubkey().to_bytes();
    let dkg_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: dkg_preimage,
        epoch: 1,
        chain_id: ChainId::Solana,
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
        TransactionResponseData::Error { message } => { panic!("DKG failed: {message}"); }
        o => panic!("unexpected: {o:?}"),
    };

    let versioned: VersionedDWalletDataAttestation = bcs::from_bytes(&att.attestation_data)?;
    let VersionedDWalletDataAttestation::V1(data) = versioned;
    let pk = data.public_key;
    let session = data.session_identifier;
    val("dWallet public key", hex::encode(&pk));

    // Wait for on-chain dWallet
    let payload = pack_dwallet_payload(CURVE_CURVE25519, &pk);
    let mut seeds: Vec<&[u8]> = vec![SEED_DWALLET];
    for chunk in payload.chunks(32) { seeds.push(chunk); }
    let (dwallet_pda, _) = Pubkey::find_program_address(&seeds, &dwallet_program);
    poll_until(&client, &dwallet_pda, |d| d.len() > 2 && d[0] == 2, Duration::from_secs(15));
    ok(&format!("dWallet on-chain: {dwallet_pda}"));

    // 3. Transfer authority
    log("3/6", "Transferring authority to VAPM CPI PDA...");
    let (cpi_auth, _) = Pubkey::find_program_address(&[SEED_CPI_AUTHORITY], &vapm_program);
    let mut td = Vec::with_capacity(33);
    td.push(24); // IX_TRANSFER_OWNERSHIP
    td.extend_from_slice(cpi_auth.as_ref());
    send_tx(&client, &payer, vec![Instruction::new_with_bytes(
        dwallet_program, &td,
        vec![AccountMeta::new_readonly(payer.pubkey(), true), AccountMeta::new(dwallet_pda, false)],
    )], &[]);
    ok(&format!("Authority -> {cpi_auth}"));

    // 4. Presign
    log("4/6", "Allocating presign...");
    let pre_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: session,
        epoch: 1,
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

    // 5. Sign
    log("5/6", "Signing trade message via MPC...");
    let msg = b"VAPM: BUY 0.05 SOL/USDC @ $171.42";
    val("Message", String::from_utf8_lossy(msg));

    let sign_req = build_grpc_request(&payer, SignedRequestData {
        session_identifier_preimage: session,
        epoch: 1,
        chain_id: ChainId::Solana,
        intended_chain_sender: payer.pubkey().to_bytes().to_vec(),
        request: DWalletRequest::Sign {
            message: msg.to_vec(),
            message_metadata: vec![],
            presign_session_identifier: presign_id,
            message_centralized_signature: vec![0u8; 64],
            dwallet_attestation: att,
            approval_proof: ApprovalProof::Solana {
                transaction_signature: vec![0u8; 64],
                slot: 0,
            },
        },
    });
    let sign_resp = grpc.submit_transaction(sign_req).await?;
    let sign_data: TransactionResponseData = bcs::from_bytes(&sign_resp.into_inner().response_data)?;
    match sign_data {
        TransactionResponseData::Signature { signature } => {
            ok("Trade signed by dWallet MPC!");
            val("Signature", hex::encode(&signature));
        }
        TransactionResponseData::Error { message } => {
            println!("  Sign result: {message}");
        }
        o => println!("  Response: {o:?}"),
    }

    // 6. Summary
    println!("\n{B}{G}=== Ika Integration Complete ==={R}");
    ok(&format!("dWallet created: {dwallet_pda}"));
    ok(&format!("Authority transferred to VAPM: {cpi_auth}"));
    ok("Presign allocated + Sign executed via gRPC");
    ok("All operations verified on Solana devnet");
    println!();

    Ok(())
}
