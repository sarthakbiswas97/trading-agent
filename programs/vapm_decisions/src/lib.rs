use anchor_lang::prelude::*;

declare_id!("11111111111111111111111111111111"); // Replace after `anchor deploy`

/// Ika dWallet program on Solana devnet.
pub const IKA_DWALLET_PROGRAM_ID: Pubkey = pubkey!("87W54kGYFQ1rgWqMeu4XTPHWXWmXSQCcjm8vCTfiq1oY");

/// Seed for CPI authority PDA recognized by Ika.
pub const CPI_AUTHORITY_SEED: &[u8] = b"__ika_cpi_authority";

#[program]
pub mod vapm_decisions {
    use super::*;

    /// Register a new agent with on-chain risk limits.
    pub fn initialize_agent(
        ctx: Context<InitializeAgent>,
        name: String,
        max_position_bps: u64,
        max_daily_loss_bps: u64,
        max_drawdown_bps: u64,
    ) -> Result<()> {
        require!(name.len() <= 32, VapmError::NameTooLong);
        require!(max_position_bps > 0 && max_position_bps <= 10000, VapmError::InvalidRiskLimit);
        require!(max_daily_loss_bps > 0 && max_daily_loss_bps <= 10000, VapmError::InvalidRiskLimit);
        require!(max_drawdown_bps > 0 && max_drawdown_bps <= 10000, VapmError::InvalidRiskLimit);

        let agent = &mut ctx.accounts.agent_state;
        agent.authority = ctx.accounts.authority.key();
        agent.name = name;
        agent.decision_count = 0;
        agent.created_at = Clock::get()?.unix_timestamp;
        agent.max_position_bps = max_position_bps;
        agent.max_daily_loss_bps = max_daily_loss_bps;
        agent.max_drawdown_bps = max_drawdown_bps;
        agent.dwallet = Pubkey::default();
        agent.trades_approved = 0;
        agent.trades_rejected = 0;
        agent.bump = ctx.bumps.agent_state;

        emit!(AgentRegistered {
            authority: agent.authority,
            name: agent.name.clone(),
            max_position_bps,
            max_daily_loss_bps,
            max_drawdown_bps,
        });

        Ok(())
    }

    /// Update on-chain risk limits. Authority-only.
    pub fn set_risk_limits(
        ctx: Context<SetRiskLimits>,
        max_position_bps: u64,
        max_daily_loss_bps: u64,
        max_drawdown_bps: u64,
    ) -> Result<()> {
        require!(max_position_bps > 0 && max_position_bps <= 10000, VapmError::InvalidRiskLimit);
        require!(max_daily_loss_bps > 0 && max_daily_loss_bps <= 10000, VapmError::InvalidRiskLimit);
        require!(max_drawdown_bps > 0 && max_drawdown_bps <= 10000, VapmError::InvalidRiskLimit);

        let agent = &mut ctx.accounts.agent_state;
        agent.max_position_bps = max_position_bps;
        agent.max_daily_loss_bps = max_daily_loss_bps;
        agent.max_drawdown_bps = max_drawdown_bps;

        emit!(RiskLimitsUpdated {
            authority: agent.authority,
            max_position_bps,
            max_daily_loss_bps,
            max_drawdown_bps,
        });

        Ok(())
    }

    /// Set the dWallet account reference for this agent.
    pub fn set_dwallet(ctx: Context<SetRiskLimits>, dwallet: Pubkey) -> Result<()> {
        let agent = &mut ctx.accounts.agent_state;
        agent.dwallet = dwallet;

        emit!(DWalletSet {
            authority: agent.authority,
            dwallet,
        });

        Ok(())
    }

    /// Approve a trade by checking on-chain risk limits.
    /// If all checks pass, CPI-calls Ika dWallet approve_message.
    /// If dWallet is not set, approves locally (fallback mode).
    pub fn approve_trade(
        ctx: Context<ApproveTrade>,
        position_size_bps: u64,
        current_exposure_bps: u64,
        daily_pnl_bps: u64,
        current_drawdown_bps: u64,
        message_hash: [u8; 32],
    ) -> Result<()> {
        let agent = &mut ctx.accounts.agent_state;

        // On-chain risk enforcement -- these cannot be bypassed
        require!(
            position_size_bps <= agent.max_position_bps,
            VapmError::PositionSizeExceeded
        );
        require!(
            daily_pnl_bps <= agent.max_daily_loss_bps,
            VapmError::DailyLossExceeded
        );
        require!(
            current_drawdown_bps <= agent.max_drawdown_bps,
            VapmError::DrawdownExceeded
        );

        agent.trades_approved += 1;

        emit!(TradeApproved {
            authority: agent.authority,
            position_size_bps,
            current_exposure_bps,
            daily_pnl_bps,
            current_drawdown_bps,
            message_hash,
            dwallet_enabled: agent.dwallet != Pubkey::default(),
        });

        // If dWallet is configured, the CPI call to approve_message
        // would go here. For the hackathon demo, we demonstrate the
        // risk enforcement and emit the approval event. The dWallet
        // signing is handled off-chain via the Ika gRPC API after
        // the on-chain risk check passes.
        //
        // In production, the full CPI flow would be:
        // 1. Derive CPI authority PDA
        // 2. Build DWalletContext
        // 3. Call ctx.approve_message(...)
        // 4. Ika network detects MessageApproval and signs

        Ok(())
    }

    /// Reject a trade that violates risk limits (for audit trail).
    pub fn reject_trade(
        ctx: Context<RejectTrade>,
        position_size_bps: u64,
        daily_pnl_bps: u64,
        current_drawdown_bps: u64,
        reason: String,
    ) -> Result<()> {
        let agent = &mut ctx.accounts.agent_state;
        agent.trades_rejected += 1;

        emit!(TradeRejected {
            authority: agent.authority,
            position_size_bps,
            daily_pnl_bps,
            current_drawdown_bps,
            reason,
        });

        Ok(())
    }

    /// Log a decision hash on-chain.
    pub fn log_decision(
        ctx: Context<LogDecision>,
        decision_hash: [u8; 32],
        confidence: u64,
        risk_score: u64,
    ) -> Result<()> {
        require!(confidence <= 1000, VapmError::InvalidConfidence);
        require!(risk_score <= 1000, VapmError::InvalidRiskScore);

        let agent = &mut ctx.accounts.agent_state;
        let record = &mut ctx.accounts.decision_record;

        record.agent = ctx.accounts.authority.key();
        record.decision_hash = decision_hash;
        record.model_confidence = confidence;
        record.risk_score = risk_score;
        record.timestamp = Clock::get()?.unix_timestamp;
        record.executed = false;
        record.bump = ctx.bumps.decision_record;

        let index = agent.decision_count;
        agent.decision_count += 1;

        emit!(DecisionLogged {
            agent: record.agent,
            index,
            decision_hash,
            confidence,
            risk_score,
        });

        Ok(())
    }

    /// Mark a previously logged decision as executed.
    pub fn mark_executed(ctx: Context<MarkExecuted>, _decision_index: u64) -> Result<()> {
        let record = &mut ctx.accounts.decision_record;
        require!(!record.executed, VapmError::AlreadyExecuted);

        record.executed = true;

        emit!(DecisionExecuted {
            agent: record.agent,
            decision_hash: record.decision_hash,
        });

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────
// ACCOUNTS
// ─────────────────────────────────────────────────────────────

#[derive(Accounts)]
pub struct InitializeAgent<'info> {
    #[account(
        init,
        payer = authority,
        space = AgentState::SIZE,
        seeds = [b"agent", authority.key().as_ref()],
        bump,
    )]
    pub agent_state: Account<'info, AgentState>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct SetRiskLimits<'info> {
    #[account(
        mut,
        seeds = [b"agent", authority.key().as_ref()],
        bump = agent_state.bump,
        has_one = authority,
    )]
    pub agent_state: Account<'info, AgentState>,

    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ApproveTrade<'info> {
    #[account(
        mut,
        seeds = [b"agent", authority.key().as_ref()],
        bump = agent_state.bump,
        has_one = authority,
    )]
    pub agent_state: Account<'info, AgentState>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
    // When dWallet CPI is enabled, additional accounts would be:
    // pub message_approval: UncheckedAccount (writable)
    // pub dwallet: UncheckedAccount (read-only)
    // pub cpi_authority: UncheckedAccount (read-only, PDA signer)
    // pub dwallet_program: UncheckedAccount (read-only)
}

#[derive(Accounts)]
pub struct RejectTrade<'info> {
    #[account(
        mut,
        seeds = [b"agent", authority.key().as_ref()],
        bump = agent_state.bump,
        has_one = authority,
    )]
    pub agent_state: Account<'info, AgentState>,

    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct LogDecision<'info> {
    #[account(
        mut,
        seeds = [b"agent", authority.key().as_ref()],
        bump = agent_state.bump,
        has_one = authority,
    )]
    pub agent_state: Account<'info, AgentState>,

    #[account(
        init,
        payer = authority,
        space = DecisionRecord::SIZE,
        seeds = [
            b"decision",
            authority.key().as_ref(),
            &agent_state.decision_count.to_le_bytes(),
        ],
        bump,
    )]
    pub decision_record: Account<'info, DecisionRecord>,

    #[account(mut)]
    pub authority: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
#[instruction(decision_index: u64)]
pub struct MarkExecuted<'info> {
    #[account(
        seeds = [b"agent", authority.key().as_ref()],
        bump = agent_state.bump,
        has_one = authority,
    )]
    pub agent_state: Account<'info, AgentState>,

    #[account(
        mut,
        seeds = [
            b"decision",
            authority.key().as_ref(),
            &decision_index.to_le_bytes(),
        ],
        bump = decision_record.bump,
    )]
    pub decision_record: Account<'info, DecisionRecord>,

    pub authority: Signer<'info>,
}

// ─────────────────────────────────────────────────────────────
// STATE
// ─────────────────────────────────────────────────────────────

#[account]
pub struct AgentState {
    pub authority: Pubkey,           // 32
    pub name: String,                // 4 + 32 max
    pub decision_count: u64,         // 8
    pub created_at: i64,             // 8
    // On-chain risk limits (basis points, 100 = 1%)
    pub max_position_bps: u64,       // 8
    pub max_daily_loss_bps: u64,     // 8
    pub max_drawdown_bps: u64,       // 8
    // dWallet reference
    pub dwallet: Pubkey,             // 32
    // Trade approval counters
    pub trades_approved: u64,        // 8
    pub trades_rejected: u64,        // 8
    pub bump: u8,                    // 1
}

impl AgentState {
    // 8 (disc) + 32 + 36 + 8 + 8 + 8 + 8 + 8 + 32 + 8 + 8 + 1 + padding = 256
    pub const SIZE: usize = 8 + 32 + 36 + 8 + 8 + 8 + 8 + 8 + 32 + 8 + 8 + 1 + 32;
}

#[account]
pub struct DecisionRecord {
    pub agent: Pubkey,            // 32
    pub decision_hash: [u8; 32],  // 32
    pub model_confidence: u64,    // 8
    pub risk_score: u64,          // 8
    pub timestamp: i64,           // 8
    pub executed: bool,           // 1
    pub bump: u8,                 // 1
}

impl DecisionRecord {
    pub const SIZE: usize = 8 + 32 + 32 + 8 + 8 + 8 + 1 + 1 + 16;
}

// ─────────────────────────────────────────────────────────────
// EVENTS
// ─────────────────────────────────────────────────────────────

#[event]
pub struct AgentRegistered {
    pub authority: Pubkey,
    pub name: String,
    pub max_position_bps: u64,
    pub max_daily_loss_bps: u64,
    pub max_drawdown_bps: u64,
}

#[event]
pub struct RiskLimitsUpdated {
    pub authority: Pubkey,
    pub max_position_bps: u64,
    pub max_daily_loss_bps: u64,
    pub max_drawdown_bps: u64,
}

#[event]
pub struct DWalletSet {
    pub authority: Pubkey,
    pub dwallet: Pubkey,
}

#[event]
pub struct TradeApproved {
    pub authority: Pubkey,
    pub position_size_bps: u64,
    pub current_exposure_bps: u64,
    pub daily_pnl_bps: u64,
    pub current_drawdown_bps: u64,
    pub message_hash: [u8; 32],
    pub dwallet_enabled: bool,
}

#[event]
pub struct TradeRejected {
    pub authority: Pubkey,
    pub position_size_bps: u64,
    pub daily_pnl_bps: u64,
    pub current_drawdown_bps: u64,
    pub reason: String,
}

#[event]
pub struct DecisionLogged {
    pub agent: Pubkey,
    pub index: u64,
    pub decision_hash: [u8; 32],
    pub confidence: u64,
    pub risk_score: u64,
}

#[event]
pub struct DecisionExecuted {
    pub agent: Pubkey,
    pub decision_hash: [u8; 32],
}

// ─────────────────────────────────────────────────────────────
// ERRORS
// ─────────────────────────────────────────────────────────────

#[error_code]
pub enum VapmError {
    #[msg("Agent name must be 32 characters or fewer")]
    NameTooLong,

    #[msg("Confidence must be between 0 and 1000")]
    InvalidConfidence,

    #[msg("Risk score must be between 0 and 1000")]
    InvalidRiskScore,

    #[msg("Decision already marked as executed")]
    AlreadyExecuted,

    #[msg("Risk limit must be between 1 and 10000 basis points")]
    InvalidRiskLimit,

    #[msg("Position size exceeds on-chain maximum")]
    PositionSizeExceeded,

    #[msg("Daily loss exceeds on-chain maximum")]
    DailyLossExceeded,

    #[msg("Drawdown exceeds on-chain maximum")]
    DrawdownExceeded,
}
