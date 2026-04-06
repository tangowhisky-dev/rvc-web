import torch
import torch.nn.functional as F


def speaker_loss(gen_emb, ref_emb):
    """Cosine speaker identity loss.

    Args:
        gen_emb: Speaker embedding of generated audio [B, D].
        ref_emb: Speaker embedding of target audio [B, D] (detached).

    Returns:
        Scalar loss: 1 - cosine_similarity. Range [0, 2].
    """
    sim = F.cosine_similarity(gen_emb, ref_emb).mean()
    return 1.0 - sim


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """LSGAN discriminator loss: MSE toward 1 (real) and 0 (fake)."""
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    """LSGAN generator loss: MSE toward 1 (fool discriminator)."""
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def discriminator_tprls_loss(disc_real_outputs, disc_generated_outputs):
    """TPRLS discriminator loss (Truncated Paired Relative Least Squares).

    More robust than LSGAN when the discriminator dominates early training.
    Uses median-centering to reduce mode-collapse risk.
    Reference: codename-rvc-fork-4/rvc/train/losses.py
    """
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        m_DG = torch.median(dr - dg)
        diff = (dr - dg) - m_DG
        mask = dr < (dg + m_DG)
        masked = diff[mask]
        L_rel = (
            torch.mean(masked ** 2)
            if masked.numel() > 0
            else torch.tensor(0.0, device=dr.device)
        )
        loss += tau - F.relu(tau - L_rel)
    return loss


def generator_tprls_loss(disc_real_outputs, disc_generated_outputs):
    """TPRLS generator loss — paired with discriminator_tprls_loss."""
    loss = 0
    tau = 0.04
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        diff = dg - dr
        m_DG = torch.median(diff)
        rel = diff - m_DG
        mask = diff < m_DG
        masked = rel[mask]
        L_rel = (
            torch.mean(masked ** 2)
            if masked.numel() > 0
            else torch.tensor(0.0, device=dg.device)
        )
        loss += tau - F.relu(tau - L_rel)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
