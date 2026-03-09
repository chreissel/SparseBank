# ─────────────────────────────────────────────────────────────
# STEP 3 – Per-event template bank filtering
# ─────────────────────────────────────────────────────────────

def _load_model(ckpt_path: Path):
    """
    Load the trained S4D regression model from a Lightning checkpoint.
    Returns model in eval mode on CPU.
    """
    import torch

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Reconstruct model from saved hyperparameters
    hparams     = ckpt.get("hyper_parameters", {})
    model_cfg   = hparams.get("model_cfg", {
        "d_input": 1, "d_output": 1, "d_model": 64,
        "d_state": 16, "n_layers": 4, "dropout": 0.1,
    })

    # inline minimal S4D model (mirrors step 2 definition)
    import torch.nn as nn

    class S4DLayer(nn.Module):
        def __init__(self, d_model, d_state, dropout=0.0):
            super().__init__()
            self.kernel  = nn.Linear(d_model, d_model, bias=False)
            self.norm    = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)
            self.output  = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.norm(x + self.dropout(self.output(self.kernel(x))))

    class S4Model(nn.Module):
        def __init__(self, d_input, d_output, d_model, d_state, n_layers, dropout):
            super().__init__()
            self.encoder = nn.Linear(d_input, d_model)
            self.layers  = nn.ModuleList([
                S4DLayer(d_model, d_state, dropout) for _ in range(n_layers)])
            self.decoder = nn.Linear(d_model, d_output)
        def forward(self, x):
            x = self.encoder(x)
            for layer in self.layers:
                x = layer(x)
            return self.decoder(x.mean(dim=1))

    model = S4Model(**model_cfg)
    state_dict = {k.replace("model.", ""): v
                  for k, v in ckpt["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def _predict_chirp_mass(model, strain: np.ndarray) -> float:
    """
    Run inference on a single strain window.
    Returns predicted chirp mass (scalar, M_sun).
    """
    import torch
    x = torch.tensor(strain, dtype=torch.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    # TODO: check if this is really how we want to do the normalization
    x = x.unsqueeze(0).unsqueeze(-1)   # (1, L, 1)
    with torch.no_grad():
        pred = model(x)
    return float(pred.squeeze())


def _prune_bank(bank_in: Path, bank_out: Path,
                mc_pred: float, margin: float) -> int:
    """
    Copy bank_in → bank_out keeping only templates within
    [mc_pred - margin, mc_pred + margin].
    Returns number of templates kept.
    """
    mc_min = mc_pred - margin
    mc_max = mc_pred + margin

    def chirp_mass(row):
        m1, m2 = row.mass1, row.mass2
        return (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2

    try:
        from ligo.lw import ligolw, lsctables, utils as ligolw_utils

        xmldoc = ligolw_utils.load_filename(
            str(bank_in),
            contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
        sngl_table  = lsctables.SnglInspiralTable.get_table(xmldoc)
        keep        = [r for r in sngl_table if mc_min <= chirp_mass(r) <= mc_max]
        sngl_table[:] = keep
        ligolw_utils.write_filename(xmldoc, str(bank_out))
        return len(keep)

    except ImportError:
        # fallback: write a plain-text mass range file
        with open(bank_out.with_suffix(".txt"), "w") as fh:
            fh.write(f"chirp_mass_min={mc_min:.6f}\nchirp_mass_max={mc_max:.6f}\n")
        return -1   # unknown count


def filter_template_bank_per_event(cfg: dict, ckpt_path: Path) -> list[dict]:
    """
    For every event in the test set:
      1. Run the trained model to predict chirp mass
      2. Write a dedicated filtered template bank for that event

    Returns a list of dicts:
      [{"event_id": int, "mc_pred": float, "bank_path": Path}, ...]
    """
    import torch
    import h5py

    log.info("[Step 3] Loading model …")
    model = _load_model(ckpt_path)

    test_file = Path(cfg["data"]["data_dir"]) / "test" / "sig_combined_test.h5"
    with h5py.File(test_file, "r") as f:
        strains    = f["injected_data"][:]   # (N, L)
        y_true     = f["chirp_mass"][:]

    bank_in    = Path(cfg["bank_filter"]["input_bank"])
    banks_dir  = Path(cfg["bank_filter"].get("per_event_dir", "templates/per_event"))
    banks_dir.mkdir(parents=True, exist_ok=True)
    margin     = cfg["bank_filter"].get("margin", 0.1)

    n_events   = len(strains)
    log.info(f"[Step 3] Filtering bank for {n_events} events (margin ±{margin} M_sun) …")

    event_banks = []
    n_before    = None   # read once

    for i, strain in enumerate(strains):
        mc_pred  = _predict_chirp_mass(model, strain)
        bank_out = banks_dir / f"bank_event_{i:06d}.xml.gz"

        n_kept = _prune_bank(bank_in, bank_out, mc_pred, margin)

        if n_before is None and n_kept >= 0:
            # read full bank size once for logging
            try:
                from ligo.lw import ligolw, lsctables, utils as ligolw_utils
                xmldoc   = ligolw_utils.load_filename(
                    str(bank_in),
                    contenthandler=lsctables.use_in(ligolw.LIGOLWContentHandler))
                n_before = len(lsctables.SnglInspiralTable.get_table(xmldoc))
            except Exception:
                n_before = -1

        log.info(
            f"  event {i:06d} | mc_pred={mc_pred:.4f} M_sun "
            f"| templates: {n_before} → {n_kept} | {bank_out.name}"
        )

        event_banks.append({
            "event_id":  i,
            "mc_pred":   mc_pred,
            "mc_true":   float(y_true[i]),
            "bank_path": bank_out,
            "n_kept":    n_kept,
        })

    log.info(f"[Step 3] Done. {n_events} per-event banks written to {banks_dir}")
    return event_banks

