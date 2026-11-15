use std::process::Command;

/// Snapshot of GPU memory at probe time.
#[derive(Debug, Clone, Copy)]
pub struct GpuInfo {
    pub free_mib: u64,
    pub total_mib: u64,
}

/// Query the first NVIDIA GPU via `nvidia-smi`. Returns None if nvidia-smi
/// is not on PATH or the call fails (no NVIDIA driver, GPU off, multi-card
/// systems where the driver isn't responding, etc.). One-shot subprocess
/// per pipeline run -- the overhead is negligible vs the inference cost.
pub fn probe_nvidia_gpu() -> Option<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8_lossy(&output.stdout);
    let first = line.lines().next()?;
    let mut parts = first.split(',');
    let free: u64 = parts.next()?.trim().parse().ok()?;
    let total: u64 = parts.next()?.trim().parse().ok()?;
    Some(GpuInfo {
        free_mib: free,
        total_mib: total,
    })
}

// `compute_parallelism` used to live here. It divided free VRAM by a
// per-instance budget to decide how many `prkt.exe` processes could run at
// once, each holding its own copy of the same weights. Parakeet now loads one
// resident model per run, so there is nothing left to divide; `probe_nvidia_gpu`
// above is retained purely to report free VRAM in the phase banner.
