use regex::Regex;

/// A detected silence gap from FFmpeg silencedetect output.
#[derive(Debug, Clone)]
pub struct SilenceGap {
    /// End time of the silence period.
    pub end: f64,
    /// Duration of the silence period.
    pub duration: f64,
    /// Midpoint of the silence (best split point).
    pub mid: f64,
}

/// Parse FFmpeg silencedetect stderr output into silence gaps.
///
/// FFmpeg outputs lines like:
/// `[silencedetect @ 0x...] silence_end: 123.456 | silence_duration: 2.345`
pub fn parse_silence_detect(stderr: &str) -> Vec<SilenceGap> {
    let re = Regex::new(
        r"silence_end:\s*([\d.]+)\s*\|\s*silence_duration:\s*([\d.]+)"
    )
    .expect("valid regex");

    re.captures_iter(stderr)
        .filter_map(|caps| {
            let end: f64 = caps[1].parse().ok()?;
            let duration: f64 = caps[2].parse().ok()?;
            let mid = end - duration / 2.0;
            Some(SilenceGap { end, duration, mid })
        })
        .collect()
}

/// Compute optimal split points from silence gaps.
///
/// Uses a greedy algorithm: walk through the timeline in `target_chunk_sec` steps,
/// and at each step pick the silence gap closest to the target boundary.
///
/// Falls back to fixed-interval splits if no silence gaps are found.
pub fn compute_split_points(
    silences: &[SilenceGap],
    total_duration: f64,
    target_chunk_sec: f64,
    tolerance: f64,
    min_chunk_sec: f64,
) -> Vec<f64> {
    if total_duration <= target_chunk_sec {
        return Vec::new(); // No splitting needed
    }

    if silences.is_empty() {
        // Fallback: fixed-interval splits
        return fixed_interval_splits(total_duration, target_chunk_sec);
    }

    let mut split_points = Vec::new();
    let mut current_start = 0.0;

    loop {
        let target = current_start + target_chunk_sec;
        if target >= total_duration - min_chunk_sec {
            break; // Remaining audio is small enough for one chunk
        }

        // Find the silence gap closest to the target within tolerance
        let window_start = target - tolerance;
        let window_end = target + tolerance;

        let best = silences
            .iter()
            .filter(|s| s.mid >= window_start && s.mid <= window_end)
            .min_by(|a, b| {
                let da = (a.mid - target).abs();
                let db = (b.mid - target).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });

        let split_at = match best {
            Some(gap) => gap.mid,
            None => {
                // No silence in window -- use any silence after current position
                let fallback = silences
                    .iter()
                    .filter(|s| s.mid > current_start + min_chunk_sec && s.mid < total_duration - min_chunk_sec)
                    .min_by(|a, b| {
                        let da = (a.mid - target).abs();
                        let db = (b.mid - target).abs();
                        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                    });

                match fallback {
                    Some(gap) => gap.mid,
                    None => target, // Hard split at target (no silence available)
                }
            }
        };

        // Avoid duplicate or too-close split points
        if split_points.last().is_none_or(|&last: &f64| (split_at - last).abs() > min_chunk_sec) {
            split_points.push(split_at);
        }
        current_start = split_at;
    }

    split_points
}

/// Simple fixed-interval split points when no silence data is available.
fn fixed_interval_splits(total_duration: f64, interval: f64) -> Vec<f64> {
    let mut points = Vec::new();
    let mut t = interval;
    while t < total_duration - interval * 0.5 {
        points.push(t);
        t += interval;
    }
    points
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_silence_detect() {
        let stderr = r#"
[silencedetect @ 0x1234] silence_end: 45.678 | silence_duration: 3.456
[silencedetect @ 0x1234] silence_end: 120.500 | silence_duration: 2.100
some other output
[silencedetect @ 0x1234] silence_end: 300.000 | silence_duration: 5.000
"#;
        let gaps = parse_silence_detect(stderr);
        assert_eq!(gaps.len(), 3);
        assert!((gaps[0].end - 45.678).abs() < 0.001);
        assert!((gaps[0].duration - 3.456).abs() < 0.001);
    }

    #[test]
    fn test_no_split_needed() {
        let points = compute_split_points(&[], 200.0, 300.0, 60.0, 30.0);
        assert!(points.is_empty());
    }

    #[test]
    fn test_fixed_interval_fallback() {
        let points = compute_split_points(&[], 900.0, 300.0, 60.0, 30.0);
        assert_eq!(points.len(), 2); // splits at ~300 and ~600
    }

    #[test]
    fn test_silence_based_split() {
        let silences = vec![
            SilenceGap { end: 310.0, duration: 4.0, mid: 308.0 },
            SilenceGap { end: 610.0, duration: 3.0, mid: 608.5 },
        ];
        let points = compute_split_points(&silences, 900.0, 300.0, 60.0, 30.0);
        assert_eq!(points.len(), 2);
        assert!((points[0] - 308.0).abs() < 0.001);
        assert!((points[1] - 608.5).abs() < 0.001);
    }
}
