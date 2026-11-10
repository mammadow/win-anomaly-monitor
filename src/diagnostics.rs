//! Forensic Root-Cause Analysis (RCA), natural-language Q&A (`ask`),
//! system health auditing (`diagnose`), and optional local/cloud LLM integration.

use crate::config::MonitorConfig;
use crate::events::fmt_duration;
use crate::features::{fmt_bytes, get_metric, nanmedian, percentile, MetricMap};
use crate::sampler::unix_now;
use crate::storage::{EventRecord, Storage};
use chrono::{Local, TimeZone};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::time::Duration;

pub const SYSTEM_PROMPT: &str = "\
You are a senior Windows 11 systems performance engineer analyzing real-time \
telemetry from `resmon` (a kernel/hardware/UI anomaly monitor).

Rules:
1. Ground every claim strictly in the provided JSON telemetry (event details, \
   concurrent 1 Hz samples, 10 s windows, nearby events, and process attribution).
2. Name the specific executable (`process`), PID, script/service tag, window title, \
   or hardware subsystem responsible.
3. Explain the physical mechanism clearly (e.g. UI message-pump starvation vs \
   synchronous I/O wait vs kernel DPC/ISR storm vs hard-fault pagefile thrashing \
   vs Optimus D3cold wake lock vs handle table growth).
4. Finish with 2-3 concrete, prioritized remediation steps the user can take on Windows.";

fn fmt_ts(ts: f64) -> String {
    Local
        .timestamp_opt(ts as i64, 0)
        .single()
        .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_default()
}

fn detail_f64(d: &Value, keys: &[&str]) -> Option<f64> {
    for &k in keys {
        if let Some(v) = d.get(k).and_then(|x| x.as_f64()).filter(|v| v.is_finite()) {
            return Some(v);
        }
    }
    None
}

fn extract_client_names(v: &Value) -> Vec<String> {
    let Some(arr) = v.as_array() else {
        return Vec::new();
    };
    arr.iter()
        .filter_map(|item| {
            if let Some(s) = item.as_str() {
                Some(s.to_string())
            } else {
                item.get("name").and_then(|n| n.as_str()).map(|n| {
                    match item.get("sm_util_pct").and_then(|u| u.as_f64()) {
                        Some(u) if u > 0.0 => format!("{} ({:.0}%)", n, u),
                        _ => n.to_string(),
                    }
                })
            }
        })
        .collect()
}

fn memory_pressure_peak(ev: &EventRecord) -> (f64, f64, f64) {
    let d = &ev.details;
    let ram = detail_f64(d, &["ram_pct", "peak_ram_pct"]).unwrap_or(0.0);
    let commit = detail_f64(d, &["commit_pct", "peak_commit_pct"]).unwrap_or(0.0);
    let mut peak = detail_f64(d, &["peak_pct"]).unwrap_or_else(|| ram.max(commit));
    if peak <= 0.0 {
        // Parse fallback from headline e.g. "RAM 72%, commit 97%"
        for part in ev.title.split('%') {
            let digits: String = part
                .chars()
                .rev()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect();
            if let Ok(v) = digits.parse::<f64>() {
                peak = peak.max(v);
            }
        }
    }
    (ram, commit, peak)
}

pub fn summarise_samples(samples: &[MetricMap], ref_ts: f64) -> Value {
    if samples.is_empty() {
        return json!({ "count": 0 });
    }
    let cols = [
        "cpu_pct",
        "core_max_pct",
        "dpc_pct",
        "isr_pct",
        "dpc_max_core_pct",
        "ram_pct",
        "commit_pct",
        "disk_read_bps",
        "disk_write_bps",
        "net_sent_bps",
        "net_recv_bps",
        "hard_faults_ps",
        "gpu_util_pct",
        "gpu_power_w",
        "gpu_temp_c",
        "gpu_vram_pct",
        "gpu_clock_mhz",
    ];
    let mut metrics = serde_json::Map::new();
    for col in cols {
        let vals: Vec<f64> = samples
            .iter()
            .map(|s| get_metric(s, col))
            .filter(|v| v.is_finite())
            .collect();
        if vals.is_empty() {
            continue;
        }
        let mean = vals.iter().sum::<f64>() / (vals.len() as f64);
        let p95 = percentile(&vals, 95.0);
        let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        metrics.insert(
            col.to_string(),
            json!({
                "mean": (mean * 100.0).round() / 100.0,
                "p95": (p95 * 100.0).round() / 100.0,
                "max": (max * 100.0).round() / 100.0,
            }),
        );
    }

    let mut gpu_transitions = Vec::new();
    let mut prev_awake: Option<bool> = None;
    for s in samples {
        let awake_v = get_metric(s, "gpu_awake");
        if !awake_v.is_finite() {
            continue;
        }
        let cur = awake_v >= 0.5;
        if let Some(prev) = prev_awake {
            if prev != cur {
                let rel = get_metric(s, "ts") - ref_ts;
                gpu_transitions.push(format!(
                    "{} at {:+.1}s",
                    if cur { "woke up" } else { "powered off" },
                    rel
                ));
            }
        }
        prev_awake = Some(cur);
    }

    let t0 = get_metric(&samples[0], "ts");
    let t1 = get_metric(&samples[samples.len() - 1], "ts");
    json!({
        "count": samples.len(),
        "span_sec": ((t1 - t0).max(0.0) * 10.0).round() / 10.0,
        "metrics": Value::Object(metrics),
        "gpu_transitions": gpu_transitions,
    })
}

pub fn summarise_windows(windows: &[MetricMap]) -> Value {
    if windows.is_empty() {
        return json!({ "count": 0 });
    }
    let scores: Vec<f64> = windows
        .iter()
        .map(|w| get_metric(w, "score"))
        .filter(|v| v.is_finite())
        .collect();
    let anom_count = windows
        .iter()
        .filter(|w| get_metric(w, "anomalous") >= 0.5)
        .count();
    let peak_score = if scores.is_empty() {
        Value::Null
    } else {
        let m = scores.iter().copied().fold(0.0f64, f64::max);
        json!((m * 100.0).round() / 100.0)
    };
    let median_score = if scores.is_empty() {
        Value::Null
    } else {
        json!((nanmedian(&scores) * 100.0).round() / 100.0)
    };
    json!({
        "count": windows.len(),
        "anomalous_windows": anom_count,
        "peak_score": peak_score,
        "median_score": median_score,
    })
}

pub fn event_record_to_json(ev: &EventRecord) -> Value {
    let dur = (ev.ts_end - ev.ts).max(0.0);
    json!({
        "id": ev.id,
        "time": fmt_ts(ev.ts),
        "ts": ev.ts,
        "ts_end": ev.ts_end,
        "duration_sec": (dur * 10.0).round() / 10.0,
        "kind": ev.kind,
        "severity": ev.severity,
        "title": ev.title,
        "details": ev.details,
    })
}

/// Assemble a self-contained forensic packet around a specific (or latest) event.
pub fn build_event_context(
    storage: &Storage,
    cfg: &MonitorConfig,
    event_id: Option<i64>,
    kind: Option<&str>,
) -> Option<Value> {
    let ev = if let Some(id) = event_id {
        storage.get_event(id)?
    } else {
        storage.latest_event(kind)?
    };

    let pad = (cfg.ai_context_minutes.max(1) as f64) * 60.0;
    let t0 = ev.ts;
    let t1 = ev.ts_end.max(t0);

    let nearby_raw = storage.query_events(t0 - pad, None, 50);
    let nearby: Vec<Value> = nearby_raw
        .into_iter()
        .filter(|e| e.id != ev.id && e.ts <= t1 + pad)
        .take(15)
        .map(|e| {
            json!({
                "id": e.id,
                "time": fmt_ts(e.ts),
                "offset_sec": ((e.ts - t0) * 10.0).round() / 10.0,
                "duration_sec": (((e.ts_end - e.ts).max(0.0)) * 10.0).round() / 10.0,
                "kind": e.kind,
                "severity": e.severity,
                "title": e.title,
            })
        })
        .collect();

    let samples = storage.query_samples(t0 - 120.0, t1 + 60.0, 300);
    let windows = storage.query_windows_range(t0 - 600.0, t1 + 120.0);

    let mut live_state = Value::Null;
    let state_path = cfg.state_path();
    if state_path.exists() && (unix_now() - t1).abs() < 1800.0 {
        if let Ok(txt) = std::fs::read_to_string(&state_path) {
            if let Ok(st) = serde_json::from_str::<Value>(&txt) {
                live_state = json!({
                    "summary": st.get("summary").cloned().unwrap_or(Value::Null),
                    "activity": st.get("activity").cloned().unwrap_or(Value::Null),
                    "responsiveness": st.get("responsiveness").cloned().unwrap_or(Value::Null),
                });
            }
        }
    }

    Some(json!({
        "event": event_record_to_json(&ev),
        "nearby_events": nearby,
        "sample_summary": summarise_samples(&samples, t0),
        "window_summary": summarise_windows(&windows),
        "live_state": live_state,
    }))
}

/// Deterministic offline root-cause analyzer covering all 10 event kinds.
pub fn offline_explain_event(ctx: &Value, question: Option<&str>) -> String {
    let ev = &ctx["event"];
    let d = &ev["details"];
    let kind = ev["kind"].as_str().unwrap_or("unknown");
    let sev = ev["severity"].as_str().unwrap_or("info").to_uppercase();
    let dur = ev["duration_sec"].as_f64().unwrap_or(0.0);
    let samples = &ctx["sample_summary"]["metrics"];
    let nearby = ctx["nearby_events"].as_array().cloned().unwrap_or_default();

    let mut lines = vec![
        format!(
            "Event #{} — {} [{}] at {} (duration: {})",
            ev["id"].as_i64().unwrap_or(0),
            kind,
            sev,
            ev["time"].as_str().unwrap_or(""),
            fmt_duration(dur)
        ),
        format!("Headline: {}", ev["title"].as_str().unwrap_or("")),
        String::new(),
    ];

    let mut root_cause = Vec::new();
    let mut evidence = Vec::new();
    let mut actions = Vec::new();

    match kind {
        "ui_freeze" => {
            let src_raw = d
                .get("source")
                .or_else(|| d.get("scope"))
                .and_then(|v| v.as_str())
                .unwrap_or("foreground");
            let src = match src_raw {
                "desktop" => "taskbar",
                "display" => "compositor",
                other => other,
            };
            let proc_name = d["process"].as_str().unwrap_or("unknown process");
            let pid = d["pid"].as_u64().unwrap_or(0);
            let peak_ms = detail_f64(d, &["max_latency_ms"])
                .or_else(|| detail_f64(d, &["longest_stall_sec"]).map(|s| s * 1000.0))
                .unwrap_or(dur * 1000.0);

            let has_dgpu_wake = d["dgpu_transitions"]
                .as_array()
                .map(|arr| {
                    arr.iter().any(|t| {
                        t["dgpu"]
                            .as_str()
                            .map(|s| s.contains("woke"))
                            .unwrap_or(false)
                    })
                })
                .unwrap_or(false)
                || ctx["sample_summary"]["gpu_transitions"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .any(|t| t.as_str().map(|s| s.contains("woke")).unwrap_or(false))
                    })
                    .unwrap_or(false);

            let around = &d["around"];
            let conc = &d["concurrent"];
            let cause = d["cause"].as_str().unwrap_or_else(|| {
                if has_dgpu_wake {
                    "dgpu_wake_latency"
                } else if d["dpc_isr_spike"].as_bool().unwrap_or(false) {
                    "dpc_storm"
                } else if around["monitor_loop_gap_sec"].as_f64().unwrap_or(0.0) >= 2.0 {
                    "scheduler_starvation"
                } else if around["commit_max_pct"].as_f64().unwrap_or(0.0) >= 90.0 {
                    "memory_paging"
                } else {
                    "unclassified"
                }
            });
            let v3_clues: Vec<&str> = d["clues"]
                .as_array()
                .map(|arr| arr.iter().filter_map(|c| c.as_str()).collect())
                .unwrap_or_default();
            let cause_exp = d["cause_explanation"]
                .as_str()
                .map(|s| s.to_string())
                .unwrap_or_else(|| {
                    if !v3_clues.is_empty() {
                        v3_clues.join("; ")
                    } else if has_dgpu_wake {
                        "NVIDIA discrete GPU PCIe D3cold -> D0 wake transition stalled message/display path".to_string()
                    } else {
                        "transient UI message-pump stall".to_string()
                    }
                });

            let target_desc = match src {
                "taskbar" => "Windows Shell (`explorer.exe` taskbar)".to_string(),
                "compositor" => "Desktop Window Manager (`dwm.exe`)".to_string(),
                "scheduler" => "OS Kernel Thread Scheduler".to_string(),
                _ => format!("`{}` (PID {})", proc_name, pid),
            };
            root_cause.push(format!(
                "{} stopped pumping window messages for {:.0} ms (total episode {}).",
                target_desc,
                peak_ms,
                fmt_duration(dur)
            ));
            root_cause.push(format!(
                "Primary mechanism classified as `{}`: {}.",
                cause, cause_exp
            ));

            let app_cores_opt = conc["stalled_proc_cores"]
                .as_f64()
                .or_else(|| d["app_cpu_sec"].as_f64().map(|c| c / dur.max(0.25)));
            if let Some(sc) = app_cores_opt {
                if sc >= 0.7 {
                    evidence.push(format!(
                        "`{}` consumed {:.2} CPU cores during the stall — its UI thread was executing a tight synchronous loop.",
                        proc_name, sc
                    ));
                } else if sc < 0.05 {
                    evidence.push(format!(
                        "`{}` used ~0 CPU during the stall — its UI thread was blocked waiting on an OS lock, synchronous disk/network I/O, or cross-process COM/IPC call.",
                        proc_name
                    ));
                }
            }
            if conc.is_object() {
                let hf = conc["hard_faults_ps"].as_f64().unwrap_or(0.0);
                let hf_str = if hf >= 10.0 {
                    format!(", Hard faults {:.0}/s", hf)
                } else {
                    String::new()
                };
                evidence.push(format!(
                    "Concurrent system telemetry: CPU {:.0}% (busiest core {:.0}%), DPC {:.1}% + ISR {:.1}%, RAM {:.0}%, Commit {:.0}%{}, Disk {:.1}R/{:.1}W MB/s, GPU {:.0}%.",
                    conc["cpu_pct"].as_f64().unwrap_or(0.0),
                    conc["core_max_pct"].as_f64().unwrap_or(0.0),
                    conc["dpc_pct"].as_f64().unwrap_or(0.0),
                    conc["isr_pct"].as_f64().unwrap_or(0.0),
                    conc["ram_pct"].as_f64().unwrap_or(0.0),
                    conc["commit_pct"].as_f64().unwrap_or(0.0),
                    hf_str,
                    conc["disk_read_mb_s"].as_f64().unwrap_or(0.0),
                    conc["disk_write_mb_s"].as_f64().unwrap_or(0.0),
                    conc["gpu_util_pct"].as_f64().unwrap_or(0.0),
                ));
            } else if around.is_object() {
                evidence.push(format!(
                    "Concurrent system telemetry: CPU max {:.0}% (busiest core {:.0}%), DPC {:.2}% + ISR {:.2}%, RAM {:.0}%, Commit {:.0}%, Disk {} R / {} W, GPU {:.0}%.",
                    around["cpu_max_pct"].as_f64().unwrap_or(0.0),
                    around["busiest_core_pct"].as_f64().unwrap_or(0.0),
                    around["dpc_max_pct"].as_f64().unwrap_or(0.0),
                    around["isr_max_pct"].as_f64().unwrap_or(0.0),
                    around["ram_max_pct"].as_f64().unwrap_or(0.0),
                    around["commit_max_pct"].as_f64().unwrap_or(0.0),
                    around["disk_read_max"].as_str().unwrap_or("0 B/s"),
                    around["disk_write_max"].as_str().unwrap_or("0 B/s"),
                    around["gpu_util_max_pct"].as_f64().unwrap_or(0.0),
                ));
            }
            match cause {
                "dgpu_wake_latency" => {
                    actions.push(format!(
                        "Set `{}` to 'Power saving' (integrated GPU) in Windows Settings -> System -> Display -> Graphics if it does not require dedicated 3D acceleration.",
                        proc_name
                    ));
                }
                "dpc_storm" | "scheduler_starvation" => {
                    actions.push("Update network (Wi-Fi/Ethernet), GPU, and storage NVMe drivers; high DPC/ISR latency preempts user-mode threads.".to_string());
                    actions.push("Disable PCIe Link State Power Management or USB selective suspend if stalls correlate with device wakeups.".to_string());
                }
                "memory_paging" => {
                    actions.push("Close or restart top memory-consuming processes so Windows stops paging active UI working sets to disk.".to_string());
                }
                "disk_io_stall" => {
                    actions.push("Check the top I/O process in `concurrent.activity.top_io`; synchronous flushes on the system drive stall UI threads waiting on file handles.".to_string());
                }
                "gpu_compositor_stall" => {
                    actions.push("Check GPU driver stability or press Win+Ctrl+Shift+B to reset the display driver if DWM stalls recur.".to_string());
                }
                _ => {
                    actions.push(format!(
                        "If `{}` freezes repeatedly, check its extensions/plugins or add it to `ignore_processes` in `config.toml` if expected during heavy loads.",
                        proc_name
                    ));
                }
            }
        }
        "dpc_storm" => {
            let peak = d["peak_pct"].as_f64().unwrap_or(0.0);
            let core = d["dpc_max_core_pct"].as_f64().unwrap_or(0.0);
            root_cause.push(format!(
                "Kernel Deferred Procedure Calls (DPC) + Interrupt Service Routines (ISR) consumed {:.1}% of total CPU ({:.1}% on the busiest core).",
                peak, core
            ));
            root_cause.push("Kernel-mode interrupt handlers run above all normal thread priorities, starving mouse input, audio buffers, and window message queues.".to_string());
            if let Some(drivers) = d["loaded_third_party_drivers"].as_array() {
                let drv_list: Vec<&str> = drivers.iter().filter_map(|v| v.as_str()).collect();
                if !drv_list.is_empty() {
                    evidence.push(format!(
                        "Loaded non-inbox kernel drivers (`.sys`): {}.",
                        drv_list.join(", ")
                    ));
                }
            }
            actions.push(
                "Update network adapter, GPU (`nvlddmkm.sys`), and chipset drivers.".to_string(),
            );
            actions.push("If on Wi-Fi or USB Docks, test whether unplugging the dock or switching power plans stops the DPC spikes.".to_string());
        }
        "memory_pressure" => {
            let mut ram = detail_f64(d, &["ram_pct", "peak_ram_pct"])
                .or_else(|| samples["ram_pct"]["max"].as_f64())
                .unwrap_or(0.0);
            let mut commit = detail_f64(d, &["commit_pct", "peak_commit_pct"])
                .or_else(|| samples["commit_pct"]["max"].as_f64())
                .unwrap_or(0.0);
            if ram == 0.0 && commit == 0.0 {
                let title = ev["title"].as_str().unwrap_or("");
                if let Some(r_idx) = title.find("RAM ") {
                    if let Ok(v) = title[r_idx + 4..]
                        .split('%')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .parse::<f64>()
                    {
                        ram = v;
                    }
                }
                if let Some(c_idx) = title.find("commit ") {
                    if let Ok(v) = title[c_idx + 7..]
                        .split('%')
                        .next()
                        .unwrap_or("")
                        .trim()
                        .parse::<f64>()
                    {
                        commit = v;
                    }
                }
            }
            let peak = detail_f64(d, &["peak_pct"]).unwrap_or_else(|| ram.max(commit));
            let hf = detail_f64(d, &["peak_hard_faults_ps"])
                .or_else(|| samples["hard_faults_ps"]["max"].as_f64())
                .unwrap_or(0.0);
            root_cause.push(format!(
                "System physical RAM ({:.1}%) or virtual commit charge ({:.1}%, peak {:.1}%) crossed safety thresholds for {}.",
                ram, commit, peak, fmt_duration(dur)
            ));
            if hf >= 50.0 {
                root_cause.push(format!(
                    "Swap thrashing confirmed: system experienced {:.0} hard page faults/sec while paging memory to/from disk.",
                    hf
                ));
            }
            if let Some(top_mem) = d["top_memory"].as_array() {
                let fmt_list: Vec<String> = top_mem
                    .iter()
                    .take(5)
                    .map(|p| {
                        let extra = if let Some(h) = p.get("handles").and_then(|v| v.as_i64()) {
                            format!(", {} handles", h)
                        } else {
                            String::new()
                        };
                        format!(
                            "{} ({} MB across {} proc{})",
                            p["name"].as_str().unwrap_or("?"),
                            p["private_mb"].as_i64().unwrap_or(0),
                            p["processes"].as_i64().unwrap_or(1),
                            extra
                        )
                    })
                    .collect();
                if !fmt_list.is_empty() {
                    evidence.push(format!("Top memory consumers: {}.", fmt_list.join("; ")));
                }
            }
            actions.push(
                "Close or cap worker counts for the largest memory consumers listed above."
                    .to_string(),
            );
            actions.push("Ensure Windows pagefile is system-managed on a fast NVMe drive with at least 15 GB free space.".to_string());
        }
        "anomaly" => {
            let score = d["peak_score"].as_f64().unwrap_or(0.0);
            if let Some(rarity) = d["rarity_pct"].as_f64() {
                root_cause.push(format!(
                    "Adaptive kNN detector flagged a resource regime rarer than {:.2}% of your historical baseline (peak LOF score {:.2}).",
                    rarity, score
                ));
            } else if let Some(r_str) = d["rarity"].as_str() {
                root_cause.push(format!(
                    "Adaptive kNN detector flagged a resource regime ({}, peak score {:.2}).",
                    r_str, score
                ));
            } else {
                root_cause.push(format!(
                    "Adaptive kNN detector flagged an unusual resource regime (peak LOF score {:.2}).",
                    score
                ));
            }
            let feats_arr = d
                .get("deviated_features")
                .or_else(|| d.get(" deviated_features"))
                .or_else(|| d.get("why"))
                .and_then(|v| v.as_array());
            if let Some(feats) = feats_arr {
                for f in feats {
                    if let Some(txt) = f["text"].as_str() {
                        let units = f["units_off"].as_f64().unwrap_or(0.0);
                        if units > 0.0 {
                            evidence.push(format!(
                                "Deviated metric: {} (+{} meaningful units above similar historical moments).",
                                txt, units
                            ));
                        } else {
                            evidence.push(format!("Deviated metric: {}.", txt));
                        }
                    }
                }
            }
            let top_cpu_arr = d["activity"]["top_cpu"]
                .as_array()
                .or_else(|| d["top_cpu"].as_array());
            if let Some(top_cpu) = top_cpu_arr {
                if !top_cpu.is_empty() {
                    let s: Vec<String> = top_cpu
                        .iter()
                        .take(3)
                        .map(|c| {
                            let cores = detail_f64(c, &["cores", "avg_cores"]).unwrap_or(0.0);
                            format!(
                                "{} ({:.1} cores, {} MB)",
                                c["name"].as_str().unwrap_or("?"),
                                cores,
                                c["private_mb"].as_i64().unwrap_or(0)
                            )
                        })
                        .collect();
                    evidence.push(format!("Concurrent top CPU processes: {}.", s.join(", ")));
                }
            }
            let top_io_arr = d["activity"]["top_io"]
                .as_array()
                .or_else(|| d["top_io"].as_array());
            if let Some(top_io) = top_io_arr {
                if !top_io.is_empty() {
                    let s: Vec<String> = top_io
                        .iter()
                        .take(3)
                        .map(|c| {
                            if let Some(avg_mb) = c.get("avg_mb_s").and_then(|v| v.as_f64()) {
                                format!(
                                    "{} ({:.1} MB/s)",
                                    c["name"].as_str().unwrap_or("?"),
                                    avg_mb
                                )
                            } else {
                                format!(
                                    "{} ({:.1}R/{:.1}W MB/s)",
                                    c["name"].as_str().unwrap_or("?"),
                                    c["read_mb_s"].as_f64().unwrap_or(0.0),
                                    c["write_mb_s"].as_f64().unwrap_or(0.0)
                                )
                            }
                        })
                        .collect();
                    evidence.push(format!("Concurrent top I/O processes: {}.", s.join(", ")));
                }
            }
            actions.push("Inspect the top CPU/I/O processes above to verify whether this workload was intentional.".to_string());
            actions.push("Run `resmon tune` if you want to adjust `alert_quantile` or `min_consecutive_windows`.".to_string());
        }
        "memory_leak" | "handle_leak" => {
            let proc_name = d["process"].as_str().unwrap_or("unknown");
            let pid = d["pid"].as_u64().unwrap_or(0);
            let hrs = d["hours_observed"].as_f64().unwrap_or(0.0);
            if kind == "memory_leak" {
                root_cause.push(format!(
                    "`{}` (PID {}) exhibited monotonic private memory growth (+{} MB over {} h at {} MB/h, R²={}, now {} MB).",
                    proc_name,
                    pid,
                    d["growth_mb"].as_i64().unwrap_or(0),
                    hrs,
                    d["mb_per_hour"].as_f64().unwrap_or(0.0),
                    d["r2"].as_f64().unwrap_or(0.0),
                    d["private_mb"].as_i64().unwrap_or(0)
                ));
            } else {
                let rate = d["handles_per_hour"].as_i64();
                let rate_str = rate.map(|r| format!(" (+{}/h)", r)).unwrap_or_default();
                root_cause.push(format!(
                    "`{}` (PID {}) holds {} kernel handles{} after {} h of observation.",
                    proc_name,
                    pid,
                    d["handles"].as_i64().unwrap_or(0),
                    rate_str,
                    hrs
                ));
                root_cause.push("Unclosed kernel handles exhaust paged/non-paged kernel pool memory and degrade system-wide handle table lookups.".to_string());
            }
            actions.push(format!(
                "Restart `{}` (or its Windows service) to immediately reclaim leaked memory/handles.",
                proc_name
            ));
            actions.push(format!(
                "Check for updates to `{}` or disable unnecessary OEM background services if it is vendor bloatware.",
                proc_name
            ));
        }
        "gpu_thermal" | "gpu_battery_drain" | "gpu_awake_on_battery" => {
            let clients = extract_client_names(&d["gpu_clients"]);
            let c_str = if clients.is_empty() {
                "unknown process".to_string()
            } else {
                clients.join(", ")
            };
            let pwr = detail_f64(d, &["gpu_power_w", "peak_power_w", "avg_power_w"])
                .or_else(|| samples["gpu_power_w"]["mean"].as_f64())
                .unwrap_or(0.0);
            if kind == "gpu_thermal" {
                root_cause.push(format!(
                    "Discrete GPU reached {}°C drawing {:.1} W @ {} MHz (active clients: {}).",
                    d["peak_temp_c"].as_f64().unwrap_or(0.0),
                    pwr,
                    d["gpu_clock_mhz"].as_f64().unwrap_or(0.0),
                    c_str
                ));
                actions.push(
                    "Ensure laptop vents are unobstructed and fan curves are functioning."
                        .to_string(),
                );
            } else {
                root_cause.push(format!(
                    "Discrete GPU was held awake in P-state on battery power (~{:.1} W idle drain) by `{}` despite <5% GPU utilization.",
                    pwr,
                    c_str
                ));
                actions.push(format!(
                    "Close `{}` on battery or assign it to the Integrated GPU in Windows Settings -> System -> Display -> Graphics.",
                    c_str
                ));
            }
        }
        "background_cpu" => {
            let proc_name = d["process"].as_str().unwrap_or("unknown");
            let pid = d["pid"].as_u64().unwrap_or(0);
            let cores = d["avg_cores"].as_f64().unwrap_or(0.0);
            root_cause.push(format!(
                "`{}` (PID {}) consumed an average of {:.2} CPU cores continuously in the background for {} without an active foreground window.",
                proc_name,
                pid,
                cores,
                fmt_duration(dur)
            ));
            actions.push(format!(
                "Inspect `{}` (PID {}) in `resmon top` or Task Manager; terminate it if stuck or add it to `ignore_processes` in `config.toml` if expected.",
                proc_name, pid
            ));
        }
        "upload_burst" => {
            let total_mb = detail_f64(d, &["total_mb", "sent_mb"]).unwrap_or(0.0);
            let peak_bps = detail_f64(d, &["peak_bps"])
                .or_else(|| detail_f64(d, &["peak_mb_s"]).map(|m| m * 1_000_000.0))
                .or_else(|| samples["net_sent_bps"]["max"].as_f64())
                .unwrap_or(0.0);
            root_cause.push(format!(
                "Sustained outbound network transfer (~{:.1} MB total, peak {}) lasted {}.",
                total_mb,
                fmt_bytes(peak_bps, true),
                fmt_duration(dur)
            ));
            let mut top_name = String::new();
            if let Some(top_net) = d["top_net"].as_array() {
                let s: Vec<String> = top_net
                    .iter()
                    .take(3)
                    .map(|p| {
                        let nm = p["name"].as_str().unwrap_or("?");
                        if top_name.is_empty() {
                            top_name = nm.to_string();
                        }
                        format!(
                            "{} (≈{:.1} MB net/other I/O)",
                            nm,
                            p["total_mb"].as_f64().unwrap_or(0.0)
                        )
                    })
                    .collect();
                if !s.is_empty() {
                    evidence.push(format!("Top network I/O processes: {}.", s.join(", ")));
                }
            } else if let Some(top_io) = d["activity"]["top_io"].as_array() {
                if let Some(p0) = top_io.first() {
                    let nm = p0["name"].as_str().unwrap_or("?");
                    top_name = nm.to_string();
                    evidence.push(format!("Top concurrent I/O process: `{}`.", nm));
                }
            }
            if let Some(fg) = d["foreground"]
                .as_str()
                .or_else(|| d["activity"]["foreground"].as_str())
            {
                let driven = d["user_driven"].as_bool().unwrap_or(false);
                evidence.push(format!(
                    "Foreground app: `{}` ({}).",
                    fg,
                    if driven {
                        "user-driven upload"
                    } else {
                        "background transfer"
                    }
                ));
            }
            if !top_name.is_empty() {
                actions.push(format!(
                    "Verify whether `{}` was performing an intentional sync/upload; add to `ignore_processes` in `config.toml` if expected.",
                    top_name
                ));
            } else {
                actions.push("Review concurrent processes in event details to verify whether cloud sync, backup, or telemetry upload was expected.".to_string());
            }
        }
        "low_disk_space" => {
            let free_gb = detail_f64(d, &["disk_free_gb", "free_gb"]).unwrap_or(0.0);
            root_cause.push(format!(
                "System drive free space dropped to {:.1} GB, risking pagefile expansion failures and SSD write amplification.",
                free_gb
            ));
            actions.push("Clear temporary build artifacts, temp folders, or old downloads on the system drive.".to_string());
        }
        _ => {
            root_cause.push(format!(
                "Event recorded: {}",
                ev["title"].as_str().unwrap_or("")
            ));
        }
    }

    if samples.is_object() {
        if let Some(cpu_m) = samples.get("cpu_pct") {
            let dpc_max = samples
                .get("dpc_pct")
                .and_then(|x| x.get("max"))
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            let commit_max = samples
                .get("commit_pct")
                .and_then(|x| x.get("max"))
                .and_then(|v| v.as_f64());
            let commit_part = commit_max
                .map(|c| format!(", Commit max {:.1}%", c))
                .unwrap_or_default();
            evidence.push(format!(
                "Window sample stats ({}s around event): CPU mean {:.1}% / max {:.1}%, RAM max {:.1}%{}, DPC max {:.1}%.",
                ctx["sample_summary"]["span_sec"].as_f64().unwrap_or(0.0).round(),
                cpu_m["mean"].as_f64().unwrap_or(0.0),
                cpu_m["max"].as_f64().unwrap_or(0.0),
                samples["ram_pct"]["max"].as_f64().unwrap_or(0.0),
                commit_part,
                dpc_max
            ));
        }
    }
    if let Some(trans) = ctx["sample_summary"]["gpu_transitions"].as_array() {
        if !trans.is_empty() {
            let s: Vec<&str> = trans.iter().filter_map(|v| v.as_str()).take(4).collect();
            if !s.is_empty() {
                evidence.push(format!(
                    "dGPU power transitions near event: {}.",
                    s.join(", ")
                ));
            }
        }
    }

    if !nearby.is_empty() {
        let corr: Vec<String> = nearby
            .iter()
            .take(4)
            .map(|e| {
                format!(
                    "#{} {} ({:+.0}s)",
                    e["id"].as_i64().unwrap_or(0),
                    e["kind"].as_str().unwrap_or(""),
                    e["offset_sec"].as_f64().unwrap_or(0.0)
                )
            })
            .collect();
        evidence.push(format!("Correlated nearby events: {}.", corr.join(", ")));
    }

    lines.push("Root Cause Analysis:".to_string());
    for rc in root_cause {
        lines.push(format!("  • {}", rc));
    }
    if !evidence.is_empty() {
        lines.push(String::new());
        lines.push("Supporting Telemetry Evidence:".to_string());
        for item in evidence {
            lines.push(format!("  • {}", item));
        }
    }
    if !actions.is_empty() {
        lines.push(String::new());
        lines.push("Recommended Actions:".to_string());
        for (i, act) in actions.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, act));
        }
    }
    if let Some(q) = question {
        if !q.trim().is_empty() {
            lines.push(String::new());
            lines.push(format!("Question Focus (\"{}\"):", q));
            lines.push("  • Evaluated using deterministic offline telemetry analyzer (configure Ollama or `OPENAI_API_KEY` for conversational LLM synthesis).".to_string());
        }
    }
    lines.join("\n")
}

/// Call Ollama (`/api/chat`) or OpenAI-compatible (`/v1/chat/completions`) endpoint.
pub fn call_llm(
    cfg: &MonitorConfig,
    system_prompt: &str,
    user_prompt: &str,
) -> Option<(String, String)> {
    let provider = cfg.ai_provider.to_ascii_lowercase();
    if provider == "offline" {
        return None;
    }
    let timeout = Duration::from_secs_f64(cfg.ai_timeout_sec.max(1.0));
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout(timeout)
        .build();

    if provider == "auto" || provider == "ollama" {
        let base = if cfg.ai_endpoint.is_empty() {
            "http://127.0.0.1:11434"
        } else {
            cfg.ai_endpoint.trim_end_matches('/')
        };
        let model = if !cfg.ai_model.is_empty() {
            cfg.ai_model.clone()
        } else {
            detect_ollama_model(&agent, base)?
        };
        let body = json!({
            "model": model,
            "stream": false,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt },
            ],
            "options": { "temperature": 0.2 },
        });
        if let Ok(resp) = agent.post(&format!("{}/api/chat", base)).send_json(body) {
            if let Ok(val) = resp.into_json::<Value>() {
                if let Some(text) = val["message"]["content"].as_str() {
                    if !text.trim().is_empty() {
                        return Some((text.trim().to_string(), format!("ollama:{}", model)));
                    }
                }
            }
        }
        if provider == "ollama" {
            return None;
        }
    }

    if provider == "auto" || provider == "openai" {
        let api_key = std::env::var(&cfg.ai_api_key_env).unwrap_or_default();
        if api_key.is_empty() && cfg.ai_endpoint.is_empty() {
            return None;
        }
        let base = if cfg.ai_endpoint.is_empty() {
            "https://api.openai.com/v1"
        } else {
            cfg.ai_endpoint.trim_end_matches('/')
        };
        let model = if cfg.ai_model.is_empty() {
            "gpt-4o-mini"
        } else {
            cfg.ai_model.as_str()
        };
        let body = json!({
            "model": model,
            "temperature": 0.2,
            "messages": [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt },
            ],
        });
        let mut req = agent.post(&format!("{}/chat/completions", base));
        if !api_key.is_empty() {
            req = req.set("Authorization", &format!("Bearer {}", api_key));
        }
        if let Ok(resp) = req.send_json(body) {
            if let Ok(val) = resp.into_json::<Value>() {
                if let Some(text) = val["choices"][0]["message"]["content"].as_str() {
                    if !text.trim().is_empty() {
                        return Some((text.trim().to_string(), format!("openai:{}", model)));
                    }
                }
            }
        }
    }
    None
}

fn detect_ollama_model(agent: &ureq::Agent, base: &str) -> Option<String> {
    let resp = agent
        .get(&format!("{}/api/tags", base))
        .timeout(Duration::from_millis(1500))
        .call()
        .ok()?;
    let val: Value = resp.into_json().ok()?;
    let models = val["models"].as_array()?;
    models.first()?["name"].as_str().map(|s| s.to_string())
}

pub fn explain_event(
    storage: &Storage,
    cfg: &MonitorConfig,
    event_id: Option<i64>,
    kind: Option<&str>,
    question: Option<&str>,
    force_offline: bool,
) -> Option<Value> {
    let ctx = build_event_context(storage, cfg, event_id, kind)?;
    let mut user_prompt = format!(
        "Analyze this Windows resource monitor event and its surrounding telemetry:\n\n{}",
        serde_json::to_string_pretty(&ctx).unwrap_or_default()
    );
    if let Some(q) = question {
        if !q.trim().is_empty() {
            user_prompt.push_str(&format!("\n\nUser question: {}", q));
        }
    }

    if !force_offline {
        if let Some((answer, engine)) = call_llm(cfg, SYSTEM_PROMPT, &user_prompt) {
            return Some(json!({
                "engine": engine,
                "explanation": answer,
                "context": ctx,
            }));
        }
    }
    let explanation = offline_explain_event(&ctx, question);
    Some(json!({
        "engine": "offline-rca",
        "explanation": explanation,
        "context": ctx,
    }))
}

pub fn match_question_events(events: &[EventRecord], question: &str) -> Vec<EventRecord> {
    let q = question.to_ascii_lowercase();
    let keyword_map: &[(&str, &[&str])] = &[
        (
            "ui_freeze",
            &[
                "freeze",
                "froze",
                "stutter",
                "lag",
                "stall",
                "hung",
                "not responding",
                "ui",
                "mouse",
                "taskbar",
                "dwm",
            ],
        ),
        ("dpc_storm", &["dpc", "isr", "interrupt", "driver", "audio"]),
        (
            "memory_pressure",
            &[
                "ram", "memory", "commit", "pagefile", "paging", "oom", "swap",
            ],
        ),
        ("memory_leak", &["leak", "growing memory"]),
        ("handle_leak", &["handle", "handles", "acer"]),
        (
            "gpu_thermal",
            &["hot", "temp", "thermal", "overheat", "gpu"],
        ),
        (
            "gpu_battery_drain",
            &["battery", "drain", "optimus", "awake", "power"],
        ),
        (
            "gpu_awake_on_battery",
            &["battery", "drain", "optimus", "awake", "power"],
        ),
        (
            "upload_burst",
            &["upload", "network", "bandwidth", "net", "sent"],
        ),
        (
            "low_disk_space",
            &["disk", "space", "storage", "drive", "free"],
        ),
        ("background_cpu", &["background", "cpu", "fan", "spinning"]),
        (
            "anomaly",
            &["unusual", "anomaly", "strange", "weird", "spike"],
        ),
    ];

    let mut matched_kinds = Vec::new();
    for (k, words) in keyword_map {
        if words.iter().any(|w| q.contains(w)) {
            matched_kinds.push(*k);
        }
    }
    if matched_kinds.is_empty() {
        return events.iter().take(10).cloned().collect();
    }
    let filtered: Vec<EventRecord> = events
        .iter()
        .filter(|e| matched_kinds.contains(&e.kind.as_str()))
        .cloned()
        .collect();
    if filtered.is_empty() {
        events.iter().take(10).cloned().collect()
    } else {
        filtered.into_iter().take(15).collect()
    }
}

pub fn ask_question(
    storage: &Storage,
    cfg: &MonitorConfig,
    question: &str,
    hours: f64,
    force_offline: bool,
) -> Value {
    let since = unix_now() - hours * 3600.0;
    let events = storage.query_events(since, None, 200);
    let matched = match_question_events(&events, question);

    let top_ctx = matched
        .first()
        .and_then(|e| build_event_context(storage, cfg, Some(e.id), None));

    let matched_json: Vec<Value> = matched.iter().take(10).map(event_record_to_json).collect();
    let payload = json!({
        "question": question,
        "lookback_hours": hours,
        "total_events_in_window": events.len(),
        "matching_events": matched_json,
        "most_recent_match_context": top_ctx,
    });

    if !force_offline {
        let user_prompt = format!(
            "Answer the user's question about their Windows PC using this telemetry:\n\n{}\n\nQuestion: {}",
            serde_json::to_string_pretty(&payload).unwrap_or_default(),
            question
        );
        if let Some((answer, engine)) = call_llm(cfg, SYSTEM_PROMPT, &user_prompt) {
            return json!({
                "engine": engine,
                "answer": answer,
                "matched_events": matched.len(),
            });
        }
    }

    if matched.is_empty() {
        return json!({
            "engine": "offline-rca",
            "answer": format!("No events recorded in the last {:.1} hours matching \"{}\".", hours, question),
            "matched_events": 0,
        });
    }

    let mut lines = vec![format!(
        "Found {} relevant event(s) in the last {:.1} hours for \"{}\":",
        matched.len(),
        hours,
        question
    )];
    for e in matched.iter().take(5) {
        lines.push(format!(
            "  • #{} [{}] {} ({}): {}",
            e.id,
            e.severity,
            fmt_ts(e.ts),
            e.kind,
            e.title
        ));
    }
    if let Some(ref tc) = top_ctx {
        lines.push(String::new());
        lines.push("--- Deep Dive on Most Recent Matching Event ---".to_string());
        lines.push(offline_explain_event(tc, Some(question)));
    }
    json!({
        "engine": "offline-rca",
        "answer": lines.join("\n"),
        "matched_events": matched.len(),
    })
}

/// Holistic health audit across the last `hours` of recorded telemetry & events.
pub fn audit_system_health(storage: &Storage, cfg: &MonitorConfig, hours: f64) -> Value {
    let now = unix_now();
    let since = now - hours * 3600.0;
    let events = storage.query_events(since, None, 500);

    let mut by_kind: HashMap<String, Vec<&EventRecord>> = HashMap::new();
    let mut crit_count = 0usize;
    for e in &events {
        by_kind.entry(e.kind.clone()).or_default().push(e);
        if e.severity == "critical" {
            crit_count += 1;
        }
    }

    let mut findings = Vec::new();
    let mut actions = Vec::new();

    if let Some(hleaks) = by_kind.get("handle_leak") {
        let mut by_proc: HashMap<String, &EventRecord> = HashMap::new();
        for e in hleaks {
            let p = e
                .details
                .get("process")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let replace = match by_proc.get(&p) {
                None => true,
                Some(prev) => {
                    e.details["handles"].as_i64().unwrap_or(0)
                        > prev.details["handles"].as_i64().unwrap_or(0)
                }
            };
            if replace {
                by_proc.insert(p, e);
            }
        }
        for (proc_name, e) in by_proc {
            let h = e.details["handles"].as_i64().unwrap_or(0);
            let rate = e.details["handles_per_hour"].as_i64();
            let rate_str = rate.map(|r| format!(" (+{}/h)", r)).unwrap_or_default();
            findings.push(json!({
                "severity": if h >= 100_000 { "critical" } else { "warning" },
                "category": "handle_leak",
                "summary": format!("`{}` is leaking kernel handles: {} open handles{}", proc_name, h, rate_str),
                "event_id": e.id,
            }));
            actions.push(format!(
                "Restart or disable `{}` (currently holding {} handles) to prevent kernel pool exhaustion.",
                proc_name, h
            ));
        }
    }

    if let Some(mleaks) = by_kind.get("memory_leak") {
        let mut by_proc: HashMap<String, &EventRecord> = HashMap::new();
        for e in mleaks {
            let p = e
                .details
                .get("process")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            by_proc.entry(p).or_insert(e);
        }
        for (proc_name, e) in by_proc {
            findings.push(json!({
                "severity": "warning",
                "category": "memory_leak",
                "summary": format!(
                    "`{}` grew by +{} MB ({} MB/h), reaching {} MB",
                    proc_name,
                    e.details["growth_mb"].as_i64().unwrap_or(0),
                    e.details["mb_per_hour"].as_f64().unwrap_or(0.0),
                    e.details["private_mb"].as_i64().unwrap_or(0)
                ),
                "event_id": e.id,
            }));
            actions.push(format!(
                "Recycle `{}` periodically or inspect its memory growth.",
                proc_name
            ));
        }
    }

    if let Some(mp) = by_kind.get("memory_pressure") {
        let total_dur: f64 = mp.iter().map(|e| (e.ts_end - e.ts).max(0.0)).sum();
        let worst = mp
            .iter()
            .max_by(|a, b| {
                memory_pressure_peak(a)
                    .2
                    .partial_cmp(&memory_pressure_peak(b).2)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        let top_hogs: Vec<String> = worst.details["top_memory"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .take(3)
                    .map(|p| {
                        format!(
                            "{} ({} MB)",
                            p["name"].as_str().unwrap_or("?"),
                            p["private_mb"].as_i64().unwrap_or(0)
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        let hogs_str = if top_hogs.is_empty() {
            "n/a".to_string()
        } else {
            top_hogs.join(", ")
        };
        let peak_pct = memory_pressure_peak(worst).2;
        findings.push(json!({
            "severity": if peak_pct >= 97.0 { "critical" } else { "warning" },
            "category": "memory_pressure",
            "summary": format!(
                "{} memory pressure episode(s) totaling {} (peak {:.1}% commit/RAM; top consumers: {})",
                mp.len(),
                fmt_duration(total_dur),
                peak_pct,
                hogs_str
            ),
            "event_id": worst.id,
        }));
        actions.push(format!(
            "Reduce concurrent memory load from {} to keep commit charge below {:.0}%.",
            hogs_str, cfg.commit_high_pct
        ));
    }

    if let Some(dpc) = by_kind.get("dpc_storm") {
        let worst = dpc
            .iter()
            .max_by(|a, b| {
                a.details["peak_pct"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&b.details["peak_pct"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        findings.push(json!({
            "severity": "warning",
            "category": "dpc_storm",
            "summary": format!(
                "{} kernel DPC/ISR interrupt storm(s) recorded (peak {:.1}% CPU)",
                dpc.len(),
                worst.details["peak_pct"].as_f64().unwrap_or(0.0)
            ),
            "event_id": worst.id,
        }));
    }

    if let Some(freezes) = by_kind.get("ui_freeze") {
        let mut by_proc: HashMap<String, usize> = HashMap::new();
        for e in freezes {
            let p = e
                .details
                .get("process")
                .and_then(|v| v.as_str())
                .filter(|s| !s.is_empty())
                .or_else(|| e.details.get("source").and_then(|v| v.as_str()))
                .unwrap_or("unknown")
                .to_string();
            *by_proc.entry(p).or_insert(0) += 1;
        }
        let mut top_f: Vec<(String, usize)> = by_proc.into_iter().collect();
        top_f.sort_by_key(|a| std::cmp::Reverse(a.1));
        let top_str = top_f
            .into_iter()
            .take(3)
            .map(|(p, c)| format!("{} ({}x)", p, c))
            .collect::<Vec<_>>()
            .join(", ");
        findings.push(json!({
            "severity": "warning",
            "category": "ui_freeze",
            "summary": format!(
                "{} UI freeze episode(s) recorded — most frequent: {}",
                freezes.len(),
                top_str
            ),
            "event_id": freezes[0].id,
        }));
    }

    if let Some(gt) = by_kind.get("gpu_thermal") {
        let worst = gt
            .iter()
            .max_by(|a, b| {
                a.details["peak_temp_c"]
                    .as_f64()
                    .unwrap_or(0.0)
                    .partial_cmp(&b.details["peak_temp_c"].as_f64().unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        findings.push(json!({
            "severity": "warning",
            "category": "gpu_thermal",
            "summary": format!(
                "{} GPU thermal episode(s) (peak {:.0}°C)",
                gt.len(),
                worst.details["peak_temp_c"].as_f64().unwrap_or(0.0)
            ),
            "event_id": worst.id,
        }));
    }

    let mut gpu_batt: Vec<&EventRecord> = Vec::new();
    if let Some(v) = by_kind.get("gpu_battery_drain") {
        gpu_batt.extend(v.iter().copied());
    }
    if let Some(v) = by_kind.get("gpu_awake_on_battery") {
        gpu_batt.extend(v.iter().copied());
    }
    if !gpu_batt.is_empty() {
        findings.push(json!({
            "severity": "info",
            "category": "gpu_battery_drain",
            "summary": format!(
                "{} dGPU awake-on-battery episode(s)",
                gpu_batt.len()
            ),
            "event_id": gpu_batt[0].id,
        }));
    }

    if let Some(lds) = by_kind.get("low_disk_space") {
        let e0 = lds[0];
        let free = detail_f64(&e0.details, &["disk_free_gb", "free_gb"]).unwrap_or(0.0);
        findings.push(json!({
            "severity": "warning",
            "category": "low_disk_space",
            "summary": format!("System drive low on free space ({:.1} GB free)", free),
            "event_id": e0.id,
        }));
    }

    if let Some(ub) = by_kind.get("upload_burst") {
        findings.push(json!({
            "severity": "info",
            "category": "upload_burst",
            "summary": format!("{} sustained upload burst(s) recorded", ub.len()),
            "event_id": ub[0].id,
        }));
    }

    if let Some(bg) = by_kind.get("background_cpu") {
        let mut procs: Vec<String> = bg
            .iter()
            .filter_map(|e| {
                e.details
                    .get("process")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
            })
            .collect();
        procs.sort();
        procs.dedup();
        findings.push(json!({
            "severity": "info",
            "category": "background_cpu",
            "summary": format!(
                "{} sustained background CPU episode(s) ({})",
                bg.len(),
                procs.join(", ")
            ),
            "event_id": bg[0].id,
        }));
    }

    if let Some(anom) = by_kind.get("anomaly") {
        findings.push(json!({
            "severity": "info",
            "category": "anomaly",
            "summary": format!(
                "{} unusual workload episode(s) flagged by kNN detector",
                anom.len()
            ),
            "event_id": anom[0].id,
        }));
    }

    let has_crit_finding = findings
        .iter()
        .any(|f| f["severity"].as_str() == Some("critical"));
    let status = if crit_count > 0 || has_crit_finding {
        "CRITICAL"
    } else if !findings.is_empty() {
        "WARNING"
    } else {
        "HEALTHY"
    };

    let mut lines = vec![format!(
        "System Health Audit (last {:.0}h) — Status: {} ({} events, {} critical)",
        hours,
        status,
        events.len(),
        crit_count
    )];
    if findings.is_empty() {
        lines.push(
            "  No significant performance anomalies, leaks, or resource pressure detected."
                .to_string(),
        );
    } else {
        lines.push(String::new());
        lines.push("Key Findings:".to_string());
        for f in &findings {
            lines.push(format!(
                "  • [{}] {} (see event #{})",
                f["severity"].as_str().unwrap_or("info").to_uppercase(),
                f["summary"].as_str().unwrap_or(""),
                f["event_id"].as_i64().unwrap_or(0)
            ));
        }
    }
    if !actions.is_empty() {
        lines.push(String::new());
        lines.push("Prioritized Action Plan:".to_string());
        for (i, a) in actions.iter().enumerate() {
            lines.push(format!("  {}. {}", i + 1, a));
        }
    }

    json!({
        "status": status,
        "lookback_hours": hours,
        "total_events": events.len(),
        "critical_events": crit_count,
        "findings": findings,
        "actions": actions,
        "report": lines.join("\n"),
    })
}
