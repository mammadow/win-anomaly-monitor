//! Command-line interface (`run`, `status`, `report`, `explain`, `ask`,
//! `diagnose`, `tune`, `top`, `reset`).

use crate::config::{load_config, MonitorConfig};
use crate::diagnostics::{
    ask_question, audit_system_health, build_event_context, explain_event, SYSTEM_PROMPT,
};
use crate::engine::{read_state_file, Monitor};
use crate::events::fmt_duration;
use crate::replay::tune_grid;
use crate::sampler::unix_now;
use crate::storage::{db_summary, Storage};
use crate::tui::run_top;
use crate::winapi;
use chrono::{Local, TimeZone};
use clap::{Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "resmon",
    version = "4.0.0",
    about = "Lightweight Windows 11 resource & anomaly monitor (pure Rust v4.0)"
)]
pub struct CliArgs {
    /// Path to config.toml
    #[arg(short = 'c', long = "config", global = true)]
    pub config: Option<PathBuf>,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Run the background monitoring daemon (default command)
    Run {
        /// Show a live system tray icon in the Windows notification area
        #[arg(long = "tray")]
        tray: bool,
    },
    /// Show live monitor state, telemetry summary, and model readiness
    Status,
    /// Show recent anomaly and safety events
    Report {
        /// Lookback window in hours (default: 24)
        #[arg(short = 'H', long = "hours", default_value_t = 24.0)]
        hours: f64,
        /// Filter by event kind (anomaly, ui_freeze, memory_pressure, etc.)
        #[arg(short = 'k', long = "kind")]
        kind: Option<String>,
        /// Show detailed JSON fields for each event
        #[arg(short = 'v', long = "verbose")]
        verbose: bool,
        /// Output machine-readable JSON
        #[arg(long = "json")]
        json_out: bool,
    },
    /// Root-cause analysis for a specific (or latest) event
    Explain {
        /// Event ID (e.g. `42` or `#42`) or `latest`
        #[arg(default_value = "latest")]
        target: String,
        /// When target is `latest`, filter by event kind
        #[arg(short = 'k', long = "kind")]
        kind: Option<String>,
        /// Optional follow-up question about this event
        #[arg(short = 'q', long = "question")]
        question: Option<String>,
        /// Force deterministic offline RCA (skip Ollama / OpenAI)
        #[arg(long = "offline")]
        offline: bool,
        /// Export the full LLM forensic prompt + JSON telemetry packet
        #[arg(long = "prompt")]
        prompt: bool,
        /// Output machine-readable JSON
        #[arg(long = "json")]
        json_out: bool,
        /// Wait for Enter before exiting (used when opened from the system tray)
        #[arg(long = "wait", hide = true)]
        wait: bool,
    },
    /// Ask a natural-language diagnostic question about recent system behavior
    Ask {
        /// Question (e.g. "why did my PC freeze?" or "what is eating RAM?")
        #[arg(required = true)]
        question: Vec<String>,
        /// Lookback window in hours (default: 24)
        #[arg(short = 'H', long = "hours", default_value_t = 24.0)]
        hours: f64,
        /// Force deterministic offline analyzer
        #[arg(long = "offline")]
        offline: bool,
        /// Output machine-readable JSON
        #[arg(long = "json")]
        json_out: bool,
    },
    /// Holistic system health audit & prioritized action plan
    Diagnose {
        /// Lookback window in hours (default: 24)
        #[arg(short = 'H', long = "hours", default_value_t = 24.0)]
        hours: f64,
        /// Output machine-readable JSON
        #[arg(long = "json")]
        json_out: bool,
    },
    /// Replay recorded history across sensitivity settings
    Tune {
        /// Days of history to replay (default: all)
        #[arg(short = 'd', long = "days")]
        days: Option<f64>,
    },
    /// Launch interactive live terminal dashboard
    Top,
    /// Delete learned model and recorded history
    Reset {
        /// Skip confirmation prompt
        #[arg(short = 'y', long = "yes")]
        yes: bool,
    },
}

pub fn run_cli(args: CliArgs) -> i32 {
    let cfg = match load_config(args.config.as_deref()) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{}", e);
            return 2;
        }
    };

    match args.command.unwrap_or(Commands::Run { tray: false }) {
        Commands::Run { tray } => cmd_run(cfg, tray),
        Commands::Status => cmd_status(&cfg),
        Commands::Report {
            hours,
            kind,
            verbose,
            json_out,
        } => cmd_report(&cfg, hours, kind.as_deref(), verbose, json_out),
        Commands::Explain {
            target,
            kind,
            question,
            offline,
            prompt,
            json_out,
            wait,
        } => {
            let code = cmd_explain(
                &cfg,
                &target,
                kind.as_deref(),
                question.as_deref(),
                offline,
                prompt,
                json_out,
            );
            if wait {
                println!("\nPress Enter to close...");
                let _ = io::stdout().flush();
                let _ = io::stdin().read_line(&mut String::new());
            }
            code
        }
        Commands::Ask {
            question,
            hours,
            offline,
            json_out,
        } => {
            let q = question.join(" ");
            cmd_ask(&cfg, &q, hours, offline, json_out)
        }
        Commands::Diagnose { hours, json_out } => cmd_diagnose(&cfg, hours, json_out),
        Commands::Tune { days } => cmd_tune(&cfg, days),
        Commands::Top => match run_top(&cfg) {
            Ok(()) => 0,
            Err(e) => {
                eprintln!("{}", e);
                1
            }
        },
        Commands::Reset { yes } => cmd_reset(&cfg, yes),
    }
}

pub fn cmd_run(cfg: MonitorConfig, tray: bool) -> i32 {
    let data_dir = cfg.data_dir.clone();
    let append_err = |msg: &str| {
        eprintln!("{}", msg);
        let log_path = data_dir.join("monitor.log");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path)
        {
            let ts = Local::now().format("%Y-%m-%d %H:%M:%S");
            let _ = std::io::Write::write_fmt(&mut f, format_args!("{} [ERROR] {}\n", ts, msg));
        }
    };
    let mut mon = match Monitor::new(cfg) {
        Ok(m) => m,
        Err(e) => {
            append_err(&e);
            return 1;
        }
    };
    if let Err(e) = mon.run(tray) {
        append_err(&e);
        return 1;
    }
    0
}

pub fn cmd_status(cfg: &MonitorConfig) -> i32 {
    let state = read_state_file(&cfg.state_path());
    let db = db_summary(&cfg.db_path());

    if let Some(ref st) = state {
        let pid = st["pid"].as_u64().unwrap_or(0) as u32;
        let updated = st["updated_at"].as_f64().unwrap_or_else(|| {
            st["updated"]
                .as_str()
                .and_then(|s| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok())
                .and_then(|ndt| ndt.and_local_timezone(Local).single())
                .map(|dt| dt.timestamp() as f64)
                .unwrap_or(0.0)
        });
        let age = (unix_now() - updated).max(0.0);
        let stopped = st["stopped"].as_bool().unwrap_or(false);
        let alive = !stopped && winapi::pid_exists(pid) && age < 60.0;
        let eng = st["engine"].as_str().unwrap_or("python-v3");
        println!(
            "Monitor: {} [{}]",
            if alive {
                format!("RUNNING (PID {}, updated {:.0}s ago)", pid, age)
            } else {
                format!("STOPPED (last update {:.0}s ago)", age)
            },
            eng
        );
        if let Some(sum) = st["summary"].as_str() {
            println!("Now:     {}", sum);
        }
        let act = &st["activity"];
        if let Some(fg) = act["foreground"].as_str() {
            let idle_note = if act["user_idle"].as_bool().unwrap_or(false) {
                " (user idle)"
            } else {
                ""
            };
            println!("Active:  foreground = {}{}", fg, idle_note);
        }
        if let Some(top_cpu) = act["top_cpu"].as_array() {
            if !top_cpu.is_empty() {
                let s: Vec<String> = top_cpu
                    .iter()
                    .map(|c| {
                        format!(
                            "{} ({} cores)",
                            c["name"].as_str().unwrap_or("?"),
                            c["cores"].as_f64().unwrap_or(0.0)
                        )
                    })
                    .collect();
                println!("Top CPU: {}", s.join(", "));
            }
        }
        if let Some(top_mem) = act["top_memory"].as_array() {
            if !top_mem.is_empty() {
                let s: Vec<String> = top_mem
                    .iter()
                    .take(3)
                    .map(|p| {
                        format!(
                            "{} ({} MB)",
                            p["name"].as_str().unwrap_or("?"),
                            p["private_mb"].as_i64().unwrap_or(0)
                        )
                    })
                    .collect();
                println!("Top RAM: {}", s.join(", "));
            }
        }

        let resp = &st["responsiveness"];
        if resp["alive"].as_bool().unwrap_or(false) {
            let mut parts = Vec::new();
            for k in ["foreground", "taskbar", "compositor"] {
                let t = &resp["threads"][k];
                if t.is_object() {
                    let mut tag = format!("{:.0} ms", t["latency_ms"].as_f64().unwrap_or(0.0));
                    if t["hung"].as_bool().unwrap_or(false) {
                        tag.push_str(" HUNG");
                    } else if t["timed_out"].as_bool().unwrap_or(false) {
                        tag.push_str(" TIMEOUT");
                    }
                    parts.push(format!("{} {}", k, tag));
                }
            }
            if !parts.is_empty() {
                println!("UI ping: {}", parts.join(", "));
            }
        } else if st["ui_probe"].is_object() {
            let up = &st["ui_probe"];
            println!(
                "UI ping: {} probes, {} freezes, max late {:.2}s (boosted={})",
                up["probes"].as_i64().unwrap_or(0),
                up["freezes"].as_i64().unwrap_or(0),
                up["max_late_sec"].as_f64().unwrap_or(0.0),
                up["boosted"].as_bool().unwrap_or(false)
            );
        }

        let m = &st["model"];
        if m["ready"].as_bool().unwrap_or(false) || m["windows"].as_i64().unwrap_or(0) > 0 {
            let trained_ago_min = m["trained_ago_min"].as_f64().unwrap_or_else(|| {
                m["trained"]
                    .as_str()
                    .and_then(|s| {
                        chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok()
                    })
                    .and_then(|ndt| ndt.and_local_timezone(Local).single())
                    .map(|dt| ((unix_now() - dt.timestamp() as f64) / 60.0).max(0.0))
                    .unwrap_or(0.0)
            });
            println!(
                "Model:   ready ({} windows / {:.1} h of history, threshold {:.2}, trained {:.0} min ago)",
                m["windows"].as_i64().unwrap_or(0),
                m["hours_of_data"].as_f64().unwrap_or(0.0),
                m["threshold"].as_f64().unwrap_or(0.0),
                trained_ago_min
            );
        } else {
            println!(
                "Model:   collecting baseline (need {:.1} h; have {:.1} h)",
                m["min_training_hours"]
                    .as_f64()
                    .unwrap_or(cfg.min_training_hours),
                db["span_hours"].as_f64().unwrap_or(0.0)
            );
        }

        if let Some(open_evs) = st["open_events"].as_array() {
            if !open_evs.is_empty() {
                println!("Open ({}):", open_evs.len());
                for ev in open_evs {
                    let tag = ev["severity"]
                        .as_str()
                        .or_else(|| ev["kind"].as_str())
                        .unwrap_or("info");
                    println!("  [{}] {}", tag, ev["title"].as_str().unwrap_or(""));
                }
            }
        }
    } else {
        println!("Monitor: not started yet (no state.json)");
    }

    println!(
        "Storage: {} windows ({:.1} h), {} 1s samples, {} total events in {}",
        db["windows"].as_i64().unwrap_or(0),
        db["span_hours"].as_f64().unwrap_or(0.0),
        db["samples"].as_i64().unwrap_or(0),
        db["events"].as_i64().unwrap_or(0),
        cfg.db_path().display()
    );
    0
}

pub fn cmd_report(
    cfg: &MonitorConfig,
    hours: f64,
    kind: Option<&str>,
    verbose: bool,
    json_out: bool,
) -> i32 {
    if !cfg.db_path().exists() {
        println!("No database yet. Start the monitor with: resmon");
        return 0;
    }
    let st = match Storage::open_readonly(&cfg.db_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };
    let since = unix_now() - hours * 3600.0;
    let rows = st.query_events(since, kind, 200);
    if json_out {
        println!(
            "{}",
            serde_json::to_string_pretty(&rows).unwrap_or_else(|_| "[]".to_string())
        );
        return 0;
    }
    if rows.is_empty() {
        println!("No events in the last {:.0} h.", hours);
        return 0;
    }
    println!("Events in the last {:.0} h ({}):\n", hours, rows.len());
    for r in rows {
        let t_str = Local
            .timestamp_opt(r.ts as i64, 0)
            .single()
            .map(|dt| dt.format("%Y-%m-%d %H:%M:%S").to_string())
            .unwrap_or_default();
        let dur_s = (r.ts_end - r.ts).max(0.0);
        let dur = if dur_s >= 1.0 {
            format!(" ({})", fmt_duration(dur_s))
        } else {
            String::new()
        };
        println!(
            "#{:<4} {}  {:8}  {:18}{}  {}",
            r.id, t_str, r.severity, r.kind, dur, r.title
        );
        if verbose {
            if let Some(obj) = r.details.as_object() {
                for (k, v) in obj {
                    println!("      {}: {}", k, v);
                }
            }
        }
    }
    0
}

pub fn cmd_explain(
    cfg: &MonitorConfig,
    target: &str,
    kind: Option<&str>,
    question: Option<&str>,
    offline: bool,
    prompt_only: bool,
    json_out: bool,
) -> i32 {
    if !cfg.db_path().exists() {
        println!("No database yet. Start the monitor with: resmon");
        return 0;
    }
    let st = match Storage::open_readonly(&cfg.db_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };

    let target_trimmed = target.trim();
    let event_id = if target_trimmed.is_empty() || target_trimmed.eq_ignore_ascii_case("latest") {
        None
    } else {
        match target_trimmed.trim_start_matches('#').parse::<i64>() {
            Ok(id) => Some(id),
            Err(_) => {
                eprintln!(
                    "Invalid event target {:?}: pass an event ID (e.g. `42`) or `latest`.",
                    target
                );
                return 2;
            }
        }
    };

    if prompt_only {
        let Some(ctx) = build_event_context(&st, cfg, event_id, kind) else {
            println!("No matching event found.");
            return 1;
        };
        println!(
            "{}\n\n---\nTelemetry Context:\n{}",
            SYSTEM_PROMPT,
            serde_json::to_string_pretty(&ctx).unwrap_or_default()
        );
        return 0;
    }

    let Some(res) = explain_event(&st, cfg, event_id, kind, question, offline) else {
        println!("No matching event found.");
        return 1;
    };

    if json_out {
        println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
    } else {
        println!(
            "[Diagnostics Engine: {}]\n",
            res["engine"].as_str().unwrap_or("offline-rca")
        );
        println!("{}", res["explanation"].as_str().unwrap_or(""));
    }
    0
}

pub fn cmd_ask(
    cfg: &MonitorConfig,
    question: &str,
    hours: f64,
    offline: bool,
    json_out: bool,
) -> i32 {
    if !cfg.db_path().exists() {
        println!("No database yet. Start the monitor with: resmon");
        return 0;
    }
    let st = match Storage::open_readonly(&cfg.db_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };
    let res = ask_question(&st, cfg, question, hours, offline);
    if json_out {
        println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
    } else {
        println!(
            "[Diagnostics Engine: {}]\n",
            res["engine"].as_str().unwrap_or("offline-rca")
        );
        println!("{}", res["answer"].as_str().unwrap_or(""));
    }
    0
}

pub fn cmd_diagnose(cfg: &MonitorConfig, hours: f64, json_out: bool) -> i32 {
    if !cfg.db_path().exists() {
        println!("No database yet. Start the monitor with: resmon");
        return 0;
    }
    let st = match Storage::open_readonly(&cfg.db_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };
    let res = audit_system_health(&st, cfg, hours);
    if json_out {
        println!("{}", serde_json::to_string_pretty(&res).unwrap_or_default());
    } else {
        println!("{}", res["report"].as_str().unwrap_or(""));
    }
    0
}

pub fn cmd_tune(cfg: &MonitorConfig, days: Option<f64>) -> i32 {
    if !cfg.db_path().exists() {
        println!("No database yet.");
        return 0;
    }
    let st = match Storage::open_readonly(&cfg.db_path()) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}", e);
            return 1;
        }
    };
    let windows = st.load_windows(days);
    let min_w = ((cfg.min_training_hours * 3600.0) / (cfg.window_sec as f64)).max(1.0) as usize;
    if windows.len() < min_w * 2 {
        println!(
            "Need at least ~{:.1} h of history to replay; have {:.1} h.",
            cfg.min_training_hours * 2.0,
            (windows.len() as f64) * (cfg.window_sec as f64) / 3600.0
        );
        return 0;
    }
    let rows = tune_grid(&windows, cfg, &[0.0005, 0.001, 0.002, 0.005], &[2, 3, 6]);
    let span_days = rows.first().and_then(|r| r["days"].as_f64()).unwrap_or(0.0);
    println!(
        "Replayed {} windows ({:.2} days):\n",
        windows.len(),
        span_days
    );
    println!("  alert_quantile   min_consecutive_windows   episodes   episodes/day");
    for r in rows {
        let q = r["alert_quantile"].as_f64().unwrap_or(0.0);
        let k = r["min_consecutive_windows"].as_u64().unwrap_or(0) as usize;
        let mark = if (q - cfg.alert_quantile).abs() < 1e-9 && k == cfg.min_consecutive_windows {
            "  <-- current"
        } else {
            ""
        };
        println!(
            "  {:14.4}   {:23}   {:8}   {:12.2}{}",
            q,
            k,
            r["events"].as_u64().unwrap_or(0),
            r["per_day"].as_f64().unwrap_or(0.0),
            mark
        );
    }
    0
}

pub fn cmd_reset(cfg: &MonitorConfig, yes: bool) -> i32 {
    if !yes {
        print!(
            "Delete all recorded metrics, events, and learned model in {:?}? [y/N] ",
            cfg.data_dir
        );
        let _ = io::stdout().flush();
        let mut ans = String::new();
        if io::stdin().read_line(&mut ans).is_err() || !ans.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return 0;
        }
    }
    for p in [
        cfg.db_path(),
        cfg.model_path(),
        cfg.data_dir.join("model.npz"),
        cfg.state_path(),
        cfg.events_jsonl_path(),
    ] {
        for ext in ["", "-wal", "-shm"] {
            let target = PathBuf::from(format!("{}{}", p.display(), ext));
            if target.exists() {
                if let Err(e) = std::fs::remove_file(&target) {
                    eprintln!(
                        "Could not remove {} (is the monitor running?): {}",
                        target.display(),
                        e
                    );
                    return 1;
                }
            }
        }
    }
    println!("Reset complete.");
    0
}

pub fn parse_and_run_from<I, T>(itr: I) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<std::ffi::OsString> + Clone,
{
    let args = match CliArgs::try_parse_from(itr) {
        Ok(a) => a,
        Err(e) => {
            let _ = e.print();
            return 2;
        }
    };
    run_cli(args)
}
