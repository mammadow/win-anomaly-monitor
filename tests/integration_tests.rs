//! Comprehensive integration test suite for `resmon` v4.0 (pure Rust).

use resmon::cli::{cmd_ask, cmd_diagnose, cmd_explain, cmd_report, cmd_reset, cmd_status};
use resmon::config::{load_config, MonitorConfig};
use resmon::detector::{fit_model, sq_dist, sq_dist_scalar, AnomalyModel, Trainer, K_NEIGHBOURS};
use resmon::diagnostics::{
    ask_question, audit_system_health, build_event_context, explain_event, offline_explain_event,
};
use resmon::engine::InstanceLock;
use resmon::features::{
    fmt_bytes, fmt_feature, get_metric, to_matrix_row, MetricMap, WindowAggregator, ML_FEATURES,
    NUM_FEATURES,
};
use resmon::processes::{ProcStats, ProcessTracker};
use resmon::replay::{replay, tune_grid, ReplayMonitor};
use resmon::responsiveness::{ResponsivenessProbe, Stall, StallTracker};
use resmon::rules::{
    trend, BackgroundCpu, GpuAwakeOnBattery, GpuThermal, LowDiskSpace, MemoryPressure, UiFreeze,
    UploadBurst,
};
use resmon::sampler::{unix_now, SystemSampler};
use resmon::storage::{db_summary, Storage};
use resmon::winapi;
use serde_json::json;
use std::path::PathBuf;
use std::time::Duration;

fn temp_test_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "resmon_rust_test_{}_{}_{}",
        name,
        winapi::current_pid(),
        (unix_now() * 1000.0) as u64
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn make_sample(ts: f64, cpu: f64, ram: f64) -> MetricMap {
    let mut m = MetricMap::new();
    m.insert("ts".to_string(), ts);
    m.insert("cpu_pct".to_string(), cpu);
    m.insert("core_max_pct".to_string(), (cpu * 1.5).min(100.0));
    m.insert("dpc_pct".to_string(), 0.5);
    m.insert("isr_pct".to_string(), 0.2);
    m.insert("dpc_max_core_pct".to_string(), 2.0);
    m.insert("ram_pct".to_string(), ram);
    m.insert("commit_pct".to_string(), ram);
    m.insert("disk_read_bps".to_string(), 10_000.0);
    m.insert("disk_write_bps".to_string(), 20_000.0);
    m.insert("net_sent_bps".to_string(), 5_000.0);
    m.insert("net_recv_bps".to_string(), 15_000.0);
    m.insert("hard_faults_ps".to_string(), 0.0);
    m.insert("disk_free_gb".to_string(), 120.0);
    m.insert("on_ac".to_string(), 1.0);
    m.insert("user_active".to_string(), 1.0);
    m.insert("gpu_awake".to_string(), 0.0);
    m.insert("gpu_util_pct".to_string(), f64::NAN);
    m.insert("gpu_power_w".to_string(), f64::NAN);
    m.insert("gpu_temp_c".to_string(), f64::NAN);
    m.insert("gpu_vram_pct".to_string(), f64::NAN);
    m.insert("gpu_clock_mhz".to_string(), f64::NAN);
    m
}

#[test]
fn test_config_validation_and_toml_loading() {
    let dir = temp_test_dir("config");
    let toml_path = dir.join("config.toml");
    std::fs::write(
        &toml_path,
        r#"
        [sampling]
        sample_interval_sec = 1.0
        window_sec = 10
        [ai]
        ai_provider = "offline"
        "#,
    )
    .unwrap();

    let cfg = load_config(Some(&toml_path)).unwrap();
    assert_eq!(cfg.window_sec, 10);
    assert_eq!(cfg.ai_provider, "offline");

    let mut bad_cfg = MonitorConfig::default();
    bad_cfg.alert_quantile = 0.9;
    assert!(bad_cfg.validate().is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_features_and_window_aggregator() {
    let mut raw = [0.0f64; NUM_FEATURES];
    raw[0] = 25.0; // cpu_pct
    raw[10] = 55.0; // gpu_temp_c
    raw[13] = 0.0; // gpu_awake = 0 -> temp should become 0 in ML space
    let x = to_matrix_row(&raw);
    assert_eq!(x[0], 25.0);
    assert_eq!(x[10], 0.0);

    raw[13] = 1.0; // gpu_awake = 1 -> temp preserved
    let x2 = to_matrix_row(&raw);
    assert_eq!(x2[10], 55.0);

    assert_eq!(fmt_bytes(1_048_576.0, true), "1.0 MB/s");
    assert_eq!(fmt_feature(&ML_FEATURES[0], 42.3), "42%");
    assert_eq!(fmt_feature(&ML_FEATURES[10], f64::NAN), "GPU asleep");

    let mut agg = WindowAggregator::new(10);
    for t in 100..110 {
        assert!(agg.add(make_sample(t as f64, 20.0, 50.0)).is_none());
    }
    let cw = agg
        .add(make_sample(110.0, 30.0, 50.0))
        .expect("bucket 100..110 should flush at t=110");
    assert_eq!(get_metric(&cw.row, "ts"), 100.0);
    assert!((get_metric(&cw.row, "cpu_pct") - 20.0).abs() < 1e-6);
    assert!(get_metric(&cw.row, "cpu_p95").is_finite());
}

#[test]
fn test_knn_detector_catches_incidents_and_persists() {
    let dir = temp_test_dir("detector");
    let n = 400;
    let mut raw = Vec::with_capacity(n);
    let mut ts = Vec::with_capacity(n);
    for i in 0..n {
        let t = 1_700_000_000.0 + (i as f64) * 10.0;
        ts.push(t);
        let wobble = (((i * 73 + 19) % 101) as f64) * 0.15 + (i as f64) * 0.01;
        let wobble2 = (((i * 37 + 11) % 89) as f64) * 0.12;
        let mut row = [0.0f64; NUM_FEATURES];
        row[0] = 8.0 + wobble; // cpu_pct
        row[1] = 20.0 + wobble * 1.3; // core_max_pct
        row[2] = 55.0 + wobble2; // ram_pct
        row[3] = 60.0 + wobble2; // commit_pct
        row[4] = 20_000.0;
        row[5] = 30_000.0;
        row[6] = 10_000.0;
        row[7] = 50_000.0;
        row[8] = f64::NAN;
        row[9] = f64::NAN;
        row[10] = f64::NAN;
        row[11] = f64::NAN;
        row[12] = f64::NAN;
        row[13] = 0.0; // gpu_awake
        row[14] = 1.0; // on_ac
        row[15] = 1.0; // user_active
        raw.push(row);
    }

    let model = fit_model(&raw, Some(&ts), 0.01, 10.0, K_NEIGHBOURS);
    assert_eq!(model.n_windows, n);
    assert!(model.threshold > 0.0);

    // Normal window should not be flagged
    let normal_x = to_matrix_row(&raw[0]);
    let (_, normal_flag) = model.evaluate_row(&normal_x);
    assert!(!normal_flag);

    // Heavy CPU + Disk write anomaly should be flagged and explained
    let mut spike = raw[0];
    spike[0] = 85.0; // CPU 85%
    spike[1] = 100.0;
    spike[5] = 250_000_000.0; // 250 MB/s disk write
    let spike_x = to_matrix_row(&spike);
    let (score, anom) = model.evaluate_row(&spike_x);
    assert!(anom, "spike should be anomalous (score={})", score);

    let mut raw_map = MetricMap::new();
    for (j, f) in ML_FEATURES.iter().enumerate() {
        raw_map.insert(f.column.to_string(), spike[j]);
    }
    let reasons = model.explain(&spike_x, &raw_map, 3);
    assert!(!reasons.is_empty());

    // Save & load round-trip
    let model_path = dir.join("model.bin");
    model.save(&model_path).unwrap();
    let loaded = AnomalyModel::load(&model_path).unwrap();
    let (score2, anom2) = loaded.evaluate_row(&spike_x);
    assert!((score - score2).abs() < 1e-5);
    assert_eq!(anom, anom2);

    // Background trainer test
    let mut trainer = Trainer::new(&dir);
    let raw_clone = raw.clone();
    let ts_clone = ts.clone();
    assert!(trainer.start(move || Some((ts_clone, raw_clone)), 0.01, 10.0));
    trainer.join();
    let polled = trainer.poll().unwrap();
    assert!(polled.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_rules_recurrence_coalescing_and_leaks() {
    let mut cfg = MonitorConfig::default();
    cfg.memory_sustain_sec = 10.0;
    let mut mon = ReplayMonitor::new(cfg);
    let mut mp = MemoryPressure::default();

    // First high-memory episode (t = 100..120)
    let mut s_hi = make_sample(100.0, 20.0, 95.0);
    s_hi.insert("hard_faults_ps".to_string(), 120.0);
    mp.on_sample(&mut mon, &s_hi, 100.0);
    mp.on_sample(&mut mon, &s_hi, 115.0);
    assert_eq!(mon.events.events.len(), 1);
    assert!(mon.events.events[0].title.contains("swap thrashing"));

    // Clear at t = 130
    let s_lo = make_sample(130.0, 10.0, 70.0);
    mp.on_sample(&mut mon, &s_lo, 130.0);
    assert!(!mon.events.events[0].is_open);

    // Second high-memory episode within 10 minutes (t = 300..315) -> should coalesce into event #1!
    mp.on_sample(&mut mon, &s_hi, 300.0);
    mp.on_sample(&mut mon, &s_hi, 315.0);
    assert_eq!(
        mon.events.events.len(),
        1,
        "recurrence within 10 minutes must reopen existing episode instead of spamming a new event"
    );
    assert!(mon.events.events[0].is_open);
    assert_eq!(mon.events.events[0].details["recurrences"], json!(1));

    // OLS leak trend test
    let t: Vec<f64> = (0..10).map(|i| (i as f64) * 600.0).collect();
    let y: Vec<f64> = (0..10)
        .map(|i| 200_000_000.0 + (i as f64) * 100_000_000.0)
        .collect();
    let (slope, r2, monotone) = trend(&t, &y);
    assert!(r2 > 0.99);
    assert!(monotone);
    assert!(slope > 500_000_000.0);
}

#[test]
fn test_other_detectors_and_background_cpu() {
    let mut cfg = MonitorConfig::default();
    cfg.gpu_temp_sustain_sec = 5.0;
    cfg.gpu_battery_wake_sustain_sec = 5.0;
    cfg.net_upload_sustain_sec = 5.0;
    cfg.background_cpu_sustain_sec = 20.0;
    cfg.process_interval_sec = 5.0;
    let mut mon = ReplayMonitor::new(cfg);

    // LowDiskSpace
    let mut lds = LowDiskSpace::new();
    let mut s = make_sample(100.0, 10.0, 50.0);
    s.insert("disk_free_gb".to_string(), 10.0);
    lds.on_sample(&mut mon, &s, 100.0);
    assert!(mon.events.events.iter().any(|e| e.kind == "low_disk_space"));

    // GpuThermal
    let mut gt = GpuThermal::default();
    s.insert("gpu_awake".to_string(), 1.0);
    s.insert("gpu_temp_c".to_string(), 88.0);
    s.insert("gpu_power_w".to_string(), 65.0);
    gt.on_sample(&mut mon, &s, 100.0);
    gt.on_sample(&mut mon, &s, 110.0);
    assert!(mon.events.events.iter().any(|e| e.kind == "gpu_thermal"));

    // GpuAwakeOnBattery
    let mut gb = GpuAwakeOnBattery::default();
    s.insert("on_ac".to_string(), 0.0);
    s.insert("gpu_util_pct".to_string(), 1.0);
    s.insert("gpu_power_w".to_string(), 18.0);
    gb.on_sample(&mut mon, &s, 200.0);
    gb.on_sample(&mut mon, &s, 210.0);
    assert!(mon
        .events
        .events
        .iter()
        .any(|e| e.kind == "gpu_battery_drain"));

    // UploadBurst
    let mut ub = UploadBurst::default();
    s.insert("net_sent_bps".to_string(), 15.0 * 1_048_576.0);
    ub.on_sample(&mut mon, &s, 300.0);
    ub.on_sample(&mut mon, &s, 310.0);
    assert!(mon.events.events.iter().any(|e| e.kind == "upload_burst"));

    // BackgroundCpu
    let mut bg = BackgroundCpu::new();
    for step in 0..6 {
        let now = 500.0 + (step as f64) * 5.0;
        let stats = vec![ProcStats {
            key: (4444, 12345),
            name: "python.exe [worker.py]".to_string(),
            base_name: "python.exe".to_string(),
            cpu_cores: 2.5,
            cpu_pct: 15.0,
            read_bps: 0.0,
            write_bps: 0.0,
            other_bps: 0.0,
            hard_faults_ps: 0.0,
            private_bytes: 500 * 1_048_576,
            working_set: 500 * 1_048_576,
            threads: 8,
            handles: 300,
            foreground: false,
        }];
        bg.on_sweep(&mut mon, &stats, now);
    }
    let bg_ev = mon
        .events
        .events
        .iter()
        .find(|e| e.kind == "background_cpu")
        .expect("background_cpu should trigger");
    assert!(bg_ev.title.contains("python.exe [worker.py]"));
}

#[test]
fn test_stall_tracker_and_ui_freeze_attribution() {
    let mut tracker = StallTracker::new(250, 2000, 500.0);
    let stall1 = Stall {
        ts: 100.0,
        source: "foreground".to_string(),
        latency_ms: 800.0,
        timed_out: false,
        hung: false,
        pid: 1234,
        tid: 5678,
        process: "chrome.exe".to_string(),
        title: "Tab".to_string(),
        hwnd: 0xABCD,
    };
    let stall2 = Stall {
        ts: 101.0,
        latency_ms: 1500.0,
        timed_out: false,
        hung: true,
        ..stall1.clone()
    };

    tracker.step(&[stall1], 100.0, Some((100.0, 100.0)));
    tracker.step(&[stall2], 101.0, Some((101.0, 101.0)));
    let (done, ongoing) = tracker.drain();
    assert!(done.is_empty());
    assert_eq!(ongoing.len(), 1);
    assert_eq!(ongoing[0].pings, 2);
    assert!(ongoing[0].hung);

    // Expire after MERGE_GAP_SEC (3s)
    tracker.step(&[], 105.0, Some((105.0, 105.0)));
    let (done2, ongoing2) = tracker.drain();
    assert_eq!(done2.len(), 1);
    assert!(ongoing2.is_empty());

    // Classify UI freeze cause via UiFreeze
    let mut mon = ReplayMonitor::new(MonitorConfig::default());
    let mut uif = UiFreeze::new();
    let mut s = make_sample(100.5, 92.0, 60.0);
    s.insert("dpc_pct".to_string(), 22.0);
    uif.on_sample(&mut mon, &s, 100.5);
    uif.on_freezes(&mut mon, &done2, &[], 105.0);
    let ev = mon
        .events
        .events
        .iter()
        .find(|e| e.kind == "ui_freeze")
        .expect("ui_freeze should be emitted");
    assert_eq!(ev.details["cause"], json!("dpc_storm"));
}

#[test]
fn test_winapi_and_responsiveness_probe_live() {
    // Command-line tag parser
    assert_eq!(
        winapi::extract_cmdline_tag(r#""C:\Python312\python.exe" -u -m uvicorn app.main:app"#),
        Some("uvicorn".to_string())
    );
    assert_eq!(
        winapi::extract_cmdline_tag(r#"node.exe "C:\projects\web\src\server.js" --port 3000"#),
        Some("src/server.js".to_string())
    );

    // Sleep clocks with 50 ms quantum tolerance
    let (up, unbiased) = winapi::sleep_clocks();
    assert!(up >= unbiased - 0.05 && unbiased > 0.0);

    // DPC and ISR sampling with >= 0.8s window
    let cfg = MonitorConfig::default();
    let mut sm = SystemSampler::new(&cfg);
    std::thread::sleep(Duration::from_millis(800));
    let sample = sm.sample();
    let dpc = get_metric(&sample, "dpc_pct");
    let isr = get_metric(&sample, "isr_pct");
    assert!(dpc.is_finite() && (0.0..=100.0).contains(&dpc));
    assert!(isr.is_finite() && (0.0..=100.0).contains(&isr));

    // ProcessTracker sweep & Deep Attribution
    let mut pt = ProcessTracker::new(30.0);
    let _ = pt.sweep(unix_now());
    std::thread::sleep(Duration::from_millis(100));
    let stats = pt.sweep(unix_now());
    assert!(!stats.is_empty());

    // Live ResponsivenessProbe
    let mut probe = ResponsivenessProbe::new(100, 1000, 500.0);
    probe.start();
    std::thread::sleep(Duration::from_millis(250));
    let snap = probe.snapshot();
    assert_eq!(snap["alive"], json!(true));
    assert!(snap["threads"]["compositor"].is_object());
    let _ = probe.poll(unix_now());
    probe.close();
}

#[test]
fn test_storage_diagnostics_and_cli_commands() {
    let dir = temp_test_dir("storage_diag");
    let mut cfg = MonitorConfig::default();
    cfg.data_dir = dir.clone();
    cfg.ai_provider = "offline".to_string();

    let mut st = Storage::new(&cfg.db_path()).unwrap();
    let now = unix_now();
    for i in 0..10 {
        let mut s = make_sample(now - (10 - i) as f64, 35.0, 82.0);
        s.insert("hard_faults_ps".to_string(), 45.0);
        st.add_sample(&s);
    }
    st.flush();

    let ev_id = st.insert_event(
        now - 5.0,
        now,
        "handle_leak",
        "warning",
        "AcerCentralService.exe holds 2,500,000 handles (+20,000/h)",
        &json!({
            "process": "AcerCentralService.exe",
            "pid": 4120,
            "handles": 2_500_000,
            "handles_per_hour": 20_000,
            "hours_observed": 12.5
        }),
    );
    assert!(ev_id > 0);

    let sum = db_summary(&cfg.db_path());
    assert_eq!(sum["samples"], json!(10));
    assert_eq!(sum["events"], json!(1));

    // Forensic explain
    let ctx = build_event_context(&st, &cfg, Some(ev_id), None).unwrap();
    let rca = offline_explain_event(&ctx, Some("how do I fix this?"));
    assert!(rca.contains("AcerCentralService.exe"));
    assert!(rca.contains("Recommended Actions:"));

    let exp = explain_event(&st, &cfg, None, None, None, true).unwrap();
    assert_eq!(exp["engine"], json!("offline-rca"));

    // Ask
    let ask_res = ask_question(&st, &cfg, "which process is leaking handles?", 24.0, true);
    assert_eq!(ask_res["matched_events"], json!(1));

    // Health audit
    let audit = audit_system_health(&st, &cfg, 24.0);
    assert_eq!(audit["status"], json!("CRITICAL"));

    // CLI commands
    assert_eq!(cmd_status(&cfg), 0);
    assert_eq!(cmd_report(&cfg, 24.0, None, true, false), 0);
    assert_eq!(
        cmd_explain(&cfg, "latest", None, None, true, false, false),
        0
    );
    assert_eq!(cmd_ask(&cfg, "handle leak", 24.0, true, false), 0);
    assert_eq!(cmd_diagnose(&cfg, 24.0, false), 0);

    // InstanceLock test
    let mut lock = InstanceLock::new(cfg.lock_path());
    assert!(lock.acquire().is_ok());
    lock.release();

    // Reset
    drop(st);
    assert_eq!(cmd_reset(&cfg, true), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_replay_and_tune_grid() {
    let mut cfg = MonitorConfig::default();
    cfg.min_training_hours = 0.05; // 18 windows of 10s
    cfg.window_sec = 10;

    let mut windows = Vec::new();
    for i in 0..80 {
        let mut w = make_sample(1_700_000_000.0 + (i as f64) * 10.0, 12.0, 50.0);
        if (50..55).contains(&i) {
            w.insert("cpu_pct".to_string(), 95.0);
            w.insert("core_max_pct".to_string(), 100.0);
        }
        windows.push(w);
    }

    let res = replay(&windows, &cfg, 0);
    assert!(res.scored_windows > 0);
    assert!(res.anomalous_windows >= 3);
    assert!(!res.events.is_empty());

    let grid = tune_grid(&windows, &cfg, &[0.01, 0.05], &[2, 3]);
    assert_eq!(grid.len(), 4);
    // Verify lockstep tune_grid matches independent replay() counts identically
    for row in &grid {
        let mut sub = cfg.clone();
        sub.alert_quantile = row["alert_quantile"].as_f64().unwrap();
        sub.min_consecutive_windows = row["min_consecutive_windows"].as_u64().unwrap() as usize;
        let single = replay(&windows, &sub, 0);
        assert_eq!(
            row["events"].as_u64().unwrap() as usize,
            single.events.len(),
            "tune_grid count must match independent replay for {:?}",
            row
        );
    }
}

#[test]
fn test_dual_schema_v3_v4_diagnostics() {
    let dir = temp_test_dir("v3_compat");
    let mut cfg = MonitorConfig::default();
    cfg.data_dir = dir.clone();
    cfg.ai_provider = "offline".to_string();

    let mut st = Storage::new(&cfg.db_path()).unwrap();
    let now = unix_now();
    // Simulate historical Python v3 upload_burst, memory_pressure, and Optimus ui_freeze events
    let ub_id = st.insert_event(
        now - 40.0,
        now - 10.0,
        "upload_burst",
        "warning",
        "Uploaded 24.4 MB in 30s (peak 2.7 MB/s) — top: NVDisplay.Container.exe",
        &json!({
            "sent_mb": 24.4,
            "peak_mb_s": 2.7,
            "top_net": [{ "name": "NVDisplay.Container.exe", "total_mb": 49.0 }],
            "foreground": "Code.exe",
            "user_driven": false
        }),
    );
    let _mp_id = st.insert_event(
        now - 120.0,
        now - 60.0,
        "memory_pressure",
        "critical",
        "Memory pressure: RAM 72%, commit 97%; largest python.exe (19,259 MB)",
        &json!({
            "peak_ram_pct": 72.0,
            "peak_commit_pct": 97.4,
            "top_memory": [{ "name": "python.exe", "private_mb": 19259 }]
        }),
    );
    let uif_id = st.insert_event(
        now - 180.0,
        now - 178.0,
        "ui_freeze",
        "warning",
        "WhatsApp.Root.exe stopped responding for 2.1 s",
        &json!({
            "process": "WhatsApp.Root.exe",
            "pid": 9001,
            "scope": "app",
            "longest_stall_sec": 2.1,
            "dgpu_transitions": [{ "at_sec": -0.2, "dgpu": "woke up" }],
            "clues": ["dGPU woke up 0.2 s before it began (±1 s)"],
            "around": { "cpu_max_pct": 35, "busiest_core_pct": 90, "ram_max_pct": 70, "commit_max_pct": 85 }
        }),
    );

    let ub_ctx = build_event_context(&st, &cfg, Some(ub_id), None).unwrap();
    let ub_text = offline_explain_event(&ub_ctx, None);
    assert!(ub_text.contains("24.4 MB"), "{}", ub_text);
    assert!(ub_text.contains("NVDisplay.Container.exe"), "{}", ub_text);

    let uif_ctx = build_event_context(&st, &cfg, Some(uif_id), None).unwrap();
    let uif_text = offline_explain_event(&uif_ctx, None);
    assert!(uif_text.contains("dgpu_wake_latency"), "{}", uif_text);

    let audit = audit_system_health(&st, &cfg, 24.0);
    let rep = audit["report"].as_str().unwrap();
    assert!(rep.contains("97.4%"), "expected v3 peak commit in report: {}", rep);

    drop(st);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_native_rust_avx2_ecoqos_nvml_and_throttle_benefits() {
    // 1. Verify runtime AVX2 SIMD sq_dist matches scalar reference across random vectors
    let a: [f32; NUM_FEATURES] = [
        1.5, 4.0, 12.25, 18.75, 0.3, 1.1, 2.2, 0.9, 55.0, 42.5, 68.0, 31.2, 4.5, 1.0, 1.0, 0.5,
    ];
    let b: [f32; NUM_FEATURES] = [
        0.5, 2.0, 10.25, 15.0, 0.1, 0.8, 1.0, 0.2, 10.0, 20.5, 50.0, 15.0, 1.5, 0.0, 1.0, 1.0,
    ];
    let d_scalar = sq_dist_scalar(&a, &b);
    let d_simd = sq_dist(&a, &b, true);
    assert!(
        (d_scalar - d_simd).abs() < 1e-3,
        "scalar={} vs simd={}",
        d_scalar,
        d_simd
    );

    // 2. Verify Windows EcoQoS background priority + throttling setter executes cleanly
    assert!(winapi::set_current_thread_eco_qos());

    // 3. Verify NVML asleep state keeps temperature unknown (NaN) and load/power zeroed
    let asleep = winapi::NvmlTelemetry::asleep();
    assert_eq!(asleep.gpu_awake, 0.0);
    assert_eq!(asleep.gpu_util_pct, 0.0);
    assert!(asleep.gpu_temp_c.is_nan());

    // 4. Verify GpuThermal filters idle thermal flag quirks (<65C) but flags real throttle (>=65C)
    let mut cfg = MonitorConfig::default();
    cfg.gpu_temp_sustain_sec = 5.0;
    let mut mon = ReplayMonitor::new(cfg);
    let mut gt = GpuThermal::default();
    let mut cool_quirk = make_sample(100.0, 5.0, 40.0);
    cool_quirk.insert("gpu_awake".to_string(), 1.0);
    cool_quirk.insert("gpu_temp_c".to_string(), 42.0);
    cool_quirk.insert("gpu_throttle".to_string(), 0x20 as f64); // SW_THERMAL at idle temp
    gt.on_sample(&mut mon, &cool_quirk, 100.0);
    gt.on_sample(&mut mon, &cool_quirk, 110.0);
    assert!(mon.events.events.is_empty());

    let mut hot_throttle = make_sample(200.0, 10.0, 50.0);
    hot_throttle.insert("gpu_awake".to_string(), 1.0);
    hot_throttle.insert("gpu_temp_c".to_string(), 72.0);
    hot_throttle.insert("gpu_throttle".to_string(), 0x20 as f64);
    gt.on_sample(&mut mon, &hot_throttle, 200.0);
    gt.on_sample(&mut mon, &hot_throttle, 210.0);
    assert!(mon.events.events.iter().any(|e| e.kind == "gpu_thermal"));

    // 5. Verify ghost-window check & cmdline_attribution toggle
    assert!(winapi::real_window_from_ghost(0).is_none());
    let mut pt_no_cmd = ProcessTracker::new(30.0);
    pt_no_cmd.cmdline_attribution = false;
    let _ = pt_no_cmd.sweep(unix_now());
}
