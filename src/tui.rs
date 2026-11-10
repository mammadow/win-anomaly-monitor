//! Interactive live terminal dashboard (`resmon top`) built with `ratatui` + `crossterm`.

use crate::config::MonitorConfig;
use crate::diagnostics::{build_event_context, offline_explain_event};
use crate::engine::read_state_file;
use crate::features::{fmt_bytes, get_metric, get_metric_or, MetricMap};
use crate::processes::ProcessTracker;
use crate::sampler::{unix_now, SystemSampler};
use crate::storage::{EventRecord, Storage};
use chrono::{Local, TimeZone};
use crossterm::event::{self, Event as CEvent, KeyCode, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Sparkline, Table, Wrap,
};
use ratatui::Terminal;
use serde_json::Value;
use std::collections::VecDeque;
use std::io;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, PartialEq, Eq)]
enum ProcView {
    CpuMemory,
    IoFaults,
}

struct DashboardState {
    cpu_hist: VecDeque<u64>,
    commit_hist: VecDeque<u64>,
    view: ProcView,
    selected_event: usize,
    modal_text: Option<String>,
    modal_scroll: u16,
}

impl DashboardState {
    fn new() -> Self {
        Self {
            cpu_hist: VecDeque::with_capacity(64),
            commit_hist: VecDeque::with_capacity(64),
            view: ProcView::CpuMemory,
            selected_event: 0,
            modal_text: None,
            modal_scroll: 0,
        }
    }

    fn push_sample(&mut self, s: &MetricMap) {
        let push = |buf: &mut VecDeque<u64>, v: f64| {
            if buf.len() >= 60 {
                buf.pop_front();
            }
            buf.push_back(v.clamp(0.0, 100.0).round() as u64);
        };
        push(&mut self.cpu_hist, get_metric_or(s, "cpu_pct", 0.0));
        push(&mut self.commit_hist, get_metric_or(s, "commit_pct", 0.0));
    }
}

pub fn run_top(cfg: &MonitorConfig) -> Result<(), String> {
    enable_raw_mode().map_err(|e| format!("Terminal raw mode error: {}", e))?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).map_err(|e| e.to_string())?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).map_err(|e| e.to_string())?;

    let res = top_loop(&mut terminal, cfg);

    let _ = disable_raw_mode();
    let _ = execute!(terminal.backend_mut(), LeaveAlternateScreen);
    let _ = terminal.show_cursor();
    res
}

fn top_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    cfg: &MonitorConfig,
) -> Result<(), String> {
    let mut sampler = SystemSampler::new(cfg);
    let mut procs = ProcessTracker::new(30.0);
    procs.cmdline_attribution = cfg.cmdline_attribution;
    let storage = Storage::open_readonly(&cfg.db_path()).ok();

    let _ = procs.sweep(unix_now());
    std::thread::sleep(Duration::from_millis(250));

    let mut sample = sampler.sample();
    let _ = procs.sweep(unix_now());
    let mut ui_state = DashboardState::new();
    ui_state.push_sample(&sample);
    let mut daemon_state = read_state_file(&cfg.state_path());
    let mut last_refresh = Instant::now();

    loop {
        let now = unix_now();
        let (top_cpu, top_io) = procs.top_activity(now - 10.0, 10);
        let top_mem = procs.top_memory(10);
        let events = storage
            .as_ref()
            .map(|st| st.query_events(now - 24.0 * 3600.0, None, 12))
            .unwrap_or_default();
        if !events.is_empty() && ui_state.selected_event >= events.len() {
            ui_state.selected_event = events.len() - 1;
        }

        terminal
            .draw(|f| {
                draw_dashboard(
                    f,
                    &sample,
                    &top_cpu,
                    &top_io,
                    &top_mem,
                    &procs.latest,
                    &events,
                    daemon_state.as_ref(),
                    &ui_state,
                )
            })
            .map_err(|e| e.to_string())?;

        let timeout = Duration::from_millis(1000)
            .checked_sub(last_refresh.elapsed())
            .unwrap_or(Duration::from_millis(50));

        if event::poll(timeout).map_err(|e| e.to_string())? {
            if let CEvent::Key(key) = event::read().map_err(|e| e.to_string())? {
                if ui_state.modal_text.is_some() {
                    match key.code {
                        KeyCode::Esc | KeyCode::Enter | KeyCode::Char('e') | KeyCode::Char('E') => {
                            ui_state.modal_text = None;
                            ui_state.modal_scroll = 0;
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            ui_state.modal_scroll = ui_state.modal_scroll.saturating_add(1);
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            ui_state.modal_scroll = ui_state.modal_scroll.saturating_sub(1);
                        }
                        KeyCode::Char('q') | KeyCode::Char('Q') => break,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break
                        }
                        _ => {}
                    }
                } else {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Esc => break,
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            break
                        }
                        KeyCode::Tab | KeyCode::Char('i') | KeyCode::Char('I') => {
                            ui_state.view = match ui_state.view {
                                ProcView::CpuMemory => ProcView::IoFaults,
                                ProcView::IoFaults => ProcView::CpuMemory,
                            };
                        }
                        KeyCode::Down | KeyCode::Char('j') => {
                            if ui_state.selected_event + 1 < events.len() {
                                ui_state.selected_event += 1;
                            }
                        }
                        KeyCode::Up | KeyCode::Char('k') => {
                            ui_state.selected_event = ui_state.selected_event.saturating_sub(1);
                        }
                        KeyCode::Enter | KeyCode::Char('e') | KeyCode::Char('E') => {
                            if let (Some(st), Some(ev)) =
                                (storage.as_ref(), events.get(ui_state.selected_event))
                            {
                                let text = build_event_context(st, cfg, Some(ev.id), None)
                                    .map(|ctx| offline_explain_event(&ctx, None))
                                    .unwrap_or_else(|| {
                                        "No forensic context available.".to_string()
                                    });
                                ui_state.modal_text = Some(text);
                                ui_state.modal_scroll = 0;
                            }
                        }
                        KeyCode::Char('r') | KeyCode::Char('R') => {
                            sample = sampler.sample();
                            let _ = procs.sweep(unix_now());
                            ui_state.push_sample(&sample);
                            daemon_state = read_state_file(&cfg.state_path());
                            last_refresh = Instant::now();
                        }
                        _ => {}
                    }
                }
            }
        }

        if last_refresh.elapsed() >= Duration::from_millis(1000) {
            sample = sampler.sample();
            let _ = procs.sweep(unix_now());
            ui_state.push_sample(&sample);
            daemon_state = read_state_file(&cfg.state_path());
            last_refresh = Instant::now();
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn draw_dashboard(
    f: &mut ratatui::Frame<'_>,
    s: &MetricMap,
    top_cpu: &[Value],
    top_io: &[Value],
    top_mem: &[Value],
    latest_procs: &[crate::processes::ProcStats],
    events: &[EventRecord],
    daemon_state: Option<&Value>,
    ui: &DashboardState,
) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(0)
        .constraints([
            Constraint::Length(4),
            Constraint::Length(5),
            Constraint::Min(9),
            Constraint::Length(10),
        ])
        .split(f.area());

    // 1. Header + Daemon UI Probe & ML Status
    let dpc = get_metric_or(s, "dpc_pct", 0.0);
    let isr = get_metric_or(s, "isr_pct", 0.0);
    let hf = get_metric_or(s, "hard_faults_ps", 0.0);
    let dr = get_metric_or(s, "disk_read_bps", 0.0);
    let dw = get_metric_or(s, "disk_write_bps", 0.0);
    let ns = get_metric_or(s, "net_sent_bps", 0.0);
    let nr = get_metric_or(s, "net_recv_bps", 0.0);
    let free_gb = get_metric_or(s, "disk_free_gb", 0.0);
    let on_ac = if get_metric_or(s, "on_ac", 1.0) >= 0.5 {
        "AC"
    } else {
        "Battery"
    };
    let gpu_awake = get_metric(s, "gpu_awake");
    let gpu_str = if !gpu_awake.is_finite() {
        "GPU: absent".to_string()
    } else if gpu_awake < 0.5 {
        "GPU: asleep (D3cold)".to_string()
    } else {
        format!(
            "GPU: {:.0}% | {:.0}°C | {:.0}W | VRAM {:.0}% | {:.0}MHz",
            get_metric_or(s, "gpu_util_pct", 0.0),
            get_metric_or(s, "gpu_temp_c", 0.0),
            get_metric_or(s, "gpu_power_w", 0.0),
            get_metric_or(s, "gpu_vram_pct", 0.0),
            get_metric_or(s, "gpu_clock_mhz", 0.0)
        )
    };

    let (probe_str, model_str) = format_daemon_badges(daemon_state);

    let line1 = Line::from(vec![
        Span::styled(
            " resmon v4.0 (Rust) ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(format!(
            " DPC+ISR: {:.1}% | Faults: {:.0}/s | Disk: {} R / {} W | Net: {} ↓ / {} ↑ | Free: {:.1} GB ({})",
            dpc + isr,
            hf,
            fmt_bytes(dr, true),
            fmt_bytes(dw, true),
            fmt_bytes(nr, true),
            fmt_bytes(ns, true),
            free_gb,
            on_ac
        )),
    ]);
    let line2 = Line::from(vec![
        Span::styled(
            format!(" {} ", gpu_str),
            Style::default().fg(Color::Magenta),
        ),
        Span::raw(" | "),
        Span::styled(probe_str, Style::default().fg(Color::Green)),
        Span::raw(" | "),
        Span::styled(model_str, Style::default().fg(Color::Yellow)),
        Span::raw("  [Tab/i:CPU<->IO  Up/Down+Enter:Explain  q:Quit]"),
    ]);
    let header_block = Paragraph::new(vec![line1, line2]).block(
        Block::default()
            .borders(Borders::ALL)
            .title("Live Telemetry & HighQoS Probe"),
    );
    f.render_widget(header_block, chunks[0]);

    // 2. Gauges + 60s Sparklines
    let row2_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(17),
            Constraint::Percentage(17),
            Constraint::Percentage(17),
            Constraint::Percentage(17),
            Constraint::Percentage(16),
            Constraint::Percentage(16),
        ])
        .split(chunks[1]);

    render_gauge(
        f,
        row2_cols[0],
        "CPU Total",
        get_metric_or(s, "cpu_pct", 0.0),
        85.0,
    );
    render_gauge(
        f,
        row2_cols[1],
        "Busiest Core",
        get_metric_or(s, "core_max_pct", 0.0),
        95.0,
    );
    render_gauge(
        f,
        row2_cols[2],
        "Physical RAM",
        get_metric_or(s, "ram_pct", 0.0),
        90.0,
    );
    render_gauge(
        f,
        row2_cols[3],
        "Commit Charge",
        get_metric_or(s, "commit_pct", 0.0),
        90.0,
    );

    let cpu_slice: Vec<u64> = ui.cpu_hist.iter().copied().collect();
    let com_slice: Vec<u64> = ui.commit_hist.iter().copied().collect();
    let cpu_spark = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title("CPU 60s"))
        .data(&cpu_slice)
        .max(100)
        .style(Style::default().fg(Color::Cyan));
    f.render_widget(cpu_spark, row2_cols[4]);

    let com_spark = Sparkline::default()
        .block(Block::default().borders(Borders::ALL).title("Commit 60s"))
        .data(&com_slice)
        .max(100)
        .style(Style::default().fg(Color::Yellow));
    f.render_widget(com_spark, row2_cols[5]);

    // 3. Process Attribution Tables (CPU/Memory OR I/O & Page Faults)
    let mid_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(52), Constraint::Percentage(48)])
        .split(chunks[2]);

    match ui.view {
        ProcView::CpuMemory => {
            render_cpu_table(f, mid_cols[0], top_cpu);
            render_memory_table(f, mid_cols[1], top_mem);
        }
        ProcView::IoFaults => {
            render_io_table(f, mid_cols[0], top_io);
            render_faults_table(f, mid_cols[1], latest_procs);
        }
    }

    // 4. Recent Events (Selectable)
    let ev_rows: Vec<Row<'_>> = events
        .iter()
        .enumerate()
        .map(|(idx, e)| {
            let t_str = Local
                .timestamp_opt(e.ts as i64, 0)
                .single()
                .map(|dt| dt.format("%H:%M:%S").to_string())
                .unwrap_or_default();
            let sev_color = match e.severity.as_str() {
                "critical" => Color::Red,
                "warning" => Color::Yellow,
                _ => Color::Cyan,
            };
            let prefix = if idx == ui.selected_event { ">" } else { " " };
            let row_style = if idx == ui.selected_event {
                Style::default()
                    .bg(Color::DarkGray)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(format!("{}#{}", prefix, e.id)),
                Cell::from(t_str),
                Cell::from(e.severity.clone()).style(Style::default().fg(sev_color)),
                Cell::from(e.kind.clone()),
                Cell::from(e.title.clone()),
            ])
            .style(row_style)
        })
        .collect();

    let ev_table = Table::new(
        ev_rows,
        [
            Constraint::Length(8),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Length(18),
            Constraint::Min(30),
        ],
    )
    .header(
        Row::new(vec![
            "ID",
            "Time",
            "Severity",
            "Kind",
            "Headline (Enter to Explain)",
        ])
        .style(
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Recent Events (last 24h)"),
    );
    f.render_widget(ev_table, chunks[3]);

    // 5. Optional In-TUI RCA Explain Modal Overlay
    if let Some(ref text) = ui.modal_text {
        let popup = centered_rect(86, 78, f.area());
        f.render_widget(Clear, popup);
        let para = Paragraph::new(text.as_str())
            .wrap(Wrap { trim: false })
            .scroll((ui.modal_scroll, 0))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .title("Offline Root-Cause Analysis (Up/Down: Scroll | Esc/Enter: Close)")
                    .border_style(Style::default().fg(Color::Cyan)),
            );
        f.render_widget(para, popup);
    }
}

fn format_daemon_badges(daemon_state: Option<&Value>) -> (String, String) {
    let Some(st) = daemon_state else {
        return (
            "UI ping: daemon offline".to_string(),
            "ML: waiting".to_string(),
        );
    };
    let alive = !st["stopped"].as_bool().unwrap_or(false)
        && (unix_now() - st["updated_at"].as_f64().unwrap_or(0.0)).abs() < 60.0;
    let probe = &st["responsiveness"];
    let ui_str = if alive && probe["alive"].as_bool().unwrap_or(false) {
        let fmt_thr = |k: &str, label: &str| {
            let t = &probe["threads"][k];
            let ms = t["latency_ms"].as_f64().unwrap_or(0.0);
            if t["hung"].as_bool().unwrap_or(false) {
                format!("{} {:.0}ms HUNG", label, ms)
            } else {
                format!("{} {:.0}ms", label, ms)
            }
        };
        format!(
            "UI ping: {} | {} | {}",
            fmt_thr("foreground", "FG"),
            fmt_thr("taskbar", "Bar"),
            fmt_thr("compositor", "DWM")
        )
    } else if alive {
        "UI ping: active".to_string()
    } else {
        "UI ping: daemon stopped".to_string()
    };

    let m = &st["model"];
    let ml_str = if m["ready"].as_bool().unwrap_or(false) {
        format!(
            "ML: {} win (thr {:.2})",
            m["windows"].as_i64().unwrap_or(0),
            m["threshold"].as_f64().unwrap_or(0.0)
        )
    } else {
        "ML: baseline/training".to_string()
    };
    (ui_str, ml_str)
}

fn render_cpu_table(f: &mut ratatui::Frame<'_>, area: Rect, top_cpu: &[Value]) {
    let rows: Vec<Row<'_>> = top_cpu
        .iter()
        .map(|p| {
            Row::new(vec![
                Cell::from(p["name"].as_str().unwrap_or("?").to_string()),
                Cell::from(p["pid"].as_u64().unwrap_or(0).to_string()),
                Cell::from(format!("{:.2}", p["cores"].as_f64().unwrap_or(0.0))),
                Cell::from(format!("{:.1}%", p["cpu_pct"].as_f64().unwrap_or(0.0))),
                Cell::from(format!("{} MB", p["private_mb"].as_i64().unwrap_or(0))),
            ])
        })
        .collect();
    let tbl = Table::new(
        rows,
        [
            Constraint::Percentage(48),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
            Constraint::Percentage(13),
        ],
    )
    .header(
        Row::new(vec![
            "Process [Script / Service]",
            "PID",
            "Cores",
            "CPU%",
            "Priv MB",
        ])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Top CPU Processes (Deep Attribution)"),
    );
    f.render_widget(tbl, area);
}

fn render_memory_table(f: &mut ratatui::Frame<'_>, area: Rect, top_mem: &[Value]) {
    let rows: Vec<Row<'_>> = top_mem
        .iter()
        .map(|p| {
            let handles = p["handles"].as_u64().unwrap_or(0);
            let h_style = if handles >= 50_000 {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(p["name"].as_str().unwrap_or("?").to_string()),
                Cell::from(format!("{} MB", p["private_mb"].as_i64().unwrap_or(0))),
                Cell::from(format!("{} MB", p["working_set_mb"].as_i64().unwrap_or(0))),
                Cell::from(p["processes"].as_u64().unwrap_or(1).to_string()),
                Cell::from(format!("{}", handles)).style(h_style),
            ])
        })
        .collect();
    let tbl = Table::new(
        rows,
        [
            Constraint::Percentage(44),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(10),
            Constraint::Percentage(16),
        ],
    )
    .header(
        Row::new(vec![
            "Process [Script / Service]",
            "Private",
            "WorkSet",
            "Cnt",
            "Handles",
        ])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Top Memory & Handle Consumers"),
    );
    f.render_widget(tbl, area);
}

fn render_io_table(f: &mut ratatui::Frame<'_>, area: Rect, top_io: &[Value]) {
    let rows: Vec<Row<'_>> = top_io
        .iter()
        .map(|p| {
            Row::new(vec![
                Cell::from(p["name"].as_str().unwrap_or("?").to_string()),
                Cell::from(p["pid"].as_u64().unwrap_or(0).to_string()),
                Cell::from(format!(
                    "{:.2} MB/s",
                    p["read_mb_s"].as_f64().unwrap_or(0.0)
                )),
                Cell::from(format!(
                    "{:.2} MB/s",
                    p["write_mb_s"].as_f64().unwrap_or(0.0)
                )),
                Cell::from(format!(
                    "{:.2} MB/s",
                    p["other_mb_s"].as_f64().unwrap_or(0.0)
                )),
            ])
        })
        .collect();
    let tbl = Table::new(
        rows,
        [
            Constraint::Percentage(44),
            Constraint::Percentage(12),
            Constraint::Percentage(15),
            Constraint::Percentage(15),
            Constraint::Percentage(14),
        ],
    )
    .header(
        Row::new(vec![
            "Process [Script / Service]",
            "PID",
            "Read",
            "Write",
            "Net/Other",
        ])
        .style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Top Disk & Socket/Other I/O"),
    );
    f.render_widget(tbl, area);
}

fn render_faults_table(
    f: &mut ratatui::Frame<'_>,
    area: Rect,
    latest_procs: &[crate::processes::ProcStats],
) {
    let mut ranked: Vec<&crate::processes::ProcStats> = latest_procs
        .iter()
        .filter(|p| p.hard_faults_ps >= 1.0 || p.handles >= 5_000)
        .collect();
    ranked.sort_by(|a, b| {
        b.hard_faults_ps
            .partial_cmp(&a.hard_faults_ps)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.handles.cmp(&a.handles))
    });
    let rows: Vec<Row<'_>> = ranked
        .into_iter()
        .take(10)
        .map(|p| {
            let h_style = if p.handles >= 50_000 {
                Style::default().fg(Color::Red).add_modifier(Modifier::BOLD)
            } else {
                Style::default()
            };
            Row::new(vec![
                Cell::from(p.name.clone()),
                Cell::from(p.pid().to_string()),
                Cell::from(format!("{:.0}/s", p.hard_faults_ps)),
                Cell::from(format!("{} MB", p.private_bytes / 1_048_576)),
                Cell::from(format!("{}", p.handles)).style(h_style),
            ])
        })
        .collect();
    let tbl = Table::new(
        rows,
        [
            Constraint::Percentage(44),
            Constraint::Percentage(12),
            Constraint::Percentage(14),
            Constraint::Percentage(14),
            Constraint::Percentage(16),
        ],
    )
    .header(
        Row::new(vec!["Process", "PID", "Faults/s", "Private", "Handles"]).style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
    )
    .block(
        Block::default()
            .borders(Borders::ALL)
            .title("Hard Page Faults & High-Handle Processes"),
    );
    f.render_widget(tbl, area);
}

fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let vert = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);
    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vert[1])[1]
}

fn render_gauge(f: &mut ratatui::Frame<'_>, area: Rect, title: &str, pct: f64, warn_at: f64) {
    let clamped = pct.clamp(0.0, 100.0);
    let color = if clamped >= warn_at {
        Color::Red
    } else if clamped >= warn_at - 15.0 {
        Color::Yellow
    } else {
        Color::Green
    };
    let g = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(color))
        .percent(clamped.round() as u16)
        .label(format!("{:.1}%", clamped));
    f.render_widget(g, area);
}
