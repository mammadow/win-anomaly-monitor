//! `resmon` v4.0 — Pure Rust Windows 11 Resource & Anomaly Monitor.

pub mod cli;
pub mod config;
pub mod detector;
pub mod diagnostics;
pub mod engine;
pub mod events;
pub mod features;
pub mod processes;
pub mod replay;
pub mod responsiveness;
pub mod rules;
pub mod sampler;
pub mod storage;
pub mod tray;
pub mod tui;
pub mod winapi;
