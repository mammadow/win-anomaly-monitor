//! Windowless Windows GUI subsystem entry point (`resmonw.exe`) for background
//! monitoring and system tray startup (`shell:startup`) without a console window.

#![windows_subsystem = "windows"]

use clap::Parser;
use resmon::cli::{run_cli, CliArgs, Commands};
use std::io::Write;
use std::path::Path;

fn log_startup_error(msg: &str) {
    let log_path = Path::new("data").join("monitor.log");
    if let Some(parent) = log_path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
    {
        let ts = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
        let _ = writeln!(f, "{} [ERROR] [resmonw] {}", ts, msg);
    }
}

fn main() {
    let mut args = CliArgs::parse();
    if args.command.is_none() {
        args.command = Some(Commands::Run { tray: true });
    }
    let code = run_cli(args);
    if code != 0 {
        log_startup_error(&format!("resmonw exited with non-zero code {}", code));
        std::process::exit(code);
    }
}
