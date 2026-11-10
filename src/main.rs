//! Entry point for `resmon` v4.0 CLI and background daemon.

use clap::Parser;
use resmon::cli::{run_cli, CliArgs};

fn main() {
    let args = CliArgs::parse();
    let code = run_cli(args);
    if code != 0 {
        std::process::exit(code);
    }
}
