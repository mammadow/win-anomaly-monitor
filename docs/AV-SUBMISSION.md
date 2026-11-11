# Antivirus Exclusions & False-Positive Submission Guide (`resmon` v4.0)

`resmon` is a 100% user-mode, MIT-licensed Windows 11 system performance & anomaly monitor built in pure Rust ([source repository](https://github.com/mammadow/win-anomaly-monitor)). Because locally compiled or self-signed binaries have no global commercial CA reputation and inspect process command lines and window responsiveness legitimately, aggressive behavioral shields (such as **Bitdefender Advanced Threat Defense**) can flag them unless excluded locally or cleared upstream.

---

## 1. Bitdefender Total Security Configuration (Standard User vs. Developer)

### Do you still need exclusions as a Standard User (`asInvoker`)?
**Yes.** Running via `shell:startup` (`asInvoker`, non-elevated) avoids Admin/UAC privilege escalation risk and lowers behavioral suspicion compared to elevated execution, **but** `resmonw.exe` still:
1. Launches automatically at Windows logon (`Resmon.lnk`),
2. Opens read-only `PROCESS_QUERY_LIMITED_INFORMATION` handles (`NtQueryInformationProcess(ProcessCommandLineInformation = 60)`) to tag scripting runtimes (`python.exe [script.py]`, `node.exe [...]`), and
3. Sends `SendMessageTimeoutW(WM_NULL)` responsiveness pings every `250 ms` to foreground windows and the taskbar.

Because self-signed local builds lack a commercial EV/OV CA signature, **Bitdefender Antivirus** and **Advanced Threat Defense (ATD)** will still evaluate those calls heuristically unless whitelisted.

### Required Bitdefender Exceptions
Open **Bitdefender Total Security → Protection → Antivirus → Open → Settings → Manage Exceptions → + Add an Exception**, and turn **ON** both **Antivirus** and **Advanced Threat Defense** (plus **Online Threat Prevention** if shown) for:

| Path | Who Needs It | Why |
|---|---|---|
| `<repo-dir>\` *(e.g. your cloned repo folder)* | **Everyone running from repo** | Protects `target\release\resmonw.exe`, `resmon.exe`, `metrics.db`, `state.json`, and `events.jsonl` |
| `%USERPROFILE%\.cargo\` | **Only if building/testing locally** | Prevents quarantine of temporary `cargo test` / `cargo build` binaries |
| `%USERPROFILE%\.rustup\` | **Only if building/testing locally** | Prevents interference with `rustc.exe` / linker toolchains |
| `C:\Program Files\resmon\` | **Only if installed there for Admin mode** | Protects deployed `.exe` binaries outside the repo |

Also check:
- **Protection → Advanced Threat Defense → Settings → Manage Exceptions:** Verify `target\release\resmonw.exe` and `target\release\resmon.exe` (or `Program Files\resmon\*.exe`) are covered.
- **Protection → Firewall → View application rules:** Confirm no leftover containment/block rule exists on `resmon.exe` / `resmonw.exe` / `cargo.exe` from an earlier disinfection event.

---

## 2. Submitting False Positives Upstream

To request upstream definition whitelisting so exclusions are eventually unnecessary on release binaries:

- **Bitdefender Malware Research Submission Form:**
  https://www.bitdefender.com/en-us/business/submit
  *(Consumer false-positive guidance: https://www.bitdefender.com/consumer/support/answer/29358/ — Bitdefender recommends putting flagged samples into a `.zip` archive if real-time AV deletes them before upload.)*
- **Microsoft Security Intelligence (Defender / SmartScreen):**
  https://www.microsoft.com/en-us/wdsi/filesubmission *(Select "Incorrectly detected as malware (false positive)", Microsoft account sign-in required).*

### Ready-to-Paste Submission Description

```text
Software: resmon v4.0.0 — Windows 11 Resource & Anomaly Monitor (resmon.exe / resmonw.exe)
License & Source: MIT Open Source — https://github.com/mammadow/win-anomaly-monitor
Release Page: https://github.com/mammadow/win-anomaly-monitor/releases/tag/v4.0.0
Classification: False Positive (legitimate system monitoring & telemetry tool)

Explanation of flagged behaviors:
1. Single-binary pure Rust Windows 11 CLI (resmon.exe) and windowless tray daemon (resmonw.exe).
2. Calls NtQuerySystemInformation, GetIfTable2, PDH counters, CallNtPowerInformation (powrprof.dll),
   and System32\nvml.dll solely to monitor CPU/RAM/disk/net/battery/GPU metrics.
3. Queries OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION) + NtQueryInformationProcess(60)
   to label script hosts (e.g. "python.exe [train.py]") for diagnostic reporting (can be disabled
   via `cmdline_attribution = false` in config.toml).
4. Sends non-destructive WM_NULL via SendMessageTimeoutW to detect frozen UI windows.
5. Uses SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32) and spawns zero hidden PowerShell/cmd shells.
```

---

## 3. Reference SHA-256 Digests (`v4.0.0`)

### GitHub Release [`v4.0.0`](https://github.com/mammadow/win-anomaly-monitor/releases/tag/v4.0.0) Binaries
- `resmon-v4.0.0-windows-x86_64.zip`: `446b8b281ade9f10f7b0b41776d83ac4a063b3eeb43db894f1ba9cd066e5b125`
- `resmon.exe`: `d5c7347a37dbf2d4bf7a3da823dba9f5b1f69e0e43027ed30234a5f82166998c`
- `resmonw.exe`: `560843bf804689b5f1290e65a860e5eaad18eb8f2eaaf15230a0142cf33afe65`
- `SHA256SUMS.txt`: `fac026cb1825ae7d54234906814f918e6a75e6405274d423faf638de3bed0eb1`
