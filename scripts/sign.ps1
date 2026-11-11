<#
.SYNOPSIS
    Signs target\release\resmon.exe and target\release\resmonw.exe with a local SHA-256
    Authenticode code-signing certificate in Cert:\CurrentUser\My (with optional RFC 3161
    timestamping, public .cer export, and opt-in local trust store import).
#>
param(
    [string]$Subject = "CN=resmon Local Publisher",
    [string]$TimestampUrl = "http://timestamp.digicert.com",
    [switch]$TrustLocal
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$releaseDir = Join-Path $repoRoot "target\release"
$bins = @(
    (Join-Path $releaseDir "resmon.exe"),
    (Join-Path $releaseDir "resmonw.exe")
)

$signtool = Get-ChildItem "C:\Program Files (x86)\Windows Kits\10\bin" -Recurse -Filter "signtool.exe" -ErrorAction SilentlyContinue |
    Where-Object { $_.Directory.Name -eq "x64" } |
    Sort-Object FullName -Descending |
    Select-Object -First 1 -ExpandProperty FullName

if (-not $signtool) {
    throw "Windows SDK signtool.exe (x64) not found."
}

$cert = Get-ChildItem "Cert:\CurrentUser\My" -CodeSigningCert -ErrorAction SilentlyContinue |
    Where-Object { $_.Subject -eq $Subject -and $_.NotAfter -gt (Get-Date).AddDays(7) } |
    Sort-Object NotAfter -Descending |
    Select-Object -First 1

if (-not $cert) {
    $cert = New-SelfSignedCertificate `
        -Type CodeSigningCert `
        -Subject $Subject `
        -CertStoreLocation "Cert:\CurrentUser\My" `
        -NotAfter (Get-Date).AddYears(5) `
        -KeyAlgorithm RSA `
        -KeyLength 2048 `
        -HashAlgorithm SHA256
    Write-Host "Created new CodeSigning cert: $($cert.Thumbprint) ($Subject)"
} else {
    Write-Host "Using existing CodeSigning cert: $($cert.Thumbprint)"
}

New-Item -ItemType Directory -Force -Path $releaseDir | Out-Null
$pubCer = Join-Path $releaseDir "resmon-local-pub.cer"
Export-Certificate -Cert $cert -FilePath $pubCer | Out-Null
Write-Host "Exported public certificate (no private key): $pubCer"

if ($TrustLocal) {
    Import-Certificate -FilePath $pubCer -CertStoreLocation "Cert:\CurrentUser\Root" | Out-Null
    Import-Certificate -FilePath $pubCer -CertStoreLocation "Cert:\CurrentUser\TrustedPublisher" | Out-Null
    Write-Host "Imported public cert into Cert:\CurrentUser\Root and Cert:\CurrentUser\TrustedPublisher."
}

foreach ($bin in $bins) {
    if (-not (Test-Path -LiteralPath $bin)) {
        Write-Warning "Skipping missing binary: $bin"
        continue
    }

    $signed = $false
    if ($TimestampUrl) {
        & $signtool sign /fd SHA256 /tr $TimestampUrl /td SHA256 /sha1 $cert.Thumbprint /s My /d "resmon v4.0 Windows Anomaly Monitor" $bin
        if ($LASTEXITCODE -eq 0) {
            $signed = $true
        } else {
            Write-Warning "Timestamp signing via $TimestampUrl failed for $bin (exit $LASTEXITCODE); retrying without timestamp."
        }
    }
    if (-not $signed) {
        & $signtool sign /fd SHA256 /sha1 $cert.Thumbprint /s My /d "resmon v4.0 Windows Anomaly Monitor" $bin
        if ($LASTEXITCODE -ne 0) {
            throw "signtool failed for $bin (exit $LASTEXITCODE) — stop any running instance first."
        }
    }
}

$presentBins = $bins | Where-Object { Test-Path -LiteralPath $_ }
if ($presentBins) {
    Get-AuthenticodeSignature -LiteralPath $presentBins |
        Format-Table Path, Status, @{n='Signer';e={$_.SignerCertificate.Subject}}, @{n='Timestamp';e={$_.TimeStamperCertificate.Subject}} -AutoSize
    Get-FileHash -LiteralPath $presentBins -Algorithm SHA256 | Format-Table Path, Hash -AutoSize
}
