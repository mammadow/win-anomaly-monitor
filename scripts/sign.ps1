<#
.SYNOPSIS
    Signs target\release\resmon.exe and target\release\resmonw.exe with a local SHA-256
    Authenticode code-signing certificate in Cert:\CurrentUser\My.
#>
param(
    [string]$Subject = "CN=resmon Local Publisher"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$bins = @(
    (Join-Path $repoRoot "target\release\resmon.exe"),
    (Join-Path $repoRoot "target\release\resmonw.exe")
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

foreach ($bin in $bins) {
    if (-not (Test-Path -LiteralPath $bin)) {
        Write-Warning "Skipping missing binary: $bin"
        continue
    }
    & $signtool sign /fd SHA256 /sha1 $cert.Thumbprint /s My /d "resmon v4.0 Windows Anomaly Monitor" $bin
    if ($LASTEXITCODE -ne 0) {
        throw "signtool failed for $bin (exit $LASTEXITCODE)"
    }
}

Get-AuthenticodeSignature -LiteralPath $bins | Format-Table Path, Status, @{n='Signer';e={$_.SignerCertificate.Subject}} -AutoSize
