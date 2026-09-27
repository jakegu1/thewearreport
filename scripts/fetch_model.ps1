# Download the YOLOX ONNX models (Apache-2.0) from the official YOLOX release and verify
# each against a pinned SHA-256: YOLOX-s and YOLOX-m (the spot-check's default model).
# The Windows counterpart of fetch_model.sh, with the same URLs and the same pins.
#
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_model.ps1 [-Dest DIR]
#        (DIR defaults to .models\ in the repository)
#
# Fails closed. First every model file in DIR that does not match its pin is deleted, and
# so are the part files (DIR\.<name>.XXXXXX, six characters after the dot) that a killed
# run can leave. Then each missing model is downloaded to a part file, which is moved
# into place only after its checksum matches. On any failure the script exits 1 and
# leaves no unverified file behind: every part file it made is deleted on every exit.
# Run one fetch per DIR at a time.
#
# Works in Windows PowerShell 5.1 and PowerShell 7. The SHA-256 comes from .NET
# (System.Security.Cryptography.SHA256), not Get-FileHash: Windows PowerShell started
# from PowerShell 7 inherits its module path and then cannot load Get-FileHash.
[CmdletBinding()]
param(
    [string]$Dest = ''
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$BaseUrl = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/'
$Models = [ordered]@{
    'yolox_s.onnx' = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
    'yolox_m.onnx' = '21ff6cfdeb53b013bac2249599e55f00bff3cfdfdab37ed7a4620818c1d15b3f'
}

if ($Dest -eq '') {
    $Dest = Join-Path (Split-Path -Parent $PSScriptRoot) '.models'
}
$Dest = [System.IO.Path]::GetFullPath($Dest)

# The part files this run made and has not yet moved into place or deleted.
$Parts = New-Object System.Collections.Generic.List[string]

function Get-Sha256([string]$Path) {
    $sha = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $bytes = $sha.ComputeHash($stream)
    }
    finally {
        $stream.Dispose()
        $sha.Dispose()
    }
    return ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLowerInvariant()
}

function Test-Pinned([string]$Path, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    return (Get-Sha256 $Path) -eq $Sha256
}

function Remove-Parts {
    foreach ($part in @($Parts)) {
        if (Test-Path -LiteralPath $part) {
            Remove-Item -LiteralPath $part -Force -ErrorAction SilentlyContinue
        }
        if (-not (Test-Path -LiteralPath $part)) { [void]$Parts.Remove($part) }
    }
}

function Remove-Unverified([string]$Name, [string]$Sha256) {
    if (-not (Test-Path -LiteralPath $Dest -PathType Container)) { return }
    $pattern = '^\.' + [regex]::Escape($Name) + '\.[A-Za-z0-9]{6}$'
    foreach ($stale in Get-ChildItem -LiteralPath $Dest -Force) {
        if ($stale.Name -match $pattern) {
            Remove-Item -LiteralPath $stale.FullName -Force
            Write-Warning "fetch_model: removed a part file left by an earlier run"
        }
    }
    $target = Join-Path $Dest $Name
    if ((Test-Path -LiteralPath $target) -and -not (Test-Pinned $target $Sha256)) {
        Write-Warning "fetch_model: $Name does not match its pinned SHA-256; removing it"
        Remove-Item -LiteralPath $target -Force
    }
}

function Get-Model([string]$Name, [string]$Sha256) {
    $target = Join-Path $Dest $Name
    if (Test-Pinned $target $Sha256) {
        Write-Output "fetch_model: $Name present and verified"
        return
    }
    # New-Item has no -LiteralPath; this takes $Dest (already a full path) literally.
    [System.IO.Directory]::CreateDirectory($Dest) | Out-Null
    $letters = 'abcdefghijklmnopqrstuvwxyz0123456789'
    $suffix = -join (1..6 | ForEach-Object { $letters[(Get-Random -Maximum $letters.Length)] })
    $part = Join-Path $Dest ".$Name.$suffix"
    $Parts.Add($part)
    try {
        & curl.exe --proto '=https' --tlsv1.2 --max-time 300 --retry 3 `
            --proto-redir '=https' -fsSL -o $part ($BaseUrl + $Name)
        if ($LASTEXITCODE -ne 0) { throw "download of $Name failed" }
        if (-not (Test-Pinned $part $Sha256)) {
            throw "$Name failed SHA-256 verification; nothing installed"
        }
        Move-Item -LiteralPath $part -Destination $target -Force
        [void]$Parts.Remove($part)
        Write-Output "fetch_model: installed $Name to $Dest"
    }
    finally {
        Remove-Parts
    }
}

$status = 1
try {
    foreach ($name in $Models.Keys) { Remove-Unverified $name $Models[$name] }
    foreach ($name in $Models.Keys) { Get-Model $name $Models[$name] }
    $status = 0
}
catch {
    [Console]::Error.WriteLine("fetch_model: $($_.Exception.Message)")
}
finally {
    Remove-Parts
}
exit $status
