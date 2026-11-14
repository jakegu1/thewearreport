# Google SigLIP 2 (Apache-2.0), converted by onnx-community, pinned int8 artifact.
# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File scripts/fetch_probe_model.ps1 [-Dest DIR]
# No credentials. Each endpoint has a timeout and one bounded retry, in listed order.
[CmdletBinding()]
param([string] $Dest = '')
Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$Revision = 'ba1f3b0843f24bc5417d38e19c37b287d719b2f4'
$Sha256 = '0dd31785a2713f1113ef2272472165c69d580473dae38d7b47568ac587795e70'
$Bytes = 94553333
$Name = 'siglip2-base-patch16-224-int8.onnx'
$Remote = "onnx-community/siglip2-base-patch16-224-ONNX/resolve/$Revision/onnx/vision_model_int8.onnx"
$ManualUrl = "https://huggingface.co/$Remote"
if (-not $Dest) { $Dest = Join-Path (Split-Path -Parent $PSScriptRoot) '.models' }
$Dest = [IO.Path]::GetFullPath($Dest)
$Target = Join-Path $Dest $Name
$Part = ''

function Test-Pinned([string] $Path) {
    if (-not [IO.File]::Exists($Path)) { return $false }
    $stream = [IO.File]::OpenRead($Path)
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
        if ($stream.Length -ne $Bytes) { return $false }
        $digest = [BitConverter]::ToString($sha.ComputeHash($stream)).Replace('-', '').ToLowerInvariant()
        return $digest -eq $Sha256
    } finally {
        $stream.Dispose()
        $sha.Dispose()
    }
}

$status = 1
try {
    if (Test-Pinned $Target) {
        Write-Output "fetch_probe_model: $Name present and verified"
        $status = 0
    } else {
        [IO.Directory]::CreateDirectory($Dest) | Out-Null
        $Part = Join-Path $Dest ('.' + $Name + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
        foreach ($origin in @('https://huggingface.co', 'https://hf-mirror.com', 'https://modelscope.cn/models')) {
            # Invoke curl.exe with TLS verification, timeout and bounded retries.
            & curl.exe --proto '=https' --tlsv1.2 --proto-redir '=https' --connect-timeout 15 `
                --max-time 120 --retry 1 --retry-max-time 240 -fsSL -o $Part "$origin/$Remote"
            if ($LASTEXITCODE -eq 0 -and (Test-Pinned $Part)) {
                Move-Item -LiteralPath $Part -Destination $Target -Force
                Write-Output "fetch_probe_model: installed $Name; SHA-256 verified"
                $status = 0
                break
            }
        }
    }
} catch {
    # Never echo exception text: the manual instructions suffice.
    $status = 1
} finally {
    if ($Part -and [IO.File]::Exists($Part)) { [IO.File]::Delete($Part) }
}
if ($status -ne 0) {
    [Console]::Error.WriteLine('fetch_probe_model: pinned int8 download unavailable; nothing unverified installed.')
    [Console]::Error.WriteLine("Manual download: $ManualUrl")
    [Console]::Error.WriteLine("Target: $Target")
    [Console]::Error.WriteLine("Expected bytes: $Bytes; SHA-256: $Sha256")
}
exit $status
