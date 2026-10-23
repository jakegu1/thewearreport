<#
.SYNOPSIS
Offline prospective calendar planning, local preflight, or an explicit human London pass.
.DESCRIPTION
No arguments prints help. No update, installation, capture, or scheduled task is performed
by offline modes. -StartLondon hands control to the existing launcher and its time gates.
#>
[CmdletBinding(DefaultParameterSetName = 'Plan')]
param(
    [Parameter(ParameterSetName = 'Start', Mandatory = $true)]
    [switch] $StartLondon,
    [Parameter(ParameterSetName = 'Start')]
    [string] $OutDir,
    [Parameter(ParameterSetName = 'Preflight', Mandatory = $true)]
    [switch] $Preflight,
    [Parameter(ParameterSetName = 'Plan')]
    [switch] $Help,
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $Zone,
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $StartDate,
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [ValidateRange(1, 14)]
    [int] $Days = 7,
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $OnlineStart = '00:00',
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $OnlineEnd = '24:00',
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $Weekdays = 'mon,tue,wed,thu,fri,sat,sun',
    [Parameter(ParameterSetName = 'Plan')]
    [Parameter(ParameterSetName = 'Preflight')]
    [string] $Python
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = New-Object Text.UTF8Encoding($false)
$Repo = Split-Path -Parent $PSScriptRoot

function Show-LocalHelp {
    Write-Host 'Offline prospective planning (no capture):'
    Write-Host '  label-local.cmd -Zone Europe/London -StartDate 2026-10-25 -Days 7'
    Write-Host '  Optional: -OnlineStart HH:MM -OnlineEnd HH:MM (or 24:00) -Weekdays mon,tue'
    Write-Host '  Optional: -Python C:\path\to\python.exe; default: repository .venv\Scripts\python.exe'
    Write-Host '  label-local.cmd -Preflight -Zone Europe/London'
    Write-Host 'A present human may explicitly start one London pass, with no AI judge:'
    Write-Host '  label-local.cmd -StartLondon [-OutDir C:\path\to\labels]'
    Write-Host 'The existing launcher supplies its updater, daylight check, and London time gates.'
}

try {
    if ($Help -or $PSBoundParameters.Count -eq 0) {
        Show-LocalHelp
        exit 0
    }
    Set-Location -LiteralPath $Repo
    $Launcher = Join-Path $Repo 'label.cmd'
    if ($StartLondon) {
        if (-not (Test-Path -LiteralPath $Launcher -PathType Leaf)) {
            Write-Host "Missing label.cmd at $Launcher. Restore the repository entry."
            exit 1
        }
        Write-Host 'Human-only London single pass. Remain present to review; no AI judge.'
        $LaunchArguments = @('-Source', 'london', '-MaxPasses', '1', '-NoJudge')
        if ($OutDir) { $LaunchArguments += @('-OutDir', $OutDir) }
        & $Launcher @LaunchArguments
        exit $LASTEXITCODE
    }
    if (-not $Zone) {
        Write-Host 'Give -Zone with an explicit IANA display timezone; see -Help.'
        exit 2
    }
    if (-not $Python) { $Python = Join-Path $Repo '.venv\Scripts\python.exe' }
    if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
        Write-Host "Missing Python at $Python. Prepare the local environment or supply -Python. See docs/local-label.md."
        exit 1
    }
    $Python = (Resolve-Path -LiteralPath $Python).Path
    if ($Preflight) {
        foreach ($Relative in @('label-local.cmd', 'scripts\label-local.ps1', 'label.cmd',
            'scripts\label.ps1', 'pyproject.toml', 'uv.lock', '.venv\pyvenv.cfg',
            'engine\wearreport\tools\local_label_plan.py', '.models\yolox_m.onnx')) {
            $Required = Join-Path $Repo $Relative
            if (-not (Test-Path -LiteralPath $Required -PathType Leaf)) {
                Write-Host "Missing $Relative at $Required. Complete the one-time setup in docs/local-label.md."
                exit 1
            }
        }
        Write-Host "Repository: $Repo"
        Write-Host "Python: $Python"
        Write-Host "Lock: $(Join-Path $Repo 'uv.lock')"
        Write-Host "Local detector file present: $(Join-Path $Repo '.models\yolox_m.onnx')"
    }
    # Use only the module in this checkout; do not depend on an installed T-093 module.
    $env:PYTHONPATH = Join-Path $Repo 'engine'
    $PlanArguments = @('-m', 'wearreport.tools.local_label_plan', '--zone', $Zone,
        '--days', "$Days", '--online-start', $OnlineStart, '--online-end', $OnlineEnd,
        '--weekdays', $Weekdays)
    if ($StartDate) { $PlanArguments += @('--start-date', $StartDate) }
    if ($Preflight) { $PlanArguments += '--preflight' }
    & $Python @PlanArguments
    exit $LASTEXITCODE
} catch {
    Write-Host 'Local entry failed. Check the selected Python path and repository files; see docs/local-label.md.'
    exit 1
}
