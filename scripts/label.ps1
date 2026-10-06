<#
.SYNOPSIS
One labelling session on Windows: update, pick the city in daylight, label several
spot-check passes until a target, and leave the session's attribute files on the
clipboard.

.DESCRIPTION
Started by label.cmd in the repository root (double-click it). The launcher:

1. changes to the repository it lives in, runs `git pull --ff-only` and
   `uv sync --locked --no-install-package llama-cpp-python` (a failure is reported in one
   line, and the session goes on with the current version), and sets TMPDIR for its own
   process only to -TempDir, creating it;
2. picks the city: with -Source auto, Calgary if the sun is up there by the spot-check's
   daylight rule (not below -6 degrees), else London if it is, else it exits and says
   when the next window opens (UTC); -Source calgary or london forces one;
3. runs passes: each is one `spotcheck --attributes` run in the window. After a pass it
   shows the crops kept this session against -Target and waits up to -Interval minutes:
   Enter starts the next pass now, q finishes. It stops at the target, when the city
   leaves daylight, after -MaxPasses, or when two passes in a row fail;
4. keeps the hosted judge on when DEEPINFRA_API_KEY is set (process or user environment),
   or asks for the key once (masked; Enter means no judge this session). -NoJudge skips
   the judge. The key is never shown, logged, written to a file or copied;
5. prints one line per attribute file written this session, puts exactly those files'
   JSON on the clipboard, one per line, and prints a one-line summary.

It never opens, saves or copies an image: frames and crops stay inside spotcheck.

The test switches (-DryRun, -Now, -Python, -Spotcheck, -ClipboardFile) are for the
acceptance tests: spotcheck's dry-run pipeline and --judgements file reviewer, a fixed
UTC clock, another Python and script, and a file in place of the clipboard.
#>
[CmdletBinding()]
param(
    [ValidateSet('auto', 'calgary', 'london')]
    [string] $Source = 'auto',
    [ValidateRange(1, 100000)]
    [int] $Target = 60,
    [ValidateRange(0, 240)]
    [int] $Interval = 5,
    [ValidateRange(1, 100)]
    [int] $MaxPasses = 6,
    [string] $TempDir = '',
    [string] $OutDir = '',
    [switch] $NoJudge,
    [ValidateRange(1, 10000)]
    [int] $JudgeMaxRequests = 200,
    # Test switches.
    [switch] $DryRun,
    [string] $Now = '',
    [string] $Python = '',
    [string] $Spotcheck = '',
    [string] $ClipboardFile = ''
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

$KeyName = 'DEEPINFRA_API_KEY'
$JudgeModel = 'di-qwen3-vl-235b'
# The guide's settings for one pass. --n is spotcheck's largest sample (MAX_N), so that a
# pass labels every frame of the sweep that has a near-field person. Every pass, London's
# and Calgary's, also records the per-camera yield counts (--record-camera-yield).
$PassArguments = @(
    '--attributes', '--n', '500', '--min-height', '46', '--min-persons', '1',
    '--timeout', '3600', '--record-camera-yield'
)
$CityNames = @{ calgary = 'Calgary'; london = 'London' }

$StartDir = (Get-Location).Path
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Repo

function Get-FullPath([string] $Path) {
    [IO.Path]::GetFullPath([IO.Path]::Combine($StartDir, $Path))
}

# Input: the keyboard, or lines of redirected input (the tests). --------------------------

$script:Redirected = [Console]::IsInputRedirected
$script:Reader = $null
$script:Pending = $null
$script:InputEnded = $false

function Wait-Line([int] $Milliseconds) {
    # The next line of redirected input within $Milliseconds (-1: no limit), or $null if
    # none came or the input has ended.
    if ($script:InputEnded) { return $null }
    if ($null -eq $script:Reader) {
        $script:Reader = New-Object IO.StreamReader([Console]::OpenStandardInput())
    }
    if ($null -eq $script:Pending) { $script:Pending = $script:Reader.ReadLineAsync() }
    try {
        if (-not $script:Pending.Wait($Milliseconds)) { return $null }
        $line = $script:Pending.Result
    } catch {
        $line = $null
    }
    $script:Pending = $null
    if ($null -eq $line) { $script:InputEnded = $true }
    return $line
}

function Read-Answer([string] $Prompt) {
    if (-not $script:Redirected) { return Read-Host -Prompt $Prompt }
    Write-Host "${Prompt}: " -NoNewline
    $line = Wait-Line -1
    Write-Host ''
    if ($null -eq $line) { return '' }
    return $line
}

function Wait-NextPass([int] $Minutes) {
    # 'next' after $Minutes or on Enter, 'quit' on q.
    if ($Minutes -le 0) { return 'next' }
    Write-Host "Next pass in $Minutes min. Enter: start it now. q: finish the session."
    $deadline = (Get-Date).AddMinutes($Minutes)
    $live = -not $script:Redirected
    $countdown = $live -and -not [Console]::IsOutputRedirected
    if ($live) {
        while ([Console]::KeyAvailable) { [void][Console]::ReadKey($true) }
    }
    try {
        while ($true) {
            $left = $deadline - (Get-Date)
            if ($left.TotalMilliseconds -le 0) { return 'next' }
            if ($countdown) {
                $text = '{0}:{1:00}' -f [int][Math]::Floor($left.TotalMinutes), $left.Seconds
                Write-Host "`r  next pass in $text  " -NoNewline
            }
            $step = [int][Math]::Max(1, [Math]::Min(1000, $left.TotalMilliseconds))
            if ($live) {
                while ([Console]::KeyAvailable) {
                    $key = [Console]::ReadKey($true)
                    if ($key.Key -eq [ConsoleKey]::Enter) { return 'next' }
                    if ($key.KeyChar -eq 'q' -or $key.KeyChar -eq 'Q') { return 'quit' }
                }
                Start-Sleep -Milliseconds ([Math]::Min(200, $step))
            } elseif ($script:InputEnded) {
                Start-Sleep -Milliseconds $step
            } else {
                $line = Wait-Line $step
                if ($null -ne $line) {
                    if ($line.Trim() -ieq 'q') { return 'quit' }
                    return 'next'
                }
            }
        }
    } finally {
        if ($countdown) { Write-Host '' }
    }
}

# Python: the repository's environment through uv, or -Python (tests). --------------------

function Invoke-Python([string[]] $Arguments) {
    if ($Python) { & $Python @Arguments } else { & uv run --no-sync python @Arguments }
}

function Invoke-Helper([string[]] $Arguments) {
    $out = Invoke-Python (@('-m', 'wearreport.tools.label_window') + $Arguments)
    if ($LASTEXITCODE -ne 0) { return $null }
    return (@($out) -join "`n") | ConvertFrom-Json
}

function Get-City([string] $Wanted) {
    $arguments = @('city', '--source', $Wanted)
    if ($Now) { $arguments += @('--now', $Now) }
    $choice = Invoke-Helper $arguments
    if ($null -eq $choice) { throw 'cannot work out the city: the helper failed' }
    return $choice
}

function Format-Window($Window) {
    if ($Window -is [datetime]) {
        return $Window.ToUniversalTime().ToString('yyyy-MM-dd HH:mm') + ' UTC'
    }
    return ([string] $Window).Replace('T', ' ').Replace('Z', ' UTC')
}

# 1. Update and the temporary directory. --------------------------------------------------

function Invoke-Update([string] $What, [string] $Command, [string[]] $Arguments) {
    $previous = $ErrorActionPreference
    $ok = $false
    if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
        Write-Host "Update: $What failed ($Command not found); continuing with the current version."
        return
    }
    try {
        $ErrorActionPreference = 'Continue'
        $global:LASTEXITCODE = 0
        $null = & $Command @Arguments 2>&1
        $ok = ($LASTEXITCODE -eq 0)
    } catch {
        $ok = $false
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($ok) {
        Write-Host "Update: $What done."
    } else {
        Write-Host "Update: $What failed; continuing with the current version."
    }
}

Write-Host 'The Wear Report: labelling session'
Write-Host "Repository: $Repo"
$env:GIT_TERMINAL_PROMPT = '0'
Invoke-Update 'git pull' 'git' @('pull', '--ff-only')
Invoke-Update 'uv sync' 'uv' @('sync', '--locked', '--no-install-package', 'llama-cpp-python')

if (-not $TempDir) {
    if (Test-Path -LiteralPath 'D:\') {
        $TempDir = 'D:\spotcheck-tmp'
    } else {
        $TempDir = Join-Path $env:USERPROFILE 'spotcheck-tmp'
    }
}
$TempDir = Get-FullPath $TempDir
$null = New-Item -ItemType Directory -Force -Path $TempDir
$env:TMPDIR = $TempDir  # this process and its children only
$env:PYTHONUNBUFFERED = '1'
if ($OutDir) { $OutDir = Get-FullPath $OutDir } else { $OutDir = Join-Path $Repo 'spotchecks' }
$JudgementsFile = Join-Path $TempDir 'label-judgements.json'

# 2. The city. -------------------------------------------------------------------------------

$choice = Get-City $Source
if (-not $choice.city) {
    if ($Source -eq 'auto') { $where = 'Calgary or London' } else { $where = $CityNames[$Source] }
    if ($choice.next_window) {
        $opens = Format-Window $choice.next_window
        $city = $CityNames[[string] $choice.next_city]
        Write-Host "No daylight in $where now. The next window opens at $opens ($city)."
    } else {
        Write-Host "No daylight in $where within the next 48 hours."
    }
    exit 1
}
$City = [string] $choice.city
$CityName = $CityNames[$City]
Write-Host "City: $CityName."

# 3. The judge. ------------------------------------------------------------------------------

function Enable-Judge {
    if ([Environment]::GetEnvironmentVariable($KeyName, 'Process')) { return $true }
    $saved = [Environment]::GetEnvironmentVariable($KeyName, 'User')
    if ($saved) {
        [Environment]::SetEnvironmentVariable($KeyName, $saved, 'Process')
        return $true
    }
    $prompt = 'DeepInfra API key for the judge (Enter: no judge this session)'
    if ($script:Redirected) {
        Write-Host "${prompt}: " -NoNewline
        $key = Wait-Line -1
        Write-Host ''
        if ($null -eq $key) { $key = '' }
    } else {
        $secure = Read-Host -Prompt $prompt -AsSecureString
        $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
        try {
            $key = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
        } finally {
            [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
        }
    }
    $key = $key.Trim()
    if (-not $key) {
        Write-Host 'No judge this session.'
        return $false
    }
    [Environment]::SetEnvironmentVariable($KeyName, $key, 'Process')
    $answer = Read-Answer 'Save it as a user environment variable for next time? [y/N]'
    if ($answer.Trim() -match '^(y|yes)$') {
        [Environment]::SetEnvironmentVariable($KeyName, $key, 'User')
        Write-Host 'Saved for next time.'
    }
    return $true
}

$Judge = $false
if (-not $NoJudge) { $Judge = Enable-Judge }
if ($Judge) { Write-Host "Judge: $JudgeModel, at most $JudgeMaxRequests requests a pass." }

# 4. The passes. ---------------------------------------------------------------------------

function Get-AttributeFiles {
    $dir = Join-Path $OutDir 'attributes'
    if (-not (Test-Path -LiteralPath $dir -PathType Container)) { return @() }
    return @(Get-ChildItem -LiteralPath $dir -Filter '*.json' -File | ForEach-Object { $_.FullName })
}

function Remove-JudgementsFile {
    if ($DryRun -and (Test-Path -LiteralPath $JudgementsFile)) {
        Remove-Item -LiteralPath $JudgementsFile -Force
    }
}

function Get-PassCommand {
    if ($Spotcheck) { $command = @($Spotcheck) } else { $command = @('-m', 'wearreport.tools.spotcheck') }
    $command += $PassArguments
    $command += @('--out-dir', $OutDir, '--source', $City)
    if ($DryRun) {
        $command += @('--dry-run', '--judgements', $JudgementsFile)
    } else {
        $command += @('--view', 'window', '--confirm-stop')
    }
    if ($Judge) { $command += @('--judge', $JudgeModel, '--judge-max-requests', "$JudgeMaxRequests") }
    return $command
}

$SessionFiles = New-Object 'System.Collections.Generic.List[string]'
$Counts = $null
$Kept = 0
$Judged = 0
$Passes = 0
$FailedInRow = 0
$Reason = ''
try {
    while ($true) {
        $Passes++
        Remove-JudgementsFile
        $before = @(Get-AttributeFiles)
        Write-Host ''
        Write-Host "Pass $Passes (at most $MaxPasses), $CityName. Kept so far: $Kept of $Target."
        Invoke-Python (Get-PassCommand) | Out-Host
        $code = $LASTEXITCODE
        Remove-JudgementsFile
        $new = @(Get-AttributeFiles | Where-Object { $before -notcontains $_ })
        if ($code -ne 0 -or $new.Count -eq 0) {
            $FailedInRow++
            if ($FailedInRow -ge 2) { $next = 'stopping' } else { $next = 'continuing' }
            if ($code -ne 0) {
                Write-Host "Pass $Passes failed (exit $code); $next."
            } else {
                Write-Host "Pass $Passes failed: no attribute file was written; $next."
            }
        } else {
            $FailedInRow = 0
            foreach ($path in $new) { $SessionFiles.Add($path) }
            $counted = Invoke-Helper (@('count') + $SessionFiles.ToArray())
            if ($null -eq $counted) {
                Write-Host 'Cannot count the crops of this session''s files; the count is unchanged.'
            } else {
                $passKept = [int] $counted.kept - $Kept
                $Counts = $counted
                $Kept = [int] $counted.kept
                $Judged = [int] $counted.judged
                Write-Host "Pass ${Passes}: $passKept crops kept. This session: $Kept of $Target."
            }
        }
        if ($Kept -ge $Target) { $Reason = "Target reached: $Kept of $Target crops kept."; break }
        if ($FailedInRow -ge 2) { $Reason = 'Two passes in a row failed; stopping.'; break }
        if ($Passes -ge $MaxPasses) { $Reason = "Reached the maximum of $MaxPasses passes."; break }
        if ((Wait-NextPass $Interval) -eq 'quit') { $Reason = 'Session finished at your request.'; break }
        $still = Get-City $City
        if (-not $still.city) { $Reason = "$CityName has left daylight; stopping."; break }
    }
} finally {
    Remove-JudgementsFile
}

# 5. The result. ---------------------------------------------------------------------------

Write-Host ''
Write-Host $Reason
if ($SessionFiles.Count -gt 0) {
    Write-Host 'Attribute files written this session:'
    if ($null -ne $Counts) {
        foreach ($file in @($Counts.files)) {
            Write-Host "  $($file.path): $($file.kept) crops kept, $($file.judged) judged"
        }
    } else {
        foreach ($path in $SessionFiles) { Write-Host "  $path" }
    }
    $lines = foreach ($path in $SessionFiles) { [IO.File]::ReadAllText($path).Trim() }
    $text = @($lines) -join "`r`n"
    if ($ClipboardFile) {
        [IO.File]::WriteAllText((Get-FullPath $ClipboardFile), $text, (New-Object Text.UTF8Encoding($false)))
    } else {
        Set-Clipboard -Value $text
    }
    Write-Host "Session: $Passes passes, $Kept crops kept, $Judged crops judged."
    Write-Host 'Copied to the clipboard.'
} else {
    Write-Host "Session: $Passes passes, 0 crops kept, 0 crops judged."
    Write-Host 'No attribute file was written this session; the clipboard is unchanged.'
}
exit 0
