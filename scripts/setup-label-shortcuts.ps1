# Install two links to the existing launcher; do not run the launcher.
# Compatible with Windows PowerShell 5; no administrator rights are needed.
param([string]$ShortcutDir)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

try {
    [string]$Repo = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
    [string]$Target = [IO.Path]::Combine($Repo, 'label.cmd')
    if (-not (Test-Path -LiteralPath $Target -PathType Leaf)) {
        throw "Missing launcher: $Target. Update or restore the repository, then rerun setup."
    }

    if (-not $PSBoundParameters.ContainsKey('ShortcutDir')) {
        $ShortcutDir = [Environment]::GetFolderPath([Environment+SpecialFolder]::DesktopDirectory)
    }
    if ([string]::IsNullOrWhiteSpace($ShortcutDir) -or
        -not (Test-Path -LiteralPath $ShortcutDir -PathType Container)) {
        throw "Shortcut destination must be an existing directory: $ShortcutDir. Select a valid directory and rerun setup."
    }
    [string]$Destination = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $ShortcutDir).ProviderPath)
    $Shell = New-Object -ComObject WScript.Shell
    $Links = @()

    # Check both names before saving either link. Never replace another target.
    foreach ($City in @('London', 'Calgary')) {
        [string]$Path = Join-Path $Destination ("The Wear Report - {0}.lnk" -f $City)
        if (Test-Path -LiteralPath $Path) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Cannot replace $Path because it is not a file. Remove the conflicting item and rerun setup."
            }
            $Existing = $Shell.CreateShortcut($Path)
            $SameTarget = -not [string]::IsNullOrWhiteSpace($Existing.TargetPath) -and
                [string]::Equals([IO.Path]::GetFullPath($Existing.TargetPath), $Target,
                    [StringComparison]::OrdinalIgnoreCase)
            if (-not $SameTarget) {
                throw "Refusing $Path because its target belongs to another location. Delete this conflicting shortcut only if intended, then rerun setup."
            }
        }
        $Links += [PSCustomObject]@{ Path = $Path; City = $City.ToLowerInvariant() }
    }

    foreach ($Item in $Links) {
        [string]$LinkPath = $Item.Path
        $Stage = 'CreateShortcut'
        try {
            $Link = $Shell.CreateShortcut($LinkPath)
            $Stage = 'TargetPath'
            $Link.TargetPath = [string]$Target
            $Stage = 'Arguments'
            $Link.Arguments = "-Source $($Item.City) -MaxPasses 1"
            $Stage = 'WorkingDirectory'
            $Link.WorkingDirectory = [string]$Repo
            $Stage = 'Save'
            $Link.Save()
        } catch {
            $Failure = $_.Exception.Message
            $Details = ''
            if ($Stage -eq 'TargetPath') {
                # Inspect in-memory COM objects only; neither probe saves a link.
                $Requested = ($LinkPath.ToCharArray() | ForEach-Object { '{0:X4}' -f [int]$_ }) -join ','
                $Observed = ([string]$Link.FullName).ToCharArray() | ForEach-Object { '{0:X4}' -f [int]$_ }
                $Controls = @()
                foreach ($Strict in @('Off', '3.0')) {
                    try {
                        if ($Strict -eq 'Off') { Set-StrictMode -Off } else { Set-StrictMode -Version 3.0 }
                        $Probe = $Shell.CreateShortcut($LinkPath)
                        $Probe.TargetPath = [IO.Path]::Combine($env:SystemRoot, 'System32\cmd.exe')
                        $Controls += "strict=$Strict ASCII=ok"
                        try {
                            $Probe.TargetPath = $Target
                            $Controls += "strict=$Strict Unicode=ok"
                        } catch { $Controls += "strict=$Strict Unicode=$($_.Exception.Message)" }
                    } catch { $Controls += "strict=$Strict ASCII=$($_.Exception.Message)" }
                }
                Set-StrictMode -Version 3.0
                $Details = " ACP=$([Text.Encoding]::Default.CodePage); requested=$Requested; observed=$($Observed -join ','); controls=$($Controls -join ';')."
            }
            throw "Cannot save $LinkPath ($Stage; target: $Target): $Failure$Details Close programs using this link, check that the directory is writable and your permissions allow changes, then rerun setup."
        }
        Write-Host "Installed: $($Item.Path)"
    }
    Write-Host 'Setup complete. Double-click either city shortcut to start one pass.'
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)"
    exit 1
}

exit 0
