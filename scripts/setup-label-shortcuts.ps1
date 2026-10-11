# Install two links to the existing launcher; do not run the launcher.
# Compatible with Windows PowerShell 5; no administrator rights are needed.
param([string]$ShortcutDir)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

try {
    $Repo = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).ProviderPath
    $Target = Join-Path $Repo 'label.cmd'
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
    $Destination = (Resolve-Path -LiteralPath $ShortcutDir).ProviderPath
    $Shell = New-Object -ComObject WScript.Shell
    $Links = @()

    # Check both names before saving either link. Never replace another target.
    foreach ($City in @('London', 'Calgary')) {
        $Path = Join-Path $Destination ("The Wear Report - {0}.lnk" -f $City)
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
        try {
            $Link = $Shell.CreateShortcut($Item.Path)
            $Link.TargetPath = $Target
            $Link.Arguments = "-Source $($Item.City) -MaxPasses 1"
            $Link.WorkingDirectory = $Repo
            $Link.Save()
        } catch {
            throw "Cannot save $($Item.Path). Close programs using this link, check that the directory is writable and your permissions allow changes, then rerun setup."
        }
        Write-Host "Installed: $($Item.Path)"
    }
    Write-Host 'Setup complete. Double-click either city shortcut to start one pass.'
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)"
    exit 1
}

exit 0
