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
    $Shell = New-Object -ComObject Shell.Application
    $Folder = $Shell.NameSpace($Destination)
    if ($null -eq $Folder) {
        throw "Cannot read shortcut directory: $Destination. Check your permissions and rerun setup."
    }
    $Links = @()

    # Check both names before writing either link. Never replace another target.
    foreach ($City in @('London', 'Calgary')) {
        [string]$Name = 'The Wear Report - {0}.lnk' -f $City
        [string]$Path = Join-Path $Destination $Name
        $Exists = Test-Path -LiteralPath $Path
        if ($Exists) {
            if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
                throw "Cannot replace $Path because it is not a file. Remove the conflicting item and rerun setup."
            }
            $Entry = $Folder.ParseName($Name)
            $Existing = $Entry.GetLink
            $SameTarget = -not [string]::IsNullOrWhiteSpace($Existing.Path) -and
                [string]::Equals([IO.Path]::GetFullPath($Existing.Path), $Target,
                    [StringComparison]::OrdinalIgnoreCase)
            if (-not $SameTarget) {
                throw "Refusing $Path because its target belongs to another location. Delete this conflicting shortcut only if intended, then rerun setup."
            }
        }
        $Links += [PSCustomObject]@{ Path = $Path; Name = $Name; City = $City.ToLowerInvariant(); Exists = $Exists }
    }

    foreach ($Item in $Links) {
        [string]$LinkPath = $Item.Path
        $Stage = 'Write'
        try {
            if (-not $Item.Exists) {
                # A valid empty Unicode Shell Link header and terminal block.
                # Seed only this final link path so native COM can open it.
                [byte[]]$Header = New-Object byte[] 80
                $Header[0] = 0x4C
                ([Guid]'00021401-0000-0000-C000-000000000046').ToByteArray().CopyTo($Header, 4)
                $Header[20] = 0x80
                $Header[60] = 1
                [IO.File]::WriteAllBytes($LinkPath, $Header)
            }
            $Stage = 'GetLink'
            $Entry = $Folder.ParseName([string]$Item.Name)
            $Link = $Entry.GetLink
            $Stage = 'Path'
            $Link.Path = $Target
            $Stage = 'Arguments'
            $Link.Arguments = "-Source $($Item.City) -MaxPasses 1"
            $Stage = 'WorkingDirectory'
            $Link.WorkingDirectory = $Repo
            $Stage = 'Save'
            $Link.Save($LinkPath)
        } catch {
            $Recovery = ''
            if (-not $Item.Exists) { $Recovery = ' Delete this incomplete link if it exists.' }
            throw "Cannot save $LinkPath ($Stage): $($_.Exception.Message) Close programs using this link and check that the directory is writable and your permissions allow changes.$Recovery Then rerun setup."
        }
        Write-Host "Installed: $LinkPath"
    }
    Write-Host 'Setup complete. Double-click either city shortcut to start one pass.'
} catch {
    Write-Host "Setup failed: $($_.Exception.Message)"
    exit 1
}

exit 0
