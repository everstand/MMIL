param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet('push-preview', 'push', 'pull-preview', 'pull')]
    [string] $Mode
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = 'F:\Development_Project\MMIL'
$LocalRoot = '/f/Development_Project/MMIL/'
$RemoteRoot = 'server_8:/data01/chim/code/MMIL_D/'
$RsyncPath = 'C:\msys64\usr\bin\rsync.exe'
$RemoteRsyncPath = '/data01/chim/rsync-local/extract/usr/bin/rsync'
$SshCommand = 'ssh -F /c/Users/lenovo/.ssh/config -i /c/Users/lenovo/.ssh/id_rsa -o UserKnownHostsFile=/c/Users/lenovo/.ssh/known_hosts -o ClearAllForwardings=yes -o WarnWeakCrypto=no -o LogLevel=ERROR'
$SyncDirs = @(
    'tools/'
    'src/'
    'scripts/'
    'prompts/'
    'captions/'
    'captions_raw/'
)

if (-not (Test-Path -LiteralPath $RsyncPath)) {
    throw "Local rsync not found: $RsyncPath"
}
if (-not (Test-Path -LiteralPath $ProjectRoot)) {
    throw "Project root not found: $ProjectRoot"
}

$env:Path = 'C:\msys64\usr\bin;' + $env:Path
$env:MSYS2_ARG_CONV_EXCL = '--rsync-path=;--exclude-from='
$env:RSYNC_RSH = $SshCommand

$IsPreview = $Mode.EndsWith('-preview')
$IsPush = $Mode.StartsWith('push')
$RsyncOptions = @(
    '--recursive'
    '--checksum'
    '--compress'
    '--itemize-changes'
)
if ($IsPreview) {
    $RsyncOptions += '--dry-run'
}

Write-Host "Mode: $Mode"
Write-Host "Synced directories: $($SyncDirs -join ', ')"

if (-not $IsPreview) {
    $answer = Read-Host 'This will modify files at the destination. Type YES to continue'
    if ($answer -ne 'YES') {
        Write-Host 'Cancelled.'
        exit 1
    }
}

foreach ($dir in $SyncDirs) {
    $source = if ($IsPush) { "$LocalRoot$dir" } else { "$RemoteRoot$dir" }
    $destination = if ($IsPush) { "$RemoteRoot$dir" } else { "$LocalRoot$dir" }

    Write-Host ''
    Write-Host "== $dir =="
    Write-Host "Source: $source"
    Write-Host "Destination: $destination"

    & $RsyncPath `
        @RsyncOptions `
        "--rsync-path=$RemoteRsyncPath" `
        '--exclude=__pycache__/' `
        '--exclude=*.pyc' `
        $source `
        $destination

    if ($LASTEXITCODE -ne 0) {
        throw "rsync failed for $dir with exit code $LASTEXITCODE"
    }
}
