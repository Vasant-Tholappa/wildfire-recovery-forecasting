param(
    [ValidateRange(1, 168)][int]$MaxHours = 12,
    [ValidateRange(1, 2088)][int]$MaxNew = 400,
    [ValidateRange(1, 60)][int]$SessionMinutes = 15,
    [ValidateRange(1, 1440)][int]$NoProgressMinutes = 120
)

function Invoke-ProductionContinuation {
    param($Hours, $NewLimit, $SliceMinutes, $StallMinutes,
          [scriptblock]$Runner, [scriptblock]$Clock = { [DateTimeOffset]::UtcNow },
          [string]$LogDirectory = (Join-Path $PSScriptRoot 'data/processed'))
    $ErrorActionPreference = 'Stop'
    $version = 'phase3_ca175_tile1_v1'
    $lockPath = Join-Path $LogDirectory ($version + '_continuation.lock')
    $lock = [System.IO.File]::Open($lockPath, 'CreateNew', 'Write', 'None')
    $log = Join-Path $LogDirectory ($version + '_unattended_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '.log')
    $started = & $Clock
    $deadline = $started.AddHours($Hours)
    $lastProgress = $started
    $initialPending = $null
    $lastComplete = $null
    $quickReturn = $false
    function Invoke-Logged($arguments) {
        $result = & $Runner $arguments
        $text = $result.Lines -join "`n"
        Add-Content -LiteralPath $log -Value $text
        if ($result.ExitCode -ne 0) { throw "Controller exited $($result.ExitCode). Inspect $log; no automatic retry." }
        return $text
    }
    try {
        Write-Host "Continuation PID=$PID; log=$log; version=$version"
        while ($true) {
            $status = (Invoke-Logged @('status')) | ConvertFrom-Json
            if ($status.production_version -ne $version -or $status.expected_exports -ne 2100) {
                throw 'Unexpected population/version. Stopping.'
            }
            if ($status.failed -gt 0) { throw 'FAILED work items require intervention; stopping without retry.' }
            if ($null -eq $initialPending) { $initialPending = [int]$status.pending; $lastComplete = [int]$status.completed }
            $now = & $Clock
            if ($status.completed -gt $lastComplete) { $lastProgress = $now; $lastComplete = [int]$status.completed }
            # Count durable new status records, including ambiguous intents, conservatively.
            $newCount = $initialPending - [int]$status.pending
            if ($newCount -lt 0) { throw 'Persisted status count decreased; inspect before continuing.' }
            $message = "$(Get-Date -Format o) COMPLETE=$($status.completed)/2100 RUNNING=$($status.running_status) FAILED=$($status.failed) PENDING=$($status.pending) NEW=$newCount/$NewLimit"
            Write-Host $message
            Add-Content -LiteralPath $log -Value $message
            if ($status.completed -eq 2100) { Write-Host 'All products persisted COMPLETE.'; return }
            if ($now -ge $deadline -or $newCount -ge $NewLimit) { Write-Host 'Unattended safety limit reached. Remote tasks may continue; review usage before resuming.'; return }
            if ($quickReturn) { throw 'Session returned unexpectedly quickly. Inspect log before restarting.' }
            if (($now - $lastProgress).TotalMinutes -ge $StallMinutes) { throw 'No new persisted completions within the no-progress limit. Inspect tasks/log; no automatic retry.' }
            $minutes = [Math]::Floor([Math]::Min($SliceMinutes, ($deadline-$now).TotalMinutes))
            if ($minutes -lt 1) { Write-Host 'Runtime limit reached.'; return }
            Write-Host "Starting bounded session: up to $minutes minutes, max-active=4."
            $text = Invoke-Logged @('session', '--session-new-cap', [string]($NewLimit-$newCount),
                '--max-active', '4', '--max-progress', '12', '--poll-seconds', '60', '--max-minutes', [string]$minutes)
            if ($text -match '"action"\s*:\s*"(FAILED|NEEDS_RETRY|NEEDS_REVIEW)"') {
                throw 'Controller requested intervention. Inspect log; continuation stopped.'
            }
            # A controller that returns immediately without work must not cause a busy loop.
            $quickReturn = ((& $Clock) -lt $now.AddSeconds(30))
        }
    } finally {
        $lock.Dispose()
        Remove-Item -LiteralPath $lockPath
    }
}

if ($MyInvocation.InvocationName -ne '.') {
    Push-Location $PSScriptRoot
    try {
        $runner = {
            param($arguments)
            $lines = & uv run --offline python manual_population.py @arguments 2>&1
            [pscustomobject]@{ ExitCode = $LASTEXITCODE; Lines = @($lines | ForEach-Object { $_.ToString() }) }
        }
        Invoke-ProductionContinuation -Hours $MaxHours -NewLimit $MaxNew -SliceMinutes $SessionMinutes -StallMinutes $NoProgressMinutes -Runner $runner
    } finally { Pop-Location }
}
