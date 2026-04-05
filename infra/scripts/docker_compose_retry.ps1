param(
    [string]$ComposeCommandText = "up -d --build",
    [int]$MaxAttempts = 4,
    [int]$DelaySeconds = 12
)

$ErrorActionPreference = "Stop"

function Invoke-DockerComposeWithRetry {
    param(
        [ValidateNotNullOrEmpty()]
        [string]$ComposeCmdText,
        [int]$Attempts,
        [int]$Delay
    )

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        Write-Host ("[docker-retry] Attempt {0}/{1}: docker compose {2}" -f $attempt, $Attempts, $ComposeCmdText) -ForegroundColor Cyan

        $composeTokens = $ComposeCmdText -split "\s+" | Where-Object { $_ -and $_.Trim().Length -gt 0 }
        $output = & docker compose $composeTokens 2>&1
        $exitCode = $LASTEXITCODE

        if ($exitCode -eq 0) {
            Write-Host ("[docker-retry] Success on attempt {0}." -f $attempt) -ForegroundColor Green
            return 0
        }

        $text = ($output | Out-String)
        $isTransientDaemonIssue =
            $text -match "500 Internal Server Error" -or
            $text -match "error reading from server: EOF" -or
            $text -match "rpc error: code = Unavailable"

        Write-Host $text

        if (-not $isTransientDaemonIssue -or $attempt -eq $Attempts) {
            Write-Host "[docker-retry] Non-retryable failure or retries exhausted." -ForegroundColor Red
            return $exitCode
        }

        Write-Host ("[docker-retry] Detected transient Docker daemon/buildkit error. Retrying in {0} second(s)..." -f $Delay) -ForegroundColor Yellow
        Start-Sleep -Seconds $Delay
    }

    return 1
}

$code = Invoke-DockerComposeWithRetry -ComposeCmdText $ComposeCommandText -Attempts $MaxAttempts -Delay $DelaySeconds
exit $code
