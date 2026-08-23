# agent-runner.ps1 — Startet den lokalen Agent-Runner für den MyProxy-Co-Worker
#
# Der Runner long-pollt https://proxy.abigailrook.de/agent/tools/claim und
# führt Tool-Calls des Coworker-Subagents LOKAL aus (read_file/write_file/
# list_dir/web_search) — der Proxy-Container (Coolify) hat keinen Dateizugriff.
#
# Verwendung:
#   .\agent-runner.ps1              # im Vordergrund starten
#   .\agent-runner.ps1 -Install     # als geplante Aufgabe beim Login starten
#   .\agent-runner.ps1 -Uninstall   # geplante Aufgabe entfernen
#   .\agent-runner.ps1 -Status      # läuft der Runner / ist er am Proxy verbunden?
#
# Token/Workspace: aus .env (PROXY_API_KEY bzw. AGENT_RUNNER_TOKEN,
# AGENT_WORKSPACE) oder per Parameter überschreiben.

param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Status,
    [string]$ProxyUrl = "https://proxy.abigailrook.de",
    [string]$Token = "",
    [string]$Workspace = "D:\GitHub"
)

$ErrorActionPreference = "Stop"
$envFile = Join-Path $PSScriptRoot ".env"
$taskName = "MyProxy-AgentRunner"
$pythonExe = Join-Path $PSScriptRoot ".venv-1\Scripts\python.exe"
$runnerScript = Join-Path $PSScriptRoot "tools\agent_runner.py"

# ─── Hilfsfunktion: .env auslesen ───
function Get-EnvValue($key) {
    if (-not (Test-Path $envFile)) { return $null }
    $line = Select-String -Path $envFile -Pattern "^\s*$key\s*=" | Select-Object -First 1
    if ($line) { return ($line.Line -split '=', 2)[1].Trim().Trim('"').Trim("'") }
    return $null
}

if (-not $Token) {
    $Token = Get-EnvValue "AGENT_RUNNER_TOKEN"
    if (-not $Token) { $Token = Get-EnvValue "PROXY_API_KEY" }
}
$wsFromEnv = Get-EnvValue "AGENT_WORKSPACE"
if ($wsFromEnv -and $Workspace -eq "D:\GitHub") { $Workspace = $wsFromEnv }

if (-not (Test-Path $pythonExe)) {
    Write-Host "❌ Python nicht gefunden: $pythonExe" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $runnerScript)) {
    Write-Host "❌ Runner-Skript nicht gefunden: $runnerScript" -ForegroundColor Red
    exit 1
}

# ─── Uninstall ───
if ($Uninstall) {
    if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
        Write-Host "✅ Geplante Aufgabe '$taskName' entfernt." -ForegroundColor Green
    } else {
        Write-Host "ℹ️ Geplante Aufgabe '$taskName' existiert nicht." -ForegroundColor Yellow
    }
    # Läuft noch ein Runner-Prozess? Beenden.
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*agent_runner.py*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force; Write-Host "✅ Runner-Prozess $($PSItem.ProcessId) beendet." }
    exit 0
}

# ─── Status ───
if ($Status) {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -like "*agent_runner.py*" }
    if ($procs) {
        Write-Host "✅ Runner läuft lokal (PID: $(($procs | ForEach-Object ProcessId) -join ', '))" -ForegroundColor Green
    } else {
        Write-Host "❌ Runner läuft NICHT lokal." -ForegroundColor Red
    }
    try {
        $h = @{ Authorization = "Bearer $Token" }
        $st = Invoke-RestMethod -Uri "$ProxyUrl/agent/status" -Headers $h -TimeoutSec 10
        $online = if ($st.runner_online) { "verbunden ✅" } else { "NICHT verbunden ❌" }
        Write-Host "Proxy-Agent-Status: runner_online=$online, agent_mode=$($st.agent_mode), queue=$($st.queue_depth), pending=$($st.pending)"
    } catch {
        Write-Host "⚠️ Proxy-Status nicht abrufbar: $($_.Exception.Message)" -ForegroundColor Yellow
    }
    exit 0
}

# ─── Install (geplante Aufgabe beim Login) ───
if ($Install) {
    $action = New-ScheduledTaskAction -Execute $pythonExe `
        -Argument "`"$runnerScript`" --proxy `"$ProxyUrl`" --token `"$Token`" --workspace `"$Workspace`""
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit ([TimeSpan]::Zero)
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
        -Settings $settings -Description "Lokaler Agent-Runner fuer MyProxy Co-Worker (Tool-Relay)" -Force | Out-Null
    Write-Host "✅ Geplante Aufgabe '$taskName' installiert (Start bei Anmeldung)." -ForegroundColor Green
    Write-Host "   Starte jetzt einmal manuell..." -ForegroundColor Cyan
    Start-ScheduledTask -TaskName $taskName
    Start-Sleep 3
    & $MyInvocation.MyCommand.Path -Status
    exit 0
}

# ─── Direktstart (Vordergrund) ───
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    MyProxy Agent-Runner                  ║" -ForegroundColor Cyan
Write-Host "║    Proxy:     $ProxyUrl" -ForegroundColor Cyan
Write-Host "║    Workspace: $Workspace" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan

& $pythonExe $runnerScript --proxy $ProxyUrl --token $Token --workspace $Workspace
