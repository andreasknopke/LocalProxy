# proxy-status.ps1 — Status & Logs des DEPLOYED LocalProxy (Coolify, 192.168.188.134)
# Rein HTTP-API-basiert: KEIN SSH, kein Dateizugriff auf dem Server.
#   /healthz + /v1/models sind offen; /logs + /debug/* brauchen den Proxy-API-Key
#   (Bearer) — denselben Key, den auch VS Code Copilot für den Proxy nutzt.
#
# Usage:
#   .\proxy-status.ps1                  # Status-Übersicht (Coworker + Kategorien)
#   .\proxy-status.ps1 -Raw             # komplettes /healthz JSON ausgeben
#   .\proxy-status.ps1 -HostName 192.168.188.134 -Port 9001   # abweichendes Ziel
#   .\proxy-status.ps1 -Logs 40         # letzte N Log-Zeilen per API
#   .\proxy-status.ps1 -Streams 5       # letzte N I/O-Trace-Turns per API
#   .\proxy-status.ps1 -Turn <turn_id>  # vollständige I/O-Trace eines Turns per API
#   .\proxy-status.ps1 -ClearStreams    # alle Trace-Turns löschen (Rotation erzwingen)
#
# API-Key-Auflösung (höchste Priorität zuerst):
#   1. -ApiKey <key>
#   2. $env:PROXY_STATUS_API_KEY
#   3. .\.env.proxy-status.local (liegt im Repo-Root, ist gitignored). Format:
#        PROXY_STATUS_API_KEY=localfox-...
#        PROXY_STATUS_HOST=192.168.188.134   (optional)
#        PROXY_STATUS_PORT=9001              (optional)
#      Der Wert ist der PROXY_API_KEY aus den Coolify-Env-Vars der App.

param(
    [string]$HostName = "",
    [int]$Port = 0,
    [switch]$Raw,
    [int]$Logs = 0,
    [int]$Streams = 0,
    [string]$Turn = "",
    [string]$ApiKey = "",
    [switch]$ClearStreams
)

$ErrorActionPreference = "Stop"

# ─── Config: Host/Port/Key auflösen ─────────────────────────────────────────
$script:CfgHost = "192.168.188.134"
$script:CfgPort = 9001
$script:CfgKey = ""

# 1) Datei (niedrigste Priorität)
$envFile = Join-Path $PSScriptRoot ".env.proxy-status.local"
if (Test-Path $envFile) {
    foreach ($line in Get-Content $envFile -ErrorAction SilentlyContinue) {
        $t = $line.Trim()
        if (-not $t -or $t.StartsWith("#")) { continue }
        $idx = $t.IndexOf("=")
        if ($idx -lt 1) { continue }
        $k = $t.Substring(0, $idx).Trim()
        $v = $t.Substring($idx + 1).Trim()
        if ($k -eq "PROXY_STATUS_API_KEY")      { $script:CfgKey = $v }
        elseif ($k -eq "PROXY_STATUS_HOST")     { $script:CfgHost = $v }
        elseif ($k -eq "PROXY_STATUS_PORT")     { $script:CfgPort = [int]$v }
    }
}
# 2) Env
if ($env:PROXY_STATUS_API_KEY) { $script:CfgKey = $env:PROXY_STATUS_API_KEY }
if ($env:PROXY_STATUS_HOST)    { $script:CfgHost = $env:PROXY_STATUS_HOST }
if ($env:PROXY_STATUS_PORT)    { $script:CfgPort = [int]$env:PROXY_STATUS_PORT }
# 3) Parameter (höchste Priorität)
if ($HostName -ne "")  { $script:CfgHost = $HostName }
if ($Port -gt 0)       { $script:CfgPort = $Port }
if ($ApiKey -ne "")    { $script:CfgKey = $ApiKey }

$base = "http://$($script:CfgHost):$($script:CfgPort)"
$script:Headers = @{}
if ($script:CfgKey) {
    $script:Headers = @{ "Authorization" = "Bearer $($script:CfgKey)" }
}

# ─── HTTP-Helper ────────────────────────────────────────────────────────────
function Invoke-ProxyApi {
    param(
        [string]$Path,
        [ValidateSet("GET", "DELETE")] [string]$Method = "GET"
    )
    try {
        $irmArgs = @{ Uri = "$base$Path"; Method = $Method; TimeoutSec = 20 }
        if ($script:CfgKey) { $irmArgs.Headers = $script:Headers }
        return Invoke-RestMethod @irmArgs
    }
    catch [Microsoft.PowerShell.Commands.HttpResponseException] {
        $code = $_.Exception.Response.StatusCode.value__
        if ($code -eq 401 -or $code -eq 403) {
            Write-Host "❌ $code — API-Key fehlt oder falsch (Auth ist aktiviert)." -ForegroundColor Red
            Write-Host "   Key = PROXY_API_KEY aus den Coolify-Env-Vars der App." -ForegroundColor Yellow
            Write-Host "   Setze -ApiKey <key>, `$env:PROXY_STATUS_API_KEY oder" -ForegroundColor Yellow
            Write-Host "   .env.proxy-status.local (siehe Header dieses Scripts)." -ForegroundColor Yellow
            exit 1
        }
        throw
    }
}

# ─── /healthz (offen — kein Key nötig) ──────────────────────────────────────
try {
    $health = Invoke-RestMethod -Uri "$base/healthz" -TimeoutSec 10
} catch {
    Write-Host "❌ Proxy nicht erreichbar unter $base/healthz" -ForegroundColor Red
    Write-Host "   $_" -ForegroundColor DarkGray
    Write-Host "   Läuft der Coolify-Container? Port in Coolify exposition korrekt?" -ForegroundColor Yellow
    exit 1
}

if ($Raw) {
    $health | ConvertTo-Json -Depth 10
    exit 0
}

# ─── Übersicht ───
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  LocalProxy-Status  ($($script:CfgHost) / Port $($script:CfgPort))  ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ("  Version:          {0}" -f $health.version)
Write-Host ("  Default-Kategorie:{0}" -f $health.default_category)
Write-Host ""

# ─── Co-Worker-Block ───
$cw = $health.coworker
if ($cw) {
    $stateColor = if ($cw.reachable) { "Green" } elseif (-not $cw.configured) { "Yellow" } else { "Red" }
    $state = if ($cw.reachable) { "✅ ERREICHBAR — Tools werden injiziert (bei Kategorie 'local')" }
             elseif (-not $cw.enabled) { "⛔ deaktiviert (COWORKER_ENABLED=false)" }
             elseif (-not $cw.configured) { "⚠️  NICHT KONFIGURIERT (api_url/model_name fehlen in config.json)" }
             else { "❌ NICHT ERREICHBAR — $($cw.last_error)" }
    Write-Host "Co-Worker:" -ForegroundColor Cyan
    Write-Host ("  $state") -ForegroundColor $stateColor
    Write-Host ("    Fork-Join:    {0}" -f $cw.fork_join)
    Write-Host ("    Letzter Check:{0}" -f $cw.last_check)
} else {
    Write-Host "Co-Worker: ⚠️  Kein coworker-Block in /healthz — Proxy-Image älter als diese Version (redeploy!)" -ForegroundColor Yellow
    Write-Host "           Dort wird still KEIN Tool injiziert. Nach Redeploy: -Streams prüfen." -ForegroundColor Yellow
}

# ─── Kategorien ───
Write-Host ""
Write-Host "Kategorien:" -ForegroundColor Cyan
if ($health.categories) {
    $health.categories.PSObject.Properties | ForEach-Object {
        $cat = $_.Name
        $i = 0
        foreach ($m in $_.Value) {
            $mark = if ($m.active) { "→" } else { " " }
            Write-Host ("  $mark {0}[{1}]: {2}  @ {3}" -f $cat, $i, $m.model_name, $m.api_url)
            $i++
        }
    }
}

# ─── Optionale Log-Tail (per API, Key-geschützt) ───
if ($Logs -gt 0) {
    Write-Host ""
    Write-Host "Letzte $Logs Log-Zeilen:" -ForegroundColor Cyan
    try {
        $parsed = Invoke-ProxyApi "/logs?lines=$Logs"
        if ($parsed.lines) { $parsed.lines }
        else { $parsed | ConvertTo-Json -Depth 6 }
    } catch {
        Write-Host "  ❌ /logs nicht abrufbar: $_" -ForegroundColor Yellow
    }
}

# ─── I/O-Trace: Turns löschen (Optional) ───
if ($ClearStreams) {
    Write-Host ""
    Write-Host "Lösche I/O-Trace-Turns (Rotation erzwingen)..." -ForegroundColor Cyan
    try {
        $res = Invoke-ProxyApi "/debug/streams" -Method "DELETE"
        Write-Host ("  ✅ {0} Turns übrig" -f $res.remaining_turns)
    } catch {
        Write-Host "  ❌ DELETE /debug/streams fehlgeschlagen: $_" -ForegroundColor Yellow
    }
}

# ─── I/O-Trace: Übersicht (letzte N Turns) ───
if ($Streams -gt 0) {
    Write-Host ""
    Write-Host "Letzte $Streams Turns (I/O-Trace):" -ForegroundColor Cyan
    try {
        $list = Invoke-ProxyApi "/debug/streams?limit=$Streams"
        if (-not $list.turns -or $list.turns.Count -eq 0) {
            Write-Host "  (noch keine Turns getraced — Traffic erzeugen, dann erneut)" -ForegroundColor DarkGray
        } else {
            foreach ($t in $list.turns) {
                $cwWire = if ($t.coworker_tools_on_wire) { "TOOLS-am-Backend" } else { "keine-tools" }
                $cwCalls = if ($t.coworker_calls_seen -gt 0) { "CALLS=$($t.coworker_calls_seen)" } else { "0-calls" }
                $guide = if ($t.guidance_in_system) { "GUIDANCE" } else { "keine-guidance" }
                $err = if ($t.backend_error) { " ERR" } else { "" }
                Write-Host ("  {0}  cat={1} stream={2} {3} {4} {5}{6}" -f `
                    $t.turn_id, ($t.category ?? "?"), ($t.stream ?? "?"), $cwWire, $cwCalls, $guide, $err)
            }
        }
    } catch {
        Write-Host "  ❌ /debug/streams nicht abrufbar — Proxy-Image älter als diese Version? ($_)" -ForegroundColor Yellow
    }
}

# ─── I/O-Trace: voller Dump eines konkreten Turns ───
if ($Turn -ne "") {
    Write-Host ""
    Write-Host "I/O-Trace Turn '$Turn':" -ForegroundColor Cyan
    try {
        $detail = Invoke-ProxyApi "/debug/streams/$Turn"
        if ($detail.error) {
            Write-Host ("  Fehler: {0}" -f $detail.error) -ForegroundColor Red
        } else {
            $detail | ConvertTo-Json -Depth 12
        }
    } catch {
        Write-Host "❌ Konnte Turn nicht laden: $_" -ForegroundColor Red
    }
}
