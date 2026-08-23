# proxy-status.ps1 — Co-Worker- & Proxy-Status des DEPLOYED LocalProxy (192.168.188.134) abrufen
# Nutzt den SSH-Alias "proxy" (siehe proxy-ssh.ps1) und fragt /healthz auf dem Server selbst ab
# (localhost — keine Firewall/Coolify-Exposition nötig, kein Auth-Token nötig).
#
# Usage:
#   .\proxy-status.ps1                # Status-Übersicht (Coworker + Kategorien)
#   .\proxy-status.ps1 -Raw           # komplettes /healthz JSON ausgeben
#   .\proxy-status.ps1 -Port 8420     # abweichenden Proxy-Port (Default 9001)
#   .\proxy-status.ps1 -Logs 40       # zusätzlich die letzten N Log-Zeilen vom Server

param(
    [int]$Port = 9001,
    [switch]$Raw,
    [int]$Logs = 0
)

$sshTarget = "proxy"
$sshHost = "192.168.188.134"
if (-not (Select-String -Path "$env:USERPROFILE\.ssh\config" -Pattern "Host proxy" -Quiet -ErrorAction SilentlyContinue)) {
    Write-Host "❌ SSH-Alias 'proxy' nicht gefunden. Zuerst .\proxy-ssh.ps1 ausführen." -ForegroundColor Red
    exit 1
}

# /healthz auf dem Server abfragen (aus Sicht des Servers = localhost)
$json = ssh -o BatchMode=yes -o ConnectTimeout=5 $sshTarget "curl -s --max-time 10 http://localhost:$Port/healthz" 2>&1

if (-not $json -or $json -is [ErrorRecord] -or -not ($json -match '^\s*\{')) {
    Write-Host "❌ Keine gültige /healthz-Antwort von $sshTarget (Port $Port)." -ForegroundColor Red
    Write-Host "   Ausgabe: $json" -ForegroundColor DarkGray
    Write-Host "   Läuft der Proxy? Prüfen: ssh proxy 'systemctl status localproxy'" -ForegroundColor Yellow
    exit 1
}

try {
    $health = ($json -join "`n") | ConvertFrom-Json
} catch {
    Write-Host "❌ Konnte /healthz JSON nicht parsen: $_" -ForegroundColor Red
    exit 1
}

if ($Raw) {
    $health | ConvertTo-Json -Depth 10
    exit 0
}

# ─── Übersicht ───
Write-Host ""
Write-Host "╔══════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║  LocalProxy-Status  ($sshHost / Port $Port)  ║" -ForegroundColor Cyan
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
    if ($cw.last_error -and $cw.reachable) {
        # erreichbar aber Fehler-Eintrag alt → nur dezent zeigen
    }
} else {
    Write-Host "Co-Worker: ⚠️  Kein coworker-Block in /healthz — Proxy-Version zu alt (Update deployen!)" -ForegroundColor Yellow
    Write-Host "           Dort wird still KEIN Tool injiziert. Debug: ssh proxy 'grep -i coworker <logpath> | tail -5'" -ForegroundColor Yellow
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

# ─── Optionale Log-Tail ───
if ($Logs -gt 0) {
    Write-Host ""
    Write-Host "Letzte $Logs Log-Zeilen:" -ForegroundColor Cyan
    $logJson = ssh -o BatchMode=yes $sshTarget "curl -s --max-time 10 'http://localhost:$Port/logs?lines=$Logs'" 2>&1
    try {
        $parsed = ($logJson -join "`n") | ConvertFrom-Json
        if ($parsed.lines) { $parsed.lines }
        else { throw "no lines" }
    } catch {
        Write-Host "  (API-Log nicht abrufbar — Auth aktiviert? Stattdessen journalctl:)" -ForegroundColor Yellow
        ssh -o BatchMode=yes $sshTarget "journalctl -u localproxy -n $Logs --no-pager" 2>&1
    }
}
