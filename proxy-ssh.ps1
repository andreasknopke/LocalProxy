# proxy-ssh.ps1 — SSH-Helfer für den LocalProxy-Server (192.168.188.134)
# Erzeugt einen dedizierten SSH-Key (ohne Passphrase) und richtet passwortloses Einloggen ein.

$envFile = Join-Path $PSScriptRoot ".env"
$proxyHost = "192.168.188.134"
$keyName = "proxy-key"           # dedizierter Key, nicht id_rsa
$keyPath = "$env:USERPROFILE\.ssh\$keyName"
$keyPub = "$keyPath.pub"

# ─── Hilfsfunktion: .env auslesen ───
function Get-EnvValue($key) {
    $line = Select-String -Path $envFile -Pattern "^$key=" | Select-Object -First 1
    if ($line) { return ($line -split '=', 2)[1].Trim() }
    return $null
}

$proxyUser = Get-EnvValue "PROXY_AUTH_USERNAME"
$proxyPass = Get-EnvValue "PROXY_AUTH_PASSWORD"

if (-not $proxyUser -or -not $proxyPass) {
    Write-Host "❌ Zugangsdaten nicht gefunden in $envFile" -ForegroundColor Red
    exit 1
}

Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║    Proxy-SSH einrichten                  ║" -ForegroundColor Cyan
Write-Host "║    Server: $proxyUser@$proxyHost" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan

# ─── 1. Dedizierten SSH-Key ohne Passphrase erzeugen ───
if (Test-Path $keyPub) {
    Write-Host "`n✅ Dedizierter SSH-Key existiert bereits: $keyName" -ForegroundColor Green
} else {
    Write-Host "`n🔑 Erzeuge dedizierten SSH-Key '$keyName' (ohne Passphrase)..." -ForegroundColor Yellow
    # Wichtig: Leeren String pipen, -N """" funktioniert in PowerShell NICHT
    '' | ssh-keygen -t ed25519 -f $keyPath -q
    if ($LASTEXITCODE -eq 0 -and (Test-Path $keyPub)) {
        Write-Host "✅ SSH-Key erstellt: $keyPath" -ForegroundColor Green
    } else {
        Write-Host "❌ Konnte SSH-Key nicht erstellen" -ForegroundColor Red
        exit 1
    }
}

# ─── 2. Public-Key auf Server kopieren ───
Write-Host "`n📤 Kopiere Public-Key auf den Server (einmalig Passwort nötig)..." -ForegroundColor Yellow

# Public-Key per Pipe in authorized_keys schreiben (sicherer als echo im remote-cmd)
Get-Content $keyPub -Raw | ssh -o StrictHostKeyChecking=accept-new "$proxyUser@$proxyHost" "cat >> ~/.ssh/authorized_keys && sort -u ~/.ssh/authorized_keys -o ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys && echo 'OK'"

if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ Public-Key auf Server installiert!" -ForegroundColor Green
} else {
    Write-Host "`n⚠️  Automatisch nicht geklappt. Mach es manuell:" -ForegroundColor Yellow
    Write-Host "   (1) Verbinde: ssh $proxyUser@$proxyHost" -ForegroundColor White
    Write-Host "   (2) Führe aus: ssh-copy-id $proxyUser@$proxyHost" -ForegroundColor White
    Write-Host "   (3) Oder füge diesen Key in ~/.ssh/authorized_keys ein:" -ForegroundColor White
    Write-Host "`n$(Get-Content $keyPub -Raw)" -ForegroundColor DarkGray
}

# ─── 3. SSH-Config-Eintrag ───
$sshConfigPath = "$env:USERPROFILE\.ssh\config"
$configEntry = @"

Host proxy
    HostName $proxyHost
    User $proxyUser
    IdentityFile $keyPath
    ServerAliveInterval 60
    StrictHostKeyChecking accept-new

Host $proxyHost
    User $proxyUser
    IdentityFile $keyPath
    ServerAliveInterval 60
"@

if (-not (Test-Path $sshConfigPath)) {
    # Config existiert noch nicht → neu anlegen
    Add-Content -Path $sshConfigPath -Value $configEntry
    Write-Host "`n✅ SSH-Config erstellt (Host 'proxy' + IP-Eintrag)" -ForegroundColor Green
} elseif (-not (Select-String -Path $sshConfigPath -Pattern "Host proxy" -Quiet)) {
    # Host proxy fehlt → anfügen
    Add-Content -Path $sshConfigPath -Value $configEntry
    Write-Host "`n✅ SSH-Config erweitert (Host 'proxy' + IP-Eintrag)" -ForegroundColor Green
} elseif (-not (Select-String -Path $sshConfigPath -Pattern "proxy-key" -Quiet)) {
    # proxy-Eintrag existiert, aber ohne IdentityFile → per sed ersetzen
    Write-Host "`n⚠️  SSH-Config für 'proxy' fehlt IdentityFile. Bitte manuell ergänzen:" -ForegroundColor Yellow
    Write-Host "   Bearbeite $sshConfigPath" -ForegroundColor Cyan
    Write-Host "   und füge unter 'Host proxy' ein:  IdentityFile $keyPath" -ForegroundColor Cyan
} else {
    Write-Host "`n✅ SSH-Config ist bereits korrekt" -ForegroundColor Green
}

# ─── 4. Verbindung testen ───
Write-Host "`n🔍 Teste passwortlose Verbindung..." -ForegroundColor Yellow
$result = ssh -o BatchMode=yes -o ConnectTimeout=5 "proxy" "echo VERBINDUNG_OK" 2>&1

if ($result -eq "VERBINDUNG_OK") {
    Write-Host "✅ Passwortlose Verbindung erfolgreich! Kein Passwort mehr nötig." -ForegroundColor Green
} else {
    Write-Host "⚠️  BatchMode-Test fehlgeschlagen. Versuche normalen Test..." -ForegroundColor Yellow
    ssh "proxy" "echo '✅ Verbindung steht'"
}

# ─── 5. Kurzbefehle anzeigen ───
Write-Host "`n📋 Verfügbare Befehle:" -ForegroundColor Cyan
Write-Host "  ssh proxy" -ForegroundColor White
Write-Host "  ssh proxy 'systemctl status localproxy'" -ForegroundColor White
Write-Host "  ssh proxy 'journalctl -u localproxy -n 50 --no-pager'" -ForegroundColor White
Write-Host "  ssh proxy 'systemctl restart localproxy'" -ForegroundColor White
