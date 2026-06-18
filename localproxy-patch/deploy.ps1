<#
.SYNOPSIS
    Deployt proxy.py und webui.py per SCP auf den Proxy-Server und startet den Service neu.
#>

$remote = "proxy:~/localproxy/"
$healthUrl = "http://192.168.188.134:9001/healthz"

Write-Host "📤 Deploye proxy.py und webui.py ..." -ForegroundColor Cyan
scp "d:\GitHub\Asus G10\localproxy-patch\proxy.py" $remote
scp "d:\GitHub\Asus G10\localproxy-patch\webui.py" $remote

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ SCP fehlgeschlagen!" -ForegroundColor Red
    exit 1
}

Write-Host "🔄 Starte localproxy neu ..." -ForegroundColor Cyan
ssh proxy "systemctl restart localproxy"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Restart fehlgeschlagen!" -ForegroundColor Red
    exit 1
}

Write-Host "⏳ Warte auf Start ..." -ForegroundColor Yellow
Start-Sleep -Seconds 3

Write-Host "🔍 Prüfe Health-Endpoint ..." -ForegroundColor Cyan
try {
    $response = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 10
    $content = $response.Content | ConvertFrom-Json
    Write-Host "✅ Proxy läuft! Status: $($content.status)" -ForegroundColor Green
    Write-Host "   vLLM Key konfiguriert: $($content.vllm_api_key_configured)" -ForegroundColor Green
    Write-Host "   Version: $($content.version)" -ForegroundColor Gray
} catch {
    Write-Host "❌ Health-Check fehlgeschlagen: $_" -ForegroundColor Red
    exit 1
}

Write-Host "`n✨ Deploy erfolgreich!" -ForegroundColor Green
