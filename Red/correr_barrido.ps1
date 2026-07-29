# correr_barrido.ps1  —  lanza (o RETOMA) el barrido completo en CPU.
# Si el proceso se cae, vuelve a ejecutar este mismo script: retoma donde iba
# (salta las corridas cuyo shard meta ya existe). Ctrl-C para detener.
#
#   .\correr_barrido.ps1              # 12 workers (por defecto)
#   .\correr_barrido.ps1 -Workers 8  # menos workers (PC mas usable)

param([int]$Workers = 12)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPy = Join-Path (Split-Path -Parent $here) ".venv-cuda\Scripts\python.exe"
if (-not (Test-Path $venvPy)) { throw "No encuentro el venv en $venvPy" }

Set-Location $here
$env:PYTHONIOENCODING = "utf-8"
Write-Host "Lanzando barrido completo (E1-E4) en CPU con $Workers workers..." -ForegroundColor Cyan
Write-Host "Progreso en vivo:  python -m barrido.estado --watch 60" -ForegroundColor DarkGray
& $venvPy -m barrido.driver --stage ALL --workers $Workers

Write-Host "`nGenerando reporte final..." -ForegroundColor Cyan
& $venvPy -m barrido.reporte
