# Script para configurar Git y conectar con GitHub
# Ejecutar: .\configurar_git.ps1

Write-Host "Configurando repositorio Git..." -ForegroundColor Green

# Verificar si git está instalado
try {
    $gitVersion = git --version
    Write-Host "Git encontrado: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Git no está instalado." -ForegroundColor Red
    Write-Host "Por favor instala Git desde: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

# Navegar al directorio del proyecto
Set-Location $PSScriptRoot

# Inicializar repositorio si no existe
if (-not (Test-Path .git)) {
    Write-Host "Inicializando repositorio Git..." -ForegroundColor Green
    git init
} else {
    Write-Host "Repositorio Git ya inicializado." -ForegroundColor Yellow
}

# Agregar el repositorio remoto
$remoteUrl = "https://github.com/diazdiazdaniel59-droid/Daniel.diaz.git"
Write-Host "Configurando remoto 'origin'..." -ForegroundColor Green
git remote remove origin 2>$null
git remote add origin $remoteUrl
git remote -v

# Agregar todos los archivos
Write-Host "Agregando archivos al staging..." -ForegroundColor Green
git add .

# Verificar si hay cambios para commit
$status = git status --porcelain
if ($status) {
    Write-Host "Archivos listos para commit:" -ForegroundColor Green
    git status
    
    Write-Host "`nPara hacer commit y push, ejecuta:" -ForegroundColor Yellow
    Write-Host "git commit -m 'Initial commit'" -ForegroundColor Cyan
    Write-Host "git branch -M main" -ForegroundColor Cyan
    Write-Host "git push -u origin main" -ForegroundColor Cyan
} else {
    Write-Host "No hay cambios para commit." -ForegroundColor Yellow
}

Write-Host "`nConfiguración completada!" -ForegroundColor Green

