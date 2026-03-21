$BRANCH = "feature/sigma-suppression-methodB"
$SUB_DIR = "ipole-DSA"

Write-Host "--- Starting Sync Process ---" -ForegroundColor Cyan

# 1. Sync Submodule
if (Test-Path $SUB_DIR) {
    Write-Host "Entering submodule: $SUB_DIR"
    Set-Location $SUB_DIR
    
    $subDiff = git status --porcelain
    if ($subDiff) {
        Write-Host "Committing submodule changes..." -ForegroundColor Yellow
        git add .
        git commit -m "submodule fix"
    }

    Write-Host "Pushing submodule..." -ForegroundColor Magenta
    git push origin $BRANCH
    Set-Location ..
}

# 2. Sync Main Project
Write-Host "Syncing main project..." -ForegroundColor Cyan

# 检查是否包含子模块指针更新
$mainDiff = git status --porcelain
if ($mainDiff) {
    Write-Host "Committing main project changes..." -ForegroundColor Yellow
    git add .
    git commit -m "main project sync debugging spectral index p"
}

Write-Host "Pushing main project..." -ForegroundColor Magenta
git push origin $BRANCH

Write-Host "DONE: All pushed to GitHub." -ForegroundColor Green