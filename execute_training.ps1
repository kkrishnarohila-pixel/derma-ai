$ErrorActionPreference = 'Stop'

$startTime = Get-Date
Write-Host "=== Started Training Process at $($startTime) ==="

$scriptRoot = $PSScriptRoot
if (-not $scriptRoot) { $scriptRoot = Get-Location }

Write-Host "Downloading Kaggle Datasets..."
# & .\venv\Scripts\kaggle.exe datasets download -d nayanchaure/acne-dataset -p "$scriptRoot\acne_raw" --unzip
# & .\venv\Scripts\kaggle.exe datasets download -d jessicali9530/celeba-dataset -p "$scriptRoot\celeba_raw" --unzip
# & .\venv\Scripts\kaggle.exe datasets download -d kingabzpro/cosmetics-datasets -p "$scriptRoot\cosmetics" --unzip
# & .\venv\Scripts\kaggle.exe datasets download -d nadyinky/sephora-products-and-skincare-reviews -p "$scriptRoot\sephora" --unzip

Write-Host "Running real model training pipeline..."
& python "$scriptRoot\train_models_real.py"

Write-Host "Training complete!"

$endTime = Get-Date
$duration = $endTime - $startTime
Write-Host "=== Finished Training Process at $($endTime) ==="
Write-Host "TOTAL TIME TAKEN: $($duration.Hours) hours, $($duration.Minutes) minutes, $($duration.Seconds) seconds."
