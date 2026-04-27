$root = if ($env:UC_REPO_ROOT) { $env:UC_REPO_ROOT } else { (Resolve-Path "$PSScriptRoot\..\..").Path }
Set-Location $root
$env:CUDA_VISIBLE_DEVICES = '0'
$env:PYTHONUNBUFFERED = '1'

# Wait for tinyllama run (PID 46284) to finish, signalled by the JSON file.
$tl = 'results\claim21_headline_tinyllama_rho0.01.json'
while (-not (Test-Path $tl)) { Start-Sleep -Seconds 60 }

foreach ($m in @('olmo2_1b', 'qwen3_1.7b', 'smollm2_1.7b')) {
    $out = "results\claim21_headline_${m}_rho0.01.json"
    $log = "results\claim21_headline_${m}_rho0.01.log"
    $err = "results\claim21_headline_${m}_rho0.01.err"
    Write-Output "==> starting $m  $(Get-Date -Format o)" | Out-File -Append -FilePath $log -Encoding utf8
    & "python" `
        -u scripts\overlay\claim21_headline.py `
        --model $m --rho 0.010 --device cuda:0 --out $out `
        1>>$log 2>>$err
    Write-Output "==> finished $m  $(Get-Date -Format o)" | Out-File -Append -FilePath $log -Encoding utf8
}
Write-Output "ALL DONE $(Get-Date -Format o)" | Out-File -Append -FilePath results\claim21_headline_DONE.txt -Encoding utf8
