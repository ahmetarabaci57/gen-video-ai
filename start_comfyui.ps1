$comfyRoot = "C:\Users\hp\Desktop\ComfyUI"
$python = Join-Path $comfyRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Error "ComfyUI venv not found at $python"
    exit 1
}

Set-Location $comfyRoot
& $python main.py --listen 127.0.0.1 --port 8188 --lowvram --force-fp16 --preview-method none
