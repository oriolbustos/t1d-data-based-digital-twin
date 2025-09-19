$ErrorActionPreference = "Stop"

$pythonInstaller = "$env:TEMP\python-3.10.11-amd64.exe"
Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe" -OutFile $pythonInstaller
Start-Process -FilePath $pythonInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_test=0" -Wait
Remove-Item $pythonInstaller

$python310 = & py -3.10 -c "import sys; print(sys.executable)"
$aliasDir = Join-Path $env:LOCALAPPDATA "Programs\PythonAliases"
New-Item -ItemType Directory -Path $aliasDir -Force | Out-Null
$cmdPath = Join-Path $aliasDir "python3.10.cmd"
Set-Content -Path $cmdPath -Value "@echo off`r`n""$python310"" %*"

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
if (-not ($userPath -split ';' | Where-Object { $_ -ieq $aliasDir })) {
    [Environment]::SetEnvironmentVariable('Path', "$aliasDir;$userPath", 'User')
}
if (-not ($env:Path -split ';' | Where-Object { $_ -ieq $aliasDir })) {
    $env:Path = "$aliasDir;$env:Path"
}

python3.10 --version

