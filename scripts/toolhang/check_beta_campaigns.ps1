# Run once per heartbeat. Keep the last JSON line because registry sync also prints.
$ErrorActionPreference = 'Stop'
$targets = @(
    @{ Port = 1022; Id = 'TOOL_HANG-2026-09-25-01' },
    @{ Port = 1024; Id = 'TOOL_HANG-2026-09-26-02' }
)
$results = [ordered]@{}
foreach ($target in $targets) {
    $command = 'cd /root/workspace/baojiachun/scout && /root/workspace/baojiachun/.venv_mg/bin/python -m scripts.atom.beta_gate check --due-only --root data/TOOL_HANG/full/' + $target.Id
    try {
        $lines = @(& ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=10 -o ServerAliveCountMax=2 -p $target.Port root@106.14.2.243 $command)
        if ($LASTEXITCODE -ne 0) { throw "Remote check failed: $LASTEXITCODE" }
        $results[[string]$target.Port] = $lines[-1] | ConvertFrom-Json
    } catch {
        $results[[string]$target.Port] = @{ error = $_.Exception.Message }
    }
}
$results | ConvertTo-Json -Depth 15 -Compress
