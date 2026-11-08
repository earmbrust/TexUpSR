param(
    [string]$Exe = '.\build-integration\Release\trt_sr.exe',
    [string]$Model = '.\models\2x_Adore_renarchi_fp32.onnx',
    [string]$InputImage = '.\samples\busorenkin4.png',
    [int]$Runs = 3,
    [string]$OutputDir = '.\benchmark-results'
)
$ErrorActionPreference = 'Stop'
if ($Runs -lt 1) { throw 'Runs must be at least 1' }
foreach ($path in @($Exe, $Model, $InputImage)) { if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing input: $path" } }
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$configs = @(
    @{ Tile=128; Streams=2 },
    @{ Tile=256; Streams=2 },
    @{ Tile=512; Streams=2 },
    @{ Tile=256; Streams=1 }
)
$records = New-Object 'System.Collections.Generic.List[object]'
foreach ($config in $configs) {
    $tag = 't{0}-s{1}' -f $config.Tile, $config.Streams
    $output = Join-Path $OutputDir ("adore-$tag.png")
    $args = @('-i', $InputImage, '-o', $output, '-m', $Model, '--scale', '2', '--tile', [string]$config.Tile,
              '--batch', '1', '--streams', [string]$config.Streams, '--workspace', '2', '--benchmark')
    Write-Host "Preparing $tag (engine may need building)..."
    & $Exe @args 2>&1 | Out-File -LiteralPath (Join-Path $OutputDir "$tag-prepare.log") -Encoding UTF8
    if ($LASTEXITCODE -ne 0) { throw "Engine preparation failed for $tag (exit $LASTEXITCODE)" }
    for ($i=1; $i -le $Runs; $i++) {
        Write-Host "Measuring $tag run $i/$Runs"
        $log = Join-Path $OutputDir ("$tag-run$i.log")
        & $Exe @args 2>&1 | Out-File -LiteralPath $log -Encoding UTF8
        if ($LASTEXITCODE -ne 0) { throw "Benchmark failed for $tag run $i (exit $LASTEXITCODE)" }
        $content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $log).Path)
        $upscale = [regex]::Match($content, 'Upscaling wall time:\s*([0-9]+(?:\.[0-9]+)?)')
        $inference = [regex]::Match($content, 'Inference CUDA event sum:\s*([0-9]+(?:\.[0-9]+)?)')
        $total = [regex]::Match($content, 'Total application:\s*([0-9]+(?:\.[0-9]+)?)')
        $tiles = [regex]::Match($content, 'Processed\s+(\d+)/(\d+)\s+tiles')
        if (-not ($upscale.Success -and $inference.Success -and $total.Success -and $tiles.Success)) { throw "Missing benchmark data in $log" }
        $ci = [System.Globalization.CultureInfo]::InvariantCulture
        $records.Add([pscustomobject]@{
            Tile=$config.Tile; Streams=$config.Streams; Run=$i; Tiles=[int]$tiles.Groups[2].Value;
            UpscalingUs=[double]::Parse($upscale.Groups[1].Value,$ci);
            InferenceEventSumUs=[double]::Parse($inference.Groups[1].Value,$ci);
            TotalApplicationUs=[double]::Parse($total.Groups[1].Value,$ci)
        })
    }
}
$csv = Join-Path $OutputDir 'results.csv'
$records | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $csv
foreach ($config in $configs) {
    $times = @($records | Where-Object { $_.Tile -eq $config.Tile -and $_.Streams -eq $config.Streams } | ForEach-Object { $_.UpscalingUs } | Sort-Object)
    $count = $times.Count
    $median = if ($count % 2 -eq 1) { $times[[int][math]::Floor($count / 2)] } else { ($times[$count/2-1] + $times[$count/2])/2 }
    Write-Host ('Tile {0}, streams {1}: median upscaling {2:N2} ms ({3} cached runs)' -f $config.Tile, $config.Streams, ($median/1000), $count)
}
Write-Host "Benchmark CSV: $csv"
