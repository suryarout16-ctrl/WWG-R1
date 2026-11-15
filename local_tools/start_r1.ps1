<#
.SYNOPSIS
  今回の実機R1設定で LAN接続待ち → DDS接続 → 8770パネル → ブラウザ表示。
  pwsh -NoProfile -ExecutionPolicy Bypass -File .\local_tools\start_r1.ps1
  設定値はr1_startup.json、接続先・鍵・CUDAのPC固有情報はr1_local_settings.ps1。
  仮想器は起動しない。起立・歩行・発話・通話開始はパネルから操作する。
#>
[CmdletBinding()]
param(
    [ValidateSet("openai", "strata")][string]$Backend = "openai",
    [ValidateRange(1024, 65535)][int]$Port = 8770,
    [ValidateNotNullOrEmpty()][string]$Mic = "Nothing Ear (3a)",
    [string]$Interface = "Ethernet",
    [string]$PcIp = "192.168.123.164",
    [string]$RobotIp = "192.168.123.161",
    [string]$Materials = "logs\training\live_materials.json",
    [ValidateRange(0, 3600)][int]$WaitForLanSeconds = 120,
    [switch]$Offline,
    [switch]$Restart,
    [switch]$Stop,
    [switch]$NoBrowser,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Repo ".pixi\envs\default\python.exe"
$Logs = Join-Path $Repo "logs"
$Url = "http://127.0.0.1:$Port/"
function Say([string]$m, [string]$c = "Gray") { Write-Host $m -ForegroundColor $c }
function Quote-Argument([object]$value) {
    $s = [Convert]::ToString($value, [Globalization.CultureInfo]::InvariantCulture)
    if ($s -match '["\r\n]') { throw "起動引数に引用符や改行は使えません" }
    if (-not $s -or $s -match '\s') { return '"' + ($s -replace '(\\+)$', '$1$1') + '"' }
    return $s
}
function Get-PyProcs([string]$pattern) {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
        Where-Object { $_.CommandLine -and $_.CommandLine -match $pattern }
}
function Get-Panels {
    Get-PyProcs 'web_ui\.py' | Where-Object { $_.ExecutablePath -eq $Py }
}
function Get-PanelPort($proc) {
    if ($proc.CommandLine -match '--port\s+(\d+)') { return [int]$Matches[1] }
    return 8765
}
function Get-PanelState([int]$PanelPort) {
    try { Invoke-RestMethod "http://127.0.0.1:$PanelPort/api/state" -TimeoutSec 2 } catch { $null }
}
function Assert-Physical($state) {
    if (-not $state -or $state.connection_kind -ne "robot" -or $state.virtual -or $state.local -or -not $state.fsm_ok -or $state.dds_address -ne $PcIp) {
        throw "実機パネルの状態を確認できません。別のモードで起動せず停止します"
    }
}
function Stop-IdlePanel($proc) {
    $state = Get-PanelState (Get-PanelPort $proc)
    if (-not ($state -and $state.local -and $state.connection_kind -eq "local" -and -not $state.virtual)) {
        Assert-Physical $state
    }
    # ローカル試験の返答待ちは、処理中の仕事がなければ実機切替時に解除できる。
    # 実機パネルや、通話・録音・動作中のパネルには停止APIを送らない。
    $localReplyWait = -not $Offline -and -not $Stop -and $state.local -and $state.connection_kind -eq "local" -and
        -not $state.virtual -and -not $state.dds_address -and $state.body.dry -and $state.posture.dry -and
        $state.training.active -and $state.training.phase -eq "await_reply" -and $state.training.awaiting_reply
    $working = $state.busy -or $state.body.busy -or $state.body.hold_active -or $state.body.stop_required -or
        $state.posture.busy -or $state.recording -or $state.call.active -or $state.call.input_closing
    if ($localReplyWait -and -not $working) {
        Say "ローカル研修の返答待ちを終了して、実機用へ切り替えます" "Cyan"
        $panelPort = Get-PanelPort $proc
        $response = Invoke-RestMethod -Uri "http://127.0.0.1:$panelPort/api/training/stop" -Method Post `
            -ContentType "application/json" -Body "{}" -TimeoutSec 5
        if (-not $response.ok) { throw "ローカル研修の待機を終了できませんでした" }
        $state = Get-PanelState $panelPort
        if (-not ($state -and $state.local -and $state.connection_kind -eq "local" -and
                  -not $state.virtual -and -not $state.dds_address -and $state.body.dry -and $state.posture.dry)) {
            throw "切替前のローカル状態を確認できません。パネルは終了しません"
        }
    }
    if ($state.busy -or $state.body.busy -or $state.body.hold_active -or $state.body.stop_required -or
        $state.posture.busy -or $state.training.active -or $state.recording -or
        $state.call.active -or $state.call.input_closing) {
        throw "パネルを使用中です。通話・研修を終了し、動作が終わってから再実行してください"
    }
    Say "待機中のパネルを終了します (PID $($proc.ProcessId))" "Yellow"
    & taskkill.exe /PID $proc.ProcessId /T /F | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "パネルを終了できませんでした" }
}
function Test-TargetPanel($state) {
    if (-not $state -or $state.virtual) { return $false }
    if ($Offline) { return ($state.local -and $state.connection_kind -eq "local" -and -not $state.dds_address) }
    return (-not $state.local -and $state.connection_kind -eq "robot" -and $state.fsm_ok -and $state.dds_address -eq $PcIp)
}
function Open-Panel {
    Say ("$PanelLabel : $Url") "Green"
    if (-not $NoBrowser) { Start-Process $Url }
}

try {
    if (-not (Test-Path -LiteralPath $Py)) { throw "Python がありません。先にリポジトリで pixi install を実行してください" }
    Set-Location $Repo
    if ($Stop) {
        foreach ($panel in @(Get-Panels)) { Stop-IdlePanel $panel }
        Say "パネルを停止しました" "Green"
        exit 0
    }
    $StartupConfig = Get-Content -LiteralPath (Join-Path $PSScriptRoot "r1_startup.json") -Raw -Encoding UTF8 | ConvertFrom-Json
    $R1CudaDllDir = ""
    $settings = Join-Path $PSScriptRoot "r1_local_settings.ps1"
    if (Test-Path -LiteralPath $settings) { . $settings }
    else { Say "PC固有設定なし。Strata接続とCUDA DLLの設定を確認してください" "Yellow" }

    $values = [ordered]@{
        "-i" = $Interface; "--dds-address" = $PcIp; "--port" = $Port
        "--body-control" = $null; "--velocity-127-compatible" = $null
        "--forward-speed" = $StartupConfig.forward_speed; "--forward-duration" = $StartupConfig.forward_duration
        "--hold-linear-speed" = $StartupConfig.hold_linear_speed; "--hold-yaw-speed" = $StartupConfig.hold_yaw_speed
        "--materials-file" = $Materials; "--mentor-backend" = $Backend; "--persona" = "mentor"
        "--model" = $StartupConfig.model; "--strata-timeout" = $StartupConfig.strata_timeout
        "--volume" = $StartupConfig.volume; "--gain" = $StartupConfig.gain_db
        "--mic-device" = $Mic; "--call-threshold" = $StartupConfig.call_threshold; "--call-silence" = $StartupConfig.call_silence
        "--stt-model" = $StartupConfig.stt_model; "--stt-device" = $StartupConfig.stt_device
        "--stt-compute-type" = $StartupConfig.stt_compute_type; "--stt-language" = $StartupConfig.stt_language
        "--voice" = $StartupConfig.voice; "--voice-en" = $StartupConfig.voice_en
    }
    $PanelLabel = "実機パネル"
    if ($Offline) {
        $PanelLabel = "R1未接続のローカル試験パネル"
        $values.Remove("-i"); $values.Remove("--dds-address")
        $values["--local"] = $null
    }
    $pargs = @("-X", "utf8", "-B", "r1_chat\web_ui.py")
    foreach ($entry in $values.GetEnumerator()) {
        $pargs += $entry.Key
        if ($null -ne $entry.Value) { $pargs += Quote-Argument $entry.Value }
    }
    if ($R1CudaDllDir) {
        if (-not (Test-Path -LiteralPath $R1CudaDllDir)) { throw "設定されたCUDA DLLフォルダーがありません" }
        $pargs += @("--cuda-dll-dir", (Quote-Argument $R1CudaDllDir))
    }
    $argline = $pargs -join " "
    Say "設定: $Backend / $($StartupConfig.model) / $Mic"
    Say "音量 $($StartupConfig.volume)、増幅 +$($StartupConfig.gain_db) dB、押下歩行 $($StartupConfig.hold_linear_speed) m/s、旋回 $($StartupConfig.hold_yaw_speed) rad/s"
    Say "発話判定 $($StartupConfig.call_threshold)、STT $($StartupConfig.stt_model)/$($StartupConfig.stt_device)/$($StartupConfig.stt_language)、パネル $Port"
    if ($DryRun) {
        $shown = $argline
        if ($R1CudaDllDir) { $shown = $shown.Replace((Quote-Argument $R1CudaDllDir), "<CUDA>") }
        if ($Offline) { Say "[DryRun] ローカル試験 → パネル → ブラウザ（LAN/DDS/仮想器は使用しません）" "Cyan" }
        else { Say "[DryRun] LAN待ち最大$WaitForLanSeconds 秒 → 必要な場合だけEthernet設定確認 → ping → 実機パネル → ブラウザ" "Cyan" }
        Say "[DryRun] python $shown" "Cyan"
        Say "[DryRun] ロボット・ネットワーク設定・プロセス・ブラウザは変更しません" "Cyan"
        exit 0
    }

    if (@(Get-PyProcs 'virtual_r1\.py').Count) { throw "仮想R1が動いています。実機との混在を避けるため起動しません" }
    $panels = @(Get-Panels)
    if ($Offline) {
        foreach ($existing in $panels) {
            $existingState = Get-PanelState (Get-PanelPort $existing)
            if (-not ($existingState -and $existingState.local -and $existingState.connection_kind -eq "local" -and -not $existingState.virtual)) {
                throw "実機用または状態不明のパネルが動いています。ローカル試験へは切り替えません"
            }
        }
    }
    foreach ($panel in $panels) {
        if (-not $Restart -and (Get-PanelPort $panel) -eq $Port -and $panel.CommandLine.EndsWith($argline)) {
            $state = Get-PanelState $Port
            if (Test-TargetPanel $state) {
                Open-Panel
                exit 0
            }
        }
    }

    if (-not $Offline) {
    $nic = Get-NetAdapter -Name $Interface -ErrorAction SilentlyContinue
    if (-not $nic) { throw "R1用アダプター '$Interface' がありません。ケーブル・USB LANを確認してください" }
    $deadline = [DateTime]::UtcNow.AddSeconds($WaitForLanSeconds)
    if ($nic.Status -ne "Up") { Say "R1用 '$Interface' の接続を待ちます（最大$WaitForLanSeconds 秒）。ケーブルとR1の電源を確認してください" "Yellow" }
    while ($nic.Status -ne "Up" -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 1
        $nic = Get-NetAdapter -Name $Interface -ErrorAction Stop
    }
    if ($nic.Status -ne "Up") { throw "LAN未接続です。ケーブルを接続して同じコマンドを再実行してください" }
    $networkProfiles = @()
    for ($i = 0; $i -lt 10; $i++) {
        $networkProfiles = @(Get-NetConnectionProfile -InterfaceIndex $nic.ifIndex -ErrorAction SilentlyContinue)
        if ($networkProfiles.Count -eq 1) { break }
        Start-Sleep -Milliseconds 500
    }
    if ($networkProfiles.Count -ne 1) { throw "Ethernetのネットワーク識別が完了していません。少し待って再実行してください" }
    $ip = Get-NetIPAddress -InterfaceIndex $nic.ifIndex -IPAddress $PcIp -ErrorAction SilentlyContinue
    $category = $networkProfiles[0].NetworkCategory
    if (-not $ip -or $ip.PrefixLength -ne 24 -or "$category" -ne "Private") {
        Say "'$Interface' のR1用IPとPrivate設定を確認します。必要な場合はWindowsの管理者確認が表示されます" "Yellow"
        $helper = Join-Path $PSScriptRoot "prepare_r1_lan.ps1"
        $helperArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", (Quote-Argument $helper),
                        "-InterfaceIndex", $nic.ifIndex, "-PcIp", (Quote-Argument $PcIp))
        $admin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
        $options = @{FilePath="powershell.exe"; ArgumentList=$helperArgs; WindowStyle="Hidden"; Wait=$true; PassThru=$true}
        if (-not $admin) { $options.Verb = "RunAs" }
        $prepared = Start-Process @options
        if ($prepared.ExitCode -ne 0) { throw "R1用LAN設定を完了できませんでした。logsのprepare_r1_lanログを確認してください" }
    }
    $pingOk = Test-Connection -ComputerName $RobotIp -Count 1 -Quiet
    if (-not $pingOk) { Say "LANは接続済みです。R1 ($RobotIp) の応答を待ちます" "Yellow" }
    while (-not $pingOk -and [DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 1
        $pingOk = Test-Connection -ComputerName $RobotIp -Count 1 -Quiet
    }
    if (-not $pingOk) { throw "R1 ($RobotIp) にpingが届きません。電源とLANを確認して再実行してください" }
    Say "R1 ($RobotIp) ping OK / DDS送信元 $PcIp" "Green"
    }
    else { Say "ローカル試験: PCで音声を再生し、実機への接続・指令送信は行いません" "Cyan" }
    foreach ($panel in $panels) { Stop-IdlePanel $panel }
    if ($panels.Count) { Start-Sleep -Seconds 1 }
    if (Get-NetTCPConnection -LocalAddress 127.0.0.1 -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "ポート $Port は別のプログラムが使っています"
    }
    New-Item -ItemType Directory -Force -Path $Logs | Out-Null
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $pout = Join-Path $Logs "panel_${Port}_$stamp.out.log"
    $p = Start-Process -FilePath $Py -ArgumentList $pargs -WorkingDirectory $Repo -WindowStyle Hidden `
        -RedirectStandardOutput $pout -RedirectStandardError "$pout.err" -PassThru
    Say "$PanelLabel を起動中 (PID $($p.Id))。音声認識の準備を待ちます"
    $ready = $false
    for ($i = 0; $i -lt 120; $i++) {
        Start-Sleep -Milliseconds 500
        $p.Refresh()
        if ($p.HasExited) { break }
        $state = Get-PanelState $Port
        if (Test-TargetPanel $state) { $ready = $true; break }
    }
    if (-not $ready) {
        Say "$PanelLabel を起動できませんでした。ログ末尾:" "Red"
        foreach ($f in @($pout, "$pout.err")) { if (Test-Path -LiteralPath $f) { Get-Content -LiteralPath $f -Tail 12 -Encoding UTF8 } }
        throw "接続とログを確認し、同じコマンドで再実行してください"
    }
    Open-Panel
    Say "通話はパネルの「通話開始」から。起立や歩行は自動開始しません"
    exit 0
}
catch {
    Say ("起動を中断しました: " + $_.Exception.Message) "Red"
    exit 1
}
