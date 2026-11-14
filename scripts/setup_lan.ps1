<#
.SYNOPSIS
  R1 と有線LANでつなぐための設定を1本でまとめて行う（Windows 11）。

  1. 有線LANアダプタを探す（USB-LAN でも可）
  2. 名前を英数字（既定 Ethernet）に変える
  3. 固定IP 192.168.123.164/24（ゲートウェイなし）にする
  4. ネットワークを「プライベート」にする
  5. R1(192.168.123.161) に ping
  6. （任意）R1 に日本語で一言話させる

  使い方（リポジトリのルートで。管理者権限は自動で求められる）:
    powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1            # 設定 + ping + 発話テスト
    powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1 -DryRun    # 何も変えずに、やることだけ表示
    powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1 -Restore   # 元に戻す（DHCP に戻す）
#>
[CmdletBinding()]
param(
    [string]$Adapter = "",                 # 対象アダプタ名。空なら自動で探す
    [string]$NewName = "Ethernet",
    [string]$PcIp = "192.168.123.164",
    [string]$RobotIp = "192.168.123.161",
    [int]$PrefixLength = 24,
    [string]$SayText = "こんにちは、僕はR1です。接続できました。",
    [switch]$SkipSay,
    [switch]$Restore,
    [switch]$DryRun,
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BackupFile = Join-Path $RepoRoot ".lan_backup.json"   # 元の設定（.gitignore 済み）

function Say([string]$m, [string]$c = "Gray") { Write-Host $m -ForegroundColor $c }
function Step([string]$m) { Write-Host ""; Write-Host "== $m" -ForegroundColor Cyan }
function Do-Change([string]$what, [scriptblock]$action) {
    if ($DryRun) { Say "  [DryRun] $what" "Yellow" } else { Say "  $what"; & $action }
}

# ---- 管理者で動いていなければ、管理者で開き直す（DryRun はそのまま）
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin -and -not $DryRun) {
    Say "管理者権限が必要です。確認画面で「はい」を押してください。" "Yellow"
    $argList = @("-NoExit", "-ExecutionPolicy", "Bypass", "-File", "`"$PSCommandPath`"")
    foreach ($k in $PSBoundParameters.Keys) {
        $v = $PSBoundParameters[$k]
        if ($v -is [switch]) { if ($v) { $argList += "-$k" } } else { $argList += "-$k"; $argList += "`"$v`"" }
    }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList -WorkingDirectory $RepoRoot
    exit
}

# ---- ログ（画面に出た内容をそのまま logs\setup_lan_*.log に残す）
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogPath = Join-Path $LogDir ("setup_lan_{0}{1}.log" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $(if ($DryRun) { "_dryrun" } elseif ($Restore) { "_restore" } else { "" }))
Start-Transcript -Path $LogPath | Out-Null
$ifIndex = $null
try {

# ---- 1. アダプタを探す
Step "1. 有線LANアダプタを探す"
$candidates = Get-NetAdapter -Physical | Where-Object {
    $_.MediaType -eq "802.3" -and $_.InterfaceDescription -notmatch "Remote NDIS|Virtual|Bluetooth|Hyper-V|VPN|TAP|Tailscale"
}
if ($Adapter) { $candidates = Get-NetAdapter -Name $Adapter }
elseif (Get-NetAdapter -Name $NewName -ErrorAction SilentlyContinue) { $candidates = Get-NetAdapter -Name $NewName }
$candidates = @($candidates)
if ($candidates.Count -eq 0) { throw "有線LANアダプタが見つかりません。USB-LANアダプタなら挿してから実行してください。" }
if ($candidates.Count -gt 1) {
    Say "有線LANアダプタが複数あります:" "Yellow"
    for ($i = 0; $i -lt $candidates.Count; $i++) { Say ("  [{0}] {1}  ({2}, {3})" -f $i, $candidates[$i].Name, $candidates[$i].InterfaceDescription, $candidates[$i].Status) }
    $pick = Read-Host "R1 につなぐものの番号"
    $candidates = @($candidates[[int]$pick])
}
$nic = $candidates[0]
Say ("  対象: {0}  ({1}, 状態 {2})" -f $nic.Name, $nic.InterfaceDescription, $nic.Status) "Green"
$ifIndex = $nic.ifIndex

# ---- 元に戻す
if ($Restore) {
    Step "元の設定に戻す"
    $b = if (Test-Path $BackupFile) { Get-Content $BackupFile -Raw | ConvertFrom-Json } else { $null }
    $restoreIps = @(if ($b -and $b.Dhcp -eq "Disabled") { $b.Addresses | Where-Object { $_ -notlike "169.254.*" } })
    Do-Change $(if ($restoreIps) { "固定IPを外し、元のIP ($($restoreIps -join ', ')) に戻す" } else { "固定IPを外して DHCP に戻す" }) {
        Get-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
        if ($restoreIps) {
            foreach ($a in $restoreIps) { $ip, $len = $a -split "/"; New-NetIPAddress -InterfaceIndex $ifIndex -IPAddress $ip -PrefixLength ([int]$len) | Out-Null }
        } else { Set-NetIPInterface -InterfaceIndex $ifIndex -Dhcp Enabled }
    }
    if ($b) { Say "  名前は $($nic.Name) のままにしています（変更前は $($b.Name)。戻すなら設定画面で）。" }
    Say "完了" "Green"; exit
}

# ---- 確認
$oldIps = @(Get-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | ForEach-Object { "$($_.IPAddress)/$($_.PrefixLength)" })
Say ("  いまの IPv4: {0}" -f $(if ($oldIps) { $oldIps -join ", " } else { "なし" }))
Say "  これから: 名前 → $NewName / IP → $PcIp/$PrefixLength（ゲートウェイなし、ほかの IPv4 は外す）/ プライベート" "Yellow"
if (-not $Yes -and -not $DryRun) {
    if ((Read-Host "実行しますか？ (y/N)") -notmatch '^[yY]') { Say "中止しました"; exit }
}
if (-not $DryRun) {
    @{ Name = $nic.Name; Addresses = $oldIps; Dhcp = (Get-NetIPInterface -InterfaceIndex $ifIndex -AddressFamily IPv4).Dhcp.ToString(); Saved = (Get-Date).ToString("s") } |
        ConvertTo-Json | Set-Content -Path $BackupFile -Encoding UTF8
}

# ---- 2. 名前を英数字に
Step "2. アダプタ名を $NewName にする"
if ($nic.Name -ceq $NewName) { Say "  すでに $NewName です" }
else {
    if (Get-NetAdapter -Name $NewName -ErrorAction SilentlyContinue) { throw "$NewName という名前のアダプタが別にあります。-NewName で別の名前を指定してください。" }
    Do-Change "$($nic.Name) → $NewName" { Rename-NetAdapter -Name $nic.Name -NewName $NewName }
}

# ---- 3. 固定IP
Step "3. 固定IP $PcIp/$PrefixLength"
Do-Change "DHCP をオフにし、ほかの IPv4 とゲートウェイを外して $PcIp を付ける" {
    Set-NetIPInterface -InterfaceIndex $ifIndex -Dhcp Disabled
    Get-NetRoute -InterfaceIndex $ifIndex -DestinationPrefix "0.0.0.0/0" -ErrorAction SilentlyContinue | Remove-NetRoute -Confirm:$false -ErrorAction SilentlyContinue
    Get-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object { $_.IPAddress -ne $PcIp } | Remove-NetIPAddress -Confirm:$false -ErrorAction SilentlyContinue
    if (-not (Get-NetIPAddress -InterfaceIndex $ifIndex -IPAddress $PcIp -ErrorAction SilentlyContinue)) {
        New-NetIPAddress -InterfaceIndex $ifIndex -IPAddress $PcIp -PrefixLength $PrefixLength | Out-Null
    }
}

# ---- ケーブルのリンク待ち
Step "ケーブルの接続を確認"
for ($i = 0; $i -lt 30; $i++) {
    $st = (Get-NetAdapter -InterfaceIndex $ifIndex).Status
    if ($st -eq "Up") { break }
    if ($i -eq 0) { Say "  リンクが $st です。R1 のうなじのLANポートとPCをつないでください（30秒待ちます）…" "Yellow" }
    if ($DryRun) { break }
    Start-Sleep -Seconds 1
}
$st = (Get-NetAdapter -InterfaceIndex $ifIndex).Status
Say "  リンク: $st" $(if ($st -eq "Up") { "Green" } else { "Red" })

# ---- 4. プライベート
Step "4. ネットワークをプライベートにする"
if ($st -ne "Up") { Say "  リンクが無いので後回し（つないだらもう一度実行してください）" "Yellow" }
else {
    Do-Change "NetworkCategory → Private" {
        try { Set-NetConnectionProfile -InterfaceIndex $ifIndex -NetworkCategory Private }
        catch { Say "  プライベートにできませんでした: $($_.Exception.Message)" "Yellow" }
    }
}

# ---- 5. ping
Step "5. R1 ($RobotIp) に ping"
$pingOk = $false
if ($st -eq "Up" -and -not $DryRun) {
    Start-Sleep -Seconds 2
    $pingOk = Test-Connection -ComputerName $RobotIp -Count 3 -Quiet
}
if ($DryRun) { Say "  [DryRun] Test-Connection $RobotIp" "Yellow" }
elseif ($pingOk) { Say "  応答あり" "Green" }
else {
    Say "  応答がありません。確認: R1 の起動完了（「Zero Torque Mode. StartUp Completed.」）/ ケーブル / 同じIPのPCがほかにいないか" "Red"
    exit 1
}

# ---- 6. 発話テスト
if (-not $SkipSay -and -not $DryRun) {
    Step "6. R1 に日本語で話させる（日本語の音声を流すことはスタッフに一声かけてから）"
    if ($Yes -or (Read-Host "話させますか？ (y/N)") -match '^[yY]') {
        Push-Location $RepoRoot
        try { pixi run chat -i $NewName --say $SayText } finally { Pop-Location }
        Say ""
        Say "次: pixi run chat -i $NewName --echo  （聞き取りの確認）→  pixi run chat -i $NewName  （本番）" "Cyan"
    }
}
Say ""
Say "完了。元に戻すとき: powershell -ExecutionPolicy Bypass -File scripts\setup_lan.ps1 -Restore" "Green"

} catch {
    Say "[エラー] $($_.Exception.Message)" "Red"
    Say ($_.ScriptStackTrace) "DarkGray"
    throw
} finally {
    # 最後の状態をログに残す（困ったときにこのログを送ればよい）
    Write-Host ""
    Write-Host "---- 診断情報 $(Get-Date -Format s) ----"
    if ($ifIndex) {
        Get-NetAdapter -InterfaceIndex $ifIndex -ErrorAction SilentlyContinue | Format-List Name, InterfaceDescription, Status, LinkSpeed, MacAddress | Out-String | Write-Host
        Get-NetIPInterface -InterfaceIndex $ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Format-Table InterfaceAlias, Dhcp, ConnectionState -AutoSize | Out-String | Write-Host
        Get-NetIPAddress -InterfaceIndex $ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue | Format-Table IPAddress, PrefixLength, PrefixOrigin -AutoSize | Out-String | Write-Host
        Get-NetConnectionProfile -InterfaceIndex $ifIndex -ErrorAction SilentlyContinue | Format-Table Name, NetworkCategory -AutoSize | Out-String | Write-Host
    }
    Write-Host "ログ: $LogPath"
    Stop-Transcript | Out-Null
}
