<#
.SYNOPSIS
  指定した物理EthernetだけにR1用IPv4を追加し、Privateにする管理者用補助。
  自己昇格、アダプター名変更、既存IP/経路/DNSの削除、ロボット操作は行わない。
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][ValidateRange(1, 2147483647)][int]$InterfaceIndex,
    [Parameter(Mandatory = $true)][string]$PcIp,
    [ValidateRange(1, 30)][int]$PrefixLength = 24
)

$ErrorActionPreference = "Stop"
$result = [ordered]@{
    time = (Get-Date).ToString("o")
    interface_index = $InterfaceIndex
    requested_ip = $PcIp
    prefix_length = $PrefixLength
    success = $false
    added_ip = $false
    changed_category = $false
    before = $null
    after = $null
    error = ""
}
$exitCode = 1
$selected = $null

function Get-SelectedEthernet {
    $matches = @(Get-NetAdapter -Physical -IncludeHidden |
        Where-Object { $_.ifIndex -eq $InterfaceIndex })
    if ($matches.Count -ne 1) { throw "指定した物理アダプターを一意に確認できません: $InterfaceIndex" }
    $adapter = $matches[0]
    if ($adapter.MediaType -ne "802.3" -or $adapter.Virtual -eq $true -or
        $adapter.InterfaceDescription -match "Wi-Fi|Wireless|802\.11|Virtual|Hyper-V|VPN|TAP|Tailscale|Bluetooth|Remote NDIS") {
        throw "対象は物理Ethernetに限定します: $($adapter.Name)"
    }
    if ($adapter.Status -ne "Up") { throw "Ethernetが接続されていません: $($adapter.Name) ($($adapter.Status))" }
    return $adapter
}

function Get-SelectedProfile {
    $profiles = @(Get-NetConnectionProfile -InterfaceIndex $InterfaceIndex)
    if ($profiles.Count -ne 1) { throw "対象Ethernetのネットワークプロファイルを一意に確認できません" }
    if ($profiles[0].NetworkCategory -eq "DomainAuthenticated") {
        throw "ドメイン管理のネットワークは変更しません"
    }
    return $profiles[0]
}

function Get-SelectedSnapshot {
    $adapter = Get-SelectedEthernet
    $profile = Get-SelectedProfile
    return [ordered]@{
        name = $adapter.Name
        interface_index = $adapter.ifIndex
        interface_description = $adapter.InterfaceDescription
        category = "$($profile.NetworkCategory)"
        dhcp = "$((Get-NetIPInterface -InterfaceIndex $InterfaceIndex -AddressFamily IPv4).Dhcp)"
        addresses = @(Get-NetIPAddress -AddressFamily IPv4 |
            Where-Object { $_.InterfaceIndex -eq $InterfaceIndex } |
            Select-Object IPAddress, PrefixLength, PrefixOrigin, SuffixOrigin, AddressState)
        routes = @(Get-NetRoute | Where-Object { $_.InterfaceIndex -eq $InterfaceIndex } |
            Select-Object AddressFamily, DestinationPrefix, NextHop, RouteMetric, Protocol)
        dns = @(Get-DnsClientServerAddress -InterfaceIndex $InterfaceIndex |
            Select-Object AddressFamily, ServerAddresses)
    }
}

function Get-RouteKeys($rows) {
    @($rows | ForEach-Object {
        "{0}|{1}|{2}|{3}|{4}" -f $_.AddressFamily, $_.DestinationPrefix, $_.NextHop, $_.RouteMetric, $_.Protocol
    })
}

try {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "この補助スクリプトは管理者権限で呼び出してください"
    }

    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($PcIp, [ref]$parsed) -or
        $parsed.AddressFamily -ne [Net.Sockets.AddressFamily]::InterNetwork -or
        $parsed.ToString() -cne $PcIp) {
        throw "PcIpには通常のIPv4表記を指定してください"
    }
    $bytes = $parsed.GetAddressBytes()
    if ($bytes[0] -eq 0 -or $bytes[0] -eq 127 -or $bytes[0] -ge 224) {
        throw "PcIpにはユニキャストのIPv4を指定してください"
    }
    [uint64]$ipNumber = ([uint64]$bytes[0] * 16777216) + ([uint64]$bytes[1] * 65536) +
        ([uint64]$bytes[2] * 256) + [uint64]$bytes[3]
    [uint64]$hostSize = [math]::Pow(2, (32 - $PrefixLength))
    if (($ipNumber % $hostSize) -eq 0 -or ($ipNumber % $hostSize) -eq ($hostSize - 1)) {
        throw "ネットワークアドレスとブロードキャストアドレスは指定できません"
    }

    $selected = Get-SelectedEthernet
    $result.before = Get-SelectedSnapshot
    $existing = @($result.before.addresses | Where-Object { $_.IPAddress -eq $PcIp })
    if ($existing.Count -gt 1) { throw "対象IPが複数あり、設定を確認できません" }
    if ($existing.Count -eq 1 -and $existing[0].PrefixLength -ne $PrefixLength) {
        throw "既存IPのプレフィックスが /$PrefixLength と異なります。既存IPは変更しません"
    }
    $otherAdapters = @(Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -eq $PcIp -and $_.InterfaceIndex -ne $InterfaceIndex })
    if ($otherAdapters.Count -gt 0) { throw "対象IPが別のアダプターにもあります。自動変更しません" }
    if ($existing.Count -eq 0 -and $result.before.dhcp -ne "Disabled") {
        # New-NetIPAddressはDHCPを無効化し得る。既存のDHCPアドレス/経路の維持を優先する。
        throw "DHCPが有効なためIP追加を中止します。既存のIPと経路は変更しません"
    }

    if ($existing.Count -eq 0) {
        # 直前にも同じ物理NICとリンク状態を確認する。他のIP/経路/DNSへは触れない。
        $null = Get-SelectedEthernet
        New-NetIPAddress -InterfaceIndex $InterfaceIndex -IPAddress $PcIp -PrefixLength $PrefixLength `
            -AddressFamily IPv4 -Type Unicast | Out-Null
        $result.added_ip = $true
    }
    $profile = Get-SelectedProfile
    if ($profile.NetworkCategory -ne "Private") {
        Set-NetConnectionProfile -InterfaceIndex $InterfaceIndex -NetworkCategory Private
        $result.changed_category = $true
    }

    $preferred = $false
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        $currentIp = @(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {
            $_.InterfaceIndex -eq $InterfaceIndex -and $_.IPAddress -eq $PcIp -and $_.PrefixLength -eq $PrefixLength
        })
        if ($currentIp.Count -eq 1 -and $currentIp[0].AddressState -eq "Preferred") {
            $preferred = $true
            break
        }
        if ($currentIp.Count -eq 1 -and $currentIp[0].AddressState -eq "Duplicate") { break }
        Start-Sleep -Milliseconds 250
    }
    $result.after = Get-SelectedSnapshot
    if (-not $preferred -or $result.after.category -ne "Private") {
        throw "対象IPの利用可能状態とPrivate設定を確認できません"
    }
    if ($result.before.name -cne $result.after.name -or $result.before.dhcp -ne $result.after.dhcp) {
        throw "アダプター名またはDHCP設定の変化を検出しました"
    }
    foreach ($address in $result.before.addresses) {
        $kept = @($result.after.addresses | Where-Object {
            $_.IPAddress -eq $address.IPAddress -and $_.PrefixLength -eq $address.PrefixLength
        })
        if ($kept.Count -ne 1) { throw "既存IPv4の維持を確認できません: $($address.IPAddress)" }
    }
    $afterRouteKeys = @(Get-RouteKeys $result.after.routes)
    foreach ($key in @(Get-RouteKeys $result.before.routes)) {
        if ($key -notin $afterRouteKeys) { throw "既存経路の維持を確認できません" }
    }
    $beforeDns = @($result.before.dns | ForEach-Object { "$($_.AddressFamily)|$($_.ServerAddresses -join ',')" } | Sort-Object)
    $afterDns = @($result.after.dns | ForEach-Object { "$($_.AddressFamily)|$($_.ServerAddresses -join ',')" } | Sort-Object)
    if (($beforeDns -join "`n") -cne ($afterDns -join "`n")) { throw "既存DNS設定の維持を確認できません" }

    $result.success = $true
    $exitCode = 0
    Write-Host "LAN設定OK: $($selected.Name) (ifIndex $InterfaceIndex), $PcIp/$PrefixLength, Private"
}
catch {
    $result.error = $_.Exception.Message
    Write-Error $result.error -ErrorAction Continue
    if ($selected -and -not $result.after) {
        try { $result.after = Get-SelectedSnapshot } catch { }
    }
}
finally {
    try {
        $repoRoot = Split-Path -Parent $PSScriptRoot
        $logDir = Join-Path $repoRoot "logs"
        New-Item -ItemType Directory -Force -Path $logDir | Out-Null
        $logFile = Join-Path $logDir ("prepare_r1_lan_{0}_{1}.json" -f (Get-Date -Format "yyyyMMdd_HHmmss"), $PID)
        $result | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $logFile -Encoding UTF8
        Write-Host "LAN設定記録: $logFile"
    }
    catch { Write-Warning "LAN設定の記録を保存できませんでした" }
}
exit $exitCode
