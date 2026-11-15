<#
.SYNOPSIS
  Test local-to-robot launcher switching without running its main block.
  PowerShell 5.1 and 7: pwsh -NoProfile -File tests/test_r1_launcher.ps1
  API requests, process termination, and messages are replaced with mocks.
#>
[CmdletBinding()]
param(
    [string]$LauncherPath = ''
)

$ErrorActionPreference = 'Stop'
if (-not $LauncherPath) {
    $LauncherPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'local_tools/start_r1.ps1'
}
$parseErrors = $null
$parseTokens = $null
$launcherAst = [Management.Automation.Language.Parser]::ParseFile(
    $LauncherPath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors) { throw 'Launcher syntax could not be parsed' }

# Import declarations only. The launcher main block, LAN checks and startup never run.
foreach ($name in @('Get-PanelPort', 'Assert-Physical', 'Stop-IdlePanel', 'Test-TargetPanel')) {
    $found = @($launcherAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $true))
    if ($found.Count -ne 1) { throw "Expected one launcher function: $name" }
    . ([scriptblock]::Create($found[0].Extent.Text))
}

$script:Checks = 0
$script:Cases = 0
$PcIp = '192.168.123.164'
$Offline = $false
$Stop = $false
$Restart = $false
$pargs = @()
$proc = [pscustomobject]@{ ProcessId = 424242; CommandLine = 'python web_ui.py --port 8770' }

function Say { param($Message, $Color) }
function Get-PanelState {
    param([int]$PanelPort)
    if ($PanelPort -ne 8770) { throw 'Unexpected panel port' }
    if ($script:States.Count) { $script:LastState = $script:States.Dequeue() }
    return $script:LastState
}
function Invoke-RestMethod {
    [CmdletBinding()]
    param($Uri, $Method, $ContentType, $Body, $TimeoutSec)
    $script:Posts.Add([pscustomobject]@{ Uri = "$Uri"; Method = "$Method"; Body = $Body })
    if ($script:FailPost) { throw 'Mock transport failure' }
    return [pscustomobject]@{ ok = $script:PostOk; message = 'mock' }
}
function taskkill.exe {
    $script:Kills.Add(@($args))
    $global:LASTEXITCODE = 0
}
function New-State([string]$Kind = 'local') {
    $isLocal = $Kind -eq 'local'
    return [pscustomobject]@{
        connection_kind = $Kind; local = $isLocal; virtual = $Kind -eq 'virtual'
        dds_address = $(if ($isLocal) { $null } else { $PcIp })
        fsm_ok = -not $isLocal; busy = ''; recording = $false
        call = [pscustomobject]@{ active = $false; input_closing = $false }
        training = [pscustomobject]@{ active = $false; phase = 'idle'; awaiting_reply = $false }
        body = [pscustomobject]@{
            dry = $isLocal; busy = $false; stopping = $false; hold_active = $false; stop_required = $false
        }
        posture = [pscustomobject]@{ dry = $isLocal; busy = $false }
    }
}
function New-WaitingLocal {
    $state = New-State
    $state.training.active = $true
    $state.training.phase = 'await_reply'
    $state.training.awaiting_reply = $true
    return $state
}
function Reset-Mocks {
    $script:States = New-Object 'System.Collections.Generic.Queue[object]'
    $script:Posts = New-Object 'System.Collections.Generic.List[object]'
    $script:Kills = New-Object 'System.Collections.Generic.List[object]'
    $script:LastState = $null
    $script:FailPost = $false
    $script:PostOk = $true
    $script:Offline = $false
    $script:Stop = $false
    $script:Restart = $false
    $script:pargs = @()
}
function Assert($Condition, [string]$Message) {
    $script:Checks++
    if (-not $Condition) { throw "$script:CaseName : $Message" }
}
function Assert-Refused {
    $refused = $false
    try { Stop-IdlePanel $proc } catch { $refused = $true }
    Assert $refused 'Expected the panel shutdown to be refused'
    Assert ($script:Kills.Count -eq 0) 'A refused shutdown must not terminate a process'
}
function Run-Case([string]$Name, [scriptblock]$Test) {
    Reset-Mocks
    $script:CaseName = $Name
    $script:Cases++
    & $Test
}

Run-Case 'Idle physical panel retains the existing shutdown path' {
    $script:States.Enqueue((New-State 'robot'))
    Stop-IdlePanel $proc
    Assert ($script:Posts.Count -eq 0) 'Physical shutdown must not send cancellation requests'
    Assert ($script:Kills.Count -eq 1) 'Idle physical process should be terminated once'
    Assert (($script:Kills[0] -join ' ') -eq '/PID 424242 /T /F') 'Only the selected PID should be targeted'
}
Run-Case 'Idle local panel needs no training request' {
    $script:States.Enqueue((New-State))
    Stop-IdlePanel $proc
    Assert ($script:Posts.Count -eq 0) 'Idle local state should not need a request'
    Assert ($script:Kills.Count -eq 1) 'Idle local process should be terminated once'
}
Run-Case 'Waiting local training is ended before migration' {
    $script:States.Enqueue((New-WaitingLocal))
    $script:States.Enqueue((New-State))
    Stop-IdlePanel $proc
    Assert ($script:Posts.Count -eq 1) 'Expected one training-stop request'
    Assert ($script:Posts[0].Uri -eq 'http://127.0.0.1:8770/api/training/stop') 'Only local training should be ended'
    Assert ($script:Posts[0].Method -ieq 'Post') 'Training stop must use POST'
    Assert ($script:Kills.Count -eq 1) 'Verified idle local process should be terminated'
}

foreach ($kind in @('robot', 'virtual')) {
    Run-Case "Waiting $kind training must not be cancelled automatically" {
        $state = New-State $kind
        $state.training.active = $true
        $state.training.phase = 'await_reply'
        $state.training.awaiting_reply = $true
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Must not send a request to a physical or virtual panel'
    }
}
foreach ($mode in @('Offline', 'Stop')) {
    Run-Case "$mode does not implicitly end local training" {
        Set-Variable -Name $mode -Scope Script -Value $true
        $script:States.Enqueue((New-WaitingLocal))
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Automatic completion is limited to physical migration'
    }
}

$blockedChanges = [ordered]@{
    'call active' = { param($s) $s.call.active = $true }
    'input closing' = { param($s) $s.call.input_closing = $true }
    'manual recording' = { param($s) $s.recording = $true }
    'reply in progress' = { param($s) $s.busy = 'reply' }
    'body busy' = { param($s) $s.body.busy = $true }
    'stop verification in progress' = { param($s) $s.body.stopping = $true }
    'held movement' = { param($s) $s.body.hold_active = $true }
    'stop required' = { param($s) $s.body.stop_required = $true }
    'posture busy' = { param($s) $s.posture.busy = $true }
    'body is not a preview' = { param($s) $s.body.dry = $false }
    'posture is not a preview' = { param($s) $s.posture.dry = $false }
    'DDS configured' = { param($s) $s.dds_address = $PcIp }
    'not waiting for a reply' = { param($s) $s.training.awaiting_reply = $false }
    'training still running' = { param($s) $s.training.phase = 'explain' }
}
foreach ($entry in $blockedChanges.GetEnumerator()) {
    Run-Case "Local waiting state with $($entry.Key) is refused" {
        $state = New-WaitingLocal
        & $entry.Value $state
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Actual work or an unverified preview must not be cancelled'
    }
}

Run-Case 'Training-stop transport failure prevents termination' {
    $script:FailPost = $true
    $script:States.Enqueue((New-WaitingLocal))
    Assert-Refused
    Assert ($script:Posts.Count -eq 1) 'Expected a single failed cancellation attempt'
}
Run-Case 'Training-stop rejection prevents termination' {
    $script:PostOk = $false
    $script:States.Enqueue((New-WaitingLocal))
    Assert-Refused
    Assert ($script:Posts.Count -eq 1) 'Expected one rejected cancellation request'
}

foreach ($changedKind in @('robot', 'virtual', 'missing', 'still waiting', 'new call', 'not dry', 'DDS configured')) {
    Run-Case "Revalidation rejects $changedKind after cancellation" {
        $script:States.Enqueue((New-WaitingLocal))
        $after = New-State
        switch ($changedKind) {
            'robot' { $after = New-State 'robot' }
            'virtual' { $after = New-State 'virtual' }
            'missing' { $after = $null }
            'still waiting' { $after = New-WaitingLocal }
            'new call' { $after.call.active = $true }
            'not dry' { $after.body.dry = $false }
            'DDS configured' { $after.dds_address = $PcIp }
        }
        $script:States.Enqueue($after)
        Assert-Refused
        Assert ($script:Posts.Count -eq 1) 'Only the initially verified local panel may receive a request'
    }
}

Run-Case 'Physical target is ready while its motion interlock remains latched' {
    $state = New-State 'robot'
    $state.body.stop_required = $true
    Assert (Test-TargetPanel $state) 'Readiness must not require the motion interlock to be cleared'
}
Run-Case 'Normal launch must not discard a physical stop interlock' {
    $state = New-State 'robot'
    $state.body.stop_required = $true
    $script:pargs = @('--require-stop-verification')
    $script:States.Enqueue($state)
    Assert-Refused
    Assert ($script:Posts.Count -eq 0) 'No physical cancellation request is allowed'
}
Run-Case 'Explicit physical restart carries the stop interlock into the replacement' {
    $state = New-State 'robot'
    $state.body.stop_required = $true
    $script:Restart = $true
    $script:pargs = @('--port', '8770', '--require-stop-verification')
    $script:States.Enqueue($state)
    Stop-IdlePanel $proc
    Assert ($script:Kills.Count -eq 1) 'Verified idle physical process should be replaced once'
    Assert ($script:Posts.Count -eq 0) 'Restart must not automatically send zero or cancellation'
    Assert $state.body.stop_required 'The old interlock must not be acknowledged or modified'
}
foreach ($incoming in @(@(), @('--require-stop-verification=false'), @('--require-stop-verification-extra'))) {
    Run-Case "Restart without an exact incoming interlock flag is refused: $($incoming -join ' ')" {
        $state = New-State 'robot'
        $state.body.stop_required = $true
        $script:Restart = $true
        $script:pargs = $incoming
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'No automatic physical stop or acknowledgement is allowed'
    }
}
foreach ($mode in @('Offline', 'Stop')) {
    Run-Case "$mode cannot discard an interlock even with Restart and an incoming flag" {
        $state = New-State 'robot'
        $state.body.stop_required = $true
        $script:Restart = $true
        $script:pargs = @('--require-stop-verification')
        Set-Variable -Name $mode -Scope Script -Value $true
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'No physical stop request is allowed'
    }
}
foreach ($kind in @('local', 'virtual')) {
    Run-Case "$kind interlock is not covered by physical restart recovery" {
        $state = New-State $kind
        $state.body.stop_required = $true
        $script:Restart = $true
        $script:pargs = @('--require-stop-verification')
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Recovery must not send a cancellation request'
    }
}
$physicalBlockedChanges = [ordered]@{
    'call active' = { param($s) $s.call.active = $true }
    'input closing' = { param($s) $s.call.input_closing = $true }
    'manual recording' = { param($s) $s.recording = $true }
    'reply in progress' = { param($s) $s.busy = 'reply' }
    'body busy' = { param($s) $s.body.busy = $true }
    'stop verification in progress' = { param($s) $s.body.stopping = $true }
    'held movement' = { param($s) $s.body.hold_active = $true }
    'posture busy' = { param($s) $s.posture.busy = $true }
    'training active' = { param($s) $s.training.active = $true }
    'FSM read failed' = { param($s) $s.fsm_ok = $false }
    'DDS address mismatch' = { param($s) $s.dds_address = '192.0.2.123' }
    'body marked as preview' = { param($s) $s.body.dry = $true }
    'posture marked as preview' = { param($s) $s.posture.dry = $true }
}
foreach ($entry in $physicalBlockedChanges.GetEnumerator()) {
    Run-Case "Physical restart with $($entry.Key) is refused despite incoming interlock" {
        $state = New-State 'robot'
        $state.body.stop_required = $true
        & $entry.Value $state
        $script:Restart = $true
        $script:pargs = @('--require-stop-verification')
        $script:States.Enqueue($state)
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'No cancellation or zero request should be sent'
    }
}

Write-Host "$script:Cases launcher cases / $script:Checks assertions passed; mocked API and process operations only"
