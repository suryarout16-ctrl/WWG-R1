<#
.SYNOPSIS
  Verify graceful R1 launcher shutdown with mocked API and process objects only.
  Run with PowerShell 7 or Windows PowerShell 5.1. The launcher main block never runs.
#>
[CmdletBinding()]
param([string]$LauncherPath = '')
$ErrorActionPreference = 'Stop'
if (-not $LauncherPath) { $LauncherPath = Join-Path (Split-Path -Parent $PSScriptRoot) 'local_tools/start_r1.ps1' }
$parseErrors = $null
$parseTokens = $null
$launcherAst = [Management.Automation.Language.Parser]::ParseFile($LauncherPath, [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors) { throw 'Launcher syntax could not be parsed' }
foreach ($name in @('Get-PanelPort', 'Assert-Physical', 'Get-PanelProcessHandles', 'Wait-PanelExit', 'Stop-IdlePanel', 'Test-TargetPanel')) {
    $found = @($launcherAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name
    }, $true))
    if ($found.Count -ne 1) { throw "Expected one launcher function: $name" }
    . ([scriptblock]::Create($found[0].Extent.Text))
}
$script:OriginalWait = (Get-Command Wait-PanelExit).ScriptBlock
$script:OriginalHandles = (Get-Command Get-PanelProcessHandles).ScriptBlock
$script:Checks = 0
$script:Cases = 0
$PcIp = '192.168.123.164'
$proc = [pscustomobject]@{ ProcessId = 424242; CommandLine = 'python web_ui.py --port 8770' }

function Say { param($Message, $Color) }
function Get-PanelState {
    param([int]$PanelPort)
    if ($PanelPort -ne 8770) { throw 'Unexpected panel port' }
    return $script:State
}
function Get-PanelProcessHandles {
    param($Process)
    $script:Events.Add('capture')
    if ($script:FailCapture) { throw 'Mock process capture failure' }
    return @(& $script:OriginalHandles $Process)
}
function Wait-PanelExit {
    param($Handles)
    $script:Events.Add('wait')
    if ($script:FailWait) { throw 'Mock process exit timeout' }
    Assert (@($Handles).Count -eq 3) 'Shutdown should wait for parent and recursive children'
}
function Invoke-RestMethod {
    [CmdletBinding()]
    param($Uri, $Method, $ContentType, $Body, $TimeoutSec)
    $script:Events.Add('post')
    $script:Posts.Add([pscustomobject]@{ Uri = "$Uri"; Method = "$Method"; Body = $Body; ContentType = $ContentType; Timeout = $TimeoutSec })
    Assert ($script:OpenedHandles.Contains(424242)) 'Hold the parent OS process handle before requesting shutdown'
    if ($script:FailPost) { throw 'Mock transport failure' }
    if ($script:NullResponse) { return $null }
    return [pscustomobject]@{ ok = $script:PostOk; message = 'mock shutdown result' }
}
function taskkill.exe { $script:Forced++; throw 'A force-kill fallback is forbidden' }
function Stop-Process { $script:Forced++; throw 'A force-stop fallback is forbidden' }
function Start-Process { throw 'The tests must never start processes' }
function Get-Process {
    [CmdletBinding()]
    param([int]$Id)
    $script:Lookups.Add($Id)
    if ($script:Handles.ContainsKey($Id)) { return $script:Handles[$Id] }
    if ($ErrorActionPreference -eq 'Stop') { throw 'Mock process not found' }
    return $null
}
function Get-CimInstance {
    [CmdletBinding()]
    param($ClassName, $Filter)
    if ($ClassName -ne 'Win32_Process' -or $Filter -notmatch '^ParentProcessId=(\d+)$') { throw 'Unexpected process enumeration' }
    $parentId = [int]$Matches[1]
    $script:Enumerations.Add($parentId)
    if ($script:Tree.ContainsKey($parentId)) {
        foreach ($childId in $script:Tree[$parentId]) { [pscustomobject]@{ ProcessId = $childId } }
    }
}
function New-Handle([int]$ProcessId) {
    $handle = [pscustomobject]@{ Id = $ProcessId; HasExited = $false; WaitResult = $true; ConfirmExit = $true; ExitCode = 0 }
    $handle | Add-Member ScriptMethod WaitForExit {
        param($Milliseconds)
        $script:ProcessWaits.Add([pscustomobject]@{ Id = $this.Id; Milliseconds = $Milliseconds })
        if ($this.WaitResult -and $this.ConfirmExit) { $this.HasExited = $true }
        return $this.WaitResult
    }
    $handle | Add-Member ScriptProperty Handle {
        $script:OpenedHandles.Add($this.Id)
        $script:Events.Add("handle:$($this.Id)")
        return [IntPtr]$this.Id
    }
    return $handle
}
function New-State([string]$Kind = 'local') {
    $isLocal = $Kind -eq 'local'
    return [pscustomobject]@{
        connection_kind = $Kind; local = $isLocal; virtual = $Kind -eq 'virtual'
        process_id = 424242
        shutdown = [pscustomobject]@{ supported = $true }
        dds_address = $(if ($isLocal) { $null } else { $PcIp })
        fsm_ok = -not $isLocal; fsm = $(if ($isLocal) { $null } else { 811 })
        busy = ''; recording = $false
        call = [pscustomobject]@{ active = $false; input_closing = $false }
        training = [pscustomobject]@{ active = $false; phase = 'idle'; awaiting_reply = $false }
        body = [pscustomobject]@{ dry = $isLocal; busy = $false; stopping = $false; hold_active = $false; stop_required = $false }
        posture = [pscustomobject]@{ dry = $isLocal; busy = $false }
    }
}
function Reset-Mocks {
    $script:State = New-State 'robot'
    $script:Events = New-Object 'System.Collections.Generic.List[string]'
    $script:Posts = New-Object 'System.Collections.Generic.List[object]'
    $script:ProcessWaits = New-Object 'System.Collections.Generic.List[object]'
    $script:Lookups = New-Object 'System.Collections.Generic.List[int]'
    $script:Enumerations = New-Object 'System.Collections.Generic.List[int]'
    $script:OpenedHandles = New-Object 'System.Collections.Generic.List[int]'
    $script:Handles = @{ 424242 = (New-Handle 424242); 444444 = (New-Handle 444444); 555555 = (New-Handle 555555) }
    $script:Tree = @{ 424242 = @(444444); 444444 = @(555555); 555555 = @() }
    $script:FailPost = $false
    $script:NullResponse = $false
    $script:PostOk = $true
    $script:FailCapture = $false
    $script:FailWait = $false
    $script:Forced = 0
    $script:Offline = $false
    $script:Stop = $false
    $script:Restart = $false
}
function Assert($Condition, [string]$Message) {
    $script:Checks++
    if (-not $Condition) { throw "$script:CaseName : $Message" }
}
function Assert-Refused {
    $refused = $false
    try { Stop-IdlePanel $proc } catch { $refused = $true }
    Assert $refused 'Expected shutdown to be refused'
    Assert ($script:Events -notcontains 'wait') 'A refused request must not advance to process termination'
    Assert ($script:Forced -eq 0) 'No force-kill fallback is allowed'
}
function Run-Case([string]$Name, [scriptblock]$Test) {
    Reset-Mocks
    $script:CaseName = $Name
    $script:Cases++
    & $Test
    Assert ($script:Forced -eq 0) 'No test path may force terminate a process'
}

foreach ($kind in @('robot', 'local')) {
    Run-Case "Idle $kind uses the graceful API and waits for children" {
        $script:State = New-State $kind
        Stop-IdlePanel $proc
        Assert ((@($script:Events | Where-Object { $_ -notlike 'handle:*' }) -join ',') -eq 'capture,post,wait') 'Capture must precede shutdown, and wait must follow successful acknowledgement'
        Assert ($script:Posts.Count -eq 1) 'Exactly one shutdown request is expected'
        Assert ($script:Posts[0].Uri -eq 'http://127.0.0.1:8770/api/shutdown') 'Only the selected panel shutdown API is called'
        Assert ($script:Posts[0].Method -ieq 'Post') 'Shutdown requires POST'
        Assert ($script:Posts[0].ContentType -eq 'application/json' -and $script:Posts[0].Body -eq '{}') 'Shutdown body must not contain a motion request'
    }
}
Run-Case 'Idle local reply-wait may migrate using only the shutdown API' {
    $script:State = New-State
    $script:State.training = [pscustomobject]@{ active = $true; phase = 'await_reply'; awaiting_reply = $true }
    Stop-IdlePanel $proc
    Assert ($script:Posts.Count -eq 1 -and $script:Posts[0].Uri -like '*/api/shutdown') 'Do not separately submit speech or start training'
}
$changes = [ordered]@{
    'call active' = { param($s) $s.call.active = $true }
    'input closing' = { param($s) $s.call.input_closing = $true }
    'recording' = { param($s) $s.recording = $true }
    'reply active' = { param($s) $s.busy = 'reply' }
    'body busy' = { param($s) $s.body.busy = $true }
    'body stopping' = { param($s) $s.body.stopping = $true }
    'held movement' = { param($s) $s.body.hold_active = $true }
    'stop unconfirmed' = { param($s) $s.body.stop_required = $true }
    'training active' = { param($s) $s.training = [pscustomobject]@{ active = $true; phase = 'explain'; awaiting_reply = $false } }
}
foreach ($change in $changes.GetEnumerator()) {
    Run-Case "Implicit replacement does not interrupt $($change.Key)" {
        & $change.Value $script:State
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Implicit launch must not request cancellation'
    }
    foreach ($mode in @('Stop', 'Restart')) {
        Run-Case "Explicit $mode delegates $($change.Key) to the safe server gate" {
            Set-Variable -Name $mode -Scope Script -Value $true
            & $change.Value $script:State
            Stop-IdlePanel $proc
            Assert ($script:Posts.Count -eq 1) 'The explicit operation must use the safe API'
            Assert ($script:Events[-1] -eq 'wait') 'Only an acknowledged API may advance to exit waiting'
        }
    }
}
foreach ($mode in @('Stop', 'Restart', 'Implicit')) {
    foreach ($fsm in @(701, 702, 816)) {
        Run-Case "$mode refuses physical transition FSM $fsm" {
            if ($mode -ne 'Implicit') { Set-Variable -Name $mode -Scope Script -Value $true }
            $script:State.fsm = $fsm
            Assert-Refused
            Assert ($script:Posts.Count -eq 0) 'No interruption request is allowed during a posture or arm transition'
        }
    }
    Run-Case "$mode refuses active posture work" {
        if ($mode -ne 'Implicit') { Set-Variable -Name $mode -Scope Script -Value $true }
        $script:State.posture.busy = $true
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Do not cancel an active get-up'
    }
}
$invalid = [ordered]@{
    'null state' = { $script:State = $null }
    'virtual state' = { $script:State = New-State 'virtual' }
    'unknown state' = { $script:State = New-State 'unknown'; $script:State.fsm_ok = $false }
    'unreadable physical FSM' = { $script:State.fsm_ok = $false }
    'wrong DDS interface address' = { $script:State.dds_address = '192.0.2.9' }
    'physical body marked dry' = { $script:State.body.dry = $true }
    'physical posture marked dry' = { $script:State.posture.dry = $true }
    'local DDS configured' = { $script:State = New-State; $script:State.dds_address = $PcIp }
    'local body not dry' = { $script:State = New-State; $script:State.body.dry = $false }
    'missing shutdown capability' = { $script:State.PSObject.Properties.Remove('shutdown') }
    'unsupported old panel' = { $script:State.shutdown.supported = $false }
    'different process owns the API' = { $script:State.process_id = 424243 }
    'missing process identity' = { $script:State.PSObject.Properties.Remove('process_id') }
}
foreach ($entry in $invalid.GetEnumerator()) {
    Run-Case "Refuse $($entry.Key) even during explicit stop" {
        $script:Stop = $true
        & $entry.Value
        Assert-Refused
        Assert ($script:Posts.Count -eq 0) 'Unknown or unsupported targets must not receive shutdown requests'
    }
}
foreach ($failure in @('FailPost', 'NullResponse', 'PostOk')) {
    Run-Case "Shutdown API $failure must not lead to exit waiting" {
        $script:Stop = $true
        if ($failure -eq 'PostOk') { $script:PostOk = $false } else { Set-Variable -Name $failure -Scope Script -Value $true }
        Assert-Refused
        Assert ($script:Posts.Count -eq 1) 'One safe request is allowed, with no fallback'
    }
}
Run-Case 'Process capture failure prevents API request' {
    $script:Stop = $true
    $script:FailCapture = $true
    Assert-Refused
    Assert ($script:Posts.Count -eq 0) 'Capture the owned process handles before shutting down'
}
Run-Case 'Cleanup timeout is reported without force termination' {
    $script:Stop = $true
    $script:FailWait = $true
    $failed = $false
    try { Stop-IdlePanel $proc } catch { $failed = $true }
    Assert $failed 'A process exit timeout must be surfaced'
    Assert ($script:Posts.Count -eq 1 -and $script:Events[-1] -eq 'wait') 'Only wait after the safe acknowledgement'
}
Run-Case 'Actual wait function waits for parent and child using a bounded shared deadline' {
    $items = @($script:Handles[424242], $script:Handles[444444])
    & $script:OriginalWait $items
    Assert ($script:ProcessWaits.Count -eq 2) 'Both live process handles must be waited'
    Assert ($script:ProcessWaits[0].Milliseconds -le 20000 -and $script:ProcessWaits[0].Milliseconds -ge 0) 'Parent wait is bounded'
    Assert ($script:ProcessWaits[1].Milliseconds -le $script:ProcessWaits[0].Milliseconds) 'Child must share the original deadline'
    Assert ($items[0].HasExited -and $items[1].HasExited) 'Exit must be confirmed for both handles'
}
foreach ($blockedId in @(424242, 444444)) {
    Run-Case "Wait timeout for process $blockedId is fail closed" {
        $script:Handles[$blockedId].WaitResult = $false
        $failed = $false
        try { & $script:OriginalWait @($script:Handles[424242], $script:Handles[444444]) } catch { $failed = $true }
        Assert $failed 'A non-exiting process must block completion'
        Assert (-not $script:Handles[$blockedId].HasExited) 'The blocked process must not be killed'
    }
}
Run-Case 'An already-exited handle does not need a wait' {
    $script:Handles[424242].HasExited = $true
    & $script:OriginalWait @($script:Handles[424242])
    Assert ($script:ProcessWaits.Count -eq 0) 'Do not wait or kill an already-exited process'
}
Run-Case 'A nonzero parent cleanup exit code cannot report safe completion' {
    $script:Handles[424242].ExitCode = 1
    $failed = $false
    try { & $script:OriginalWait @($script:Handles[424242], $script:Handles[444444]) } catch { $failed = $true }
    Assert $failed 'Parent cleanup failure must be surfaced'
    Assert ($script:ProcessWaits.Count -eq 2) 'Child cleanup must still be awaited before checking the parent result'
}
Run-Case 'A missing parent exit code cannot prove cleanup succeeded' {
    $script:Handles[424242].PSObject.Properties.Remove('ExitCode')
    $failed = $false
    try { & $script:OriginalWait @($script:Handles[424242]) } catch { $failed = $true }
    Assert $failed 'Unknown cleanup result must be refused'
}
Run-Case 'An already-exited parent still requires a successful exit code' {
    $script:Handles[424242].HasExited = $true
    $script:Handles[424242].ExitCode = 7
    $failed = $false
    try { & $script:OriginalWait @($script:Handles[424242]) } catch { $failed = $true }
    Assert $failed 'An early parent failure must not be mistaken for graceful completion'
    Assert ($script:ProcessWaits.Count -eq 0) 'No process wait or force stop is needed for an already-exited parent'
}
Run-Case 'An intentionally terminated child may have a nonzero exit code' {
    $script:Handles[444444].ExitCode = 15
    & $script:OriginalWait @($script:Handles[424242], $script:Handles[444444])
    Assert ($script:ProcessWaits.Count -eq 2) 'Both parent and owned child still require exit confirmation'
    Assert ($script:Handles[424242].HasExited -and $script:Handles[444444].HasExited) 'Normal parent cleanup may terminate its owned worker'
}
Run-Case 'An empty process list cannot prove cleanup' {
    $failed = $false
    try { & $script:OriginalWait @() } catch { $failed = $true }
    Assert $failed 'At least the panel handle must be known'
}
Run-Case 'The process collector includes recursively-owned children exactly once' {
    $script:Tree[555555] = @(424242)
    $items = @(& $script:OriginalHandles $proc)
    Assert (($items.Id -join ',') -eq '424242,444444,555555') 'Only parent and recursive children should be collected'
    Assert (($script:Enumerations -join ',') -eq '424242,444444,555555') 'Process cycles must not cause duplicate traversal'
    Assert ($script:OpenedHandles.Contains(424242)) 'Capture must open the parent process handle while the panel is alive'
}
Run-Case 'An already-exited panel cannot be captured for shutdown' {
    $script:Handles[424242].HasExited = $true
    $failed = $false
    try { & $script:OriginalHandles $proc } catch { $failed = $true }
    Assert $failed 'Do not act on a stale process identity'
}
Run-Case 'Physical readiness does not silently clear the startup motion lock' {
    $script:State.body.stop_required = $true
    Assert (Test-TargetPanel $script:State) 'Panel readiness and motion permission are separate'
    Assert $script:State.body.stop_required 'A read of readiness must not acknowledge stopping'
}
Run-Case 'Offline target is local and never virtual' {
    $script:Offline = $true
    Assert (Test-TargetPanel (New-State)) 'Local mode should be a valid offline target'
    Assert (-not (Test-TargetPanel (New-State 'robot'))) 'Offline mode does not connect to a robot'
    Assert (-not (Test-TargetPanel (New-State 'virtual'))) 'Offline mode does not start a virtual robot'
}
Run-Case 'Stop DryRun branches before enumerating or stopping panels' {
    $stopBranches = @($launcherAst.FindAll({
        param($node)
        $node -is [Management.Automation.Language.IfStatementAst] -and $node.Clauses[0].Item1.Extent.Text.Trim() -eq '$Stop'
    }, $true))
    Assert ($stopBranches.Count -eq 1) 'Expected one main Stop branch'
    $first = $stopBranches[0].Clauses[0].Item2.Statements[0]
    Assert ($first -is [Management.Automation.Language.IfStatementAst]) 'DryRun must be the first action within Stop'
    Assert ($first.Clauses[0].Item1.Extent.Text.Trim() -eq '$DryRun') 'DryRun must intercept before any process enumeration'
}
Write-Host "$script:Cases launcher cases / $script:Checks assertions passed; mocked API and process objects only"
