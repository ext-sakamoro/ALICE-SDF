# Sanitizer for what the self-hosted jobs print to their public job log.
# Author: Moroya Sakamoto
#
# The editor / UAT / Unity logs name the machine: drive paths, the account's profile
# directory, the install roots, the Unity licence client's channel (named after the account),
# the serial number and the machine id. Everything a self-hosted job echoes goes through
# Protect-CiLine first:
#   - the Unity licence fields (serial, machine id, licence client channel) become <redacted>
#   - e-mail addresses become <email>
#   - a drive path (C:\..., E:/..., E:\\... in JSON), a UNC path or a Unix home path becomes
#     <path>, keeping only an ASCII file name with an extension at its end (buildplugin.log)
#   - the account and machine names from the environment become <user> / <host>
#   - a line that still holds a drive / UNC / home path afterwards is withheld whole
# The full logs stay on the runner (scripts/ci-log-retain.ps1).

Set-StrictMode -Version Latest

$script:DrivePath = '(?<![A-Za-z0-9])[A-Za-z]:[\\/](?:[^\\/\r\n"''<>|]*[\\/])*[^\s\\/"''<>|]*'
$script:UncPath = '(?<![A-Za-z0-9:\\])\\\\[^\\\s"''<>|]+\\[^\s"''<>|]*'
$script:HomePath = '(?<![A-Za-z0-9.])/(?:Users|home)/[^\s"''<>|]*'
# wider than the patterns above on purpose (any X:\ or \\x, wherever it stands): what they
# leave behind is withheld, not printed
$script:Residue = '[A-Za-z]:\\|(?<![A-Za-z0-9])[A-Za-z]:/|\\\\[^\\\s]|/(?:Users|home)/'

function Get-CiIdentity {
    # the account and machine names the logs can carry, longest first (a name that contains
    # another is replaced before it)
    $names = @()
    foreach ($v in @($env:USERNAME, $env:USER, $env:COMPUTERNAME, $env:HOSTNAME, $env:USERDOMAIN)) {
        if ($v -and $v.Length -ge 3) { $names += $v }
    }
    foreach ($p in @($env:USERPROFILE, $env:HOME)) {
        if ($p) {
            $leaf = Split-Path -Leaf $p
            if ($leaf -and $leaf.Length -ge 3) { $names += $leaf }
        }
    }
    return @($names | Sort-Object -Unique | Sort-Object Length -Descending)
}

function Protect-PathMatch([System.Text.RegularExpressions.Match] $m) {
    # trailing punctuation belongs to the sentence, not the path: "(see E:\x\a.log)."
    $v = $m.Value
    $tail = ''
    if ($v -match '^(.*?)([)\]},;:.]+)$') { $v = $Matches[1]; $tail = $Matches[2] }
    $leaf = ($v -split '[\\/]')[-1]
    if ($leaf -match '^[A-Za-z0-9_.\-]+\.[A-Za-z0-9]{1,8}$') { return "<path>\$leaf$tail" }
    return "<path>$tail"
}

function Protect-CiLine {
    [CmdletBinding()]
    param([Parameter(ValueFromPipeline = $true)] [AllowEmptyString()] [AllowNull()] $Line)
    begin { $identity = Get-CiIdentity }
    process {
        if ($null -eq $Line) { return '' }
        $s = [string] $Line
        $s = [regex]::Replace($s, '(?i)(Serial number assigned to:\s*).*', '$1<redacted>')
        $s = [regex]::Replace($s, '(?i)(Machine Id:\s*)\S+', '$1<redacted>')
        $s = [regex]::Replace($s, '(?i)LicenseClient-[^"\s]+', 'LicenseClient-<redacted>')
        $s = [regex]::Replace($s, '[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}', '<email>')
        $s = [regex]::Replace($s, $script:DrivePath, { param($m) Protect-PathMatch $m })
        $s = [regex]::Replace($s, $script:UncPath, { param($m) Protect-PathMatch $m })
        $s = [regex]::Replace($s, $script:HomePath, { param($m) Protect-PathMatch $m })
        # names last: a name inside a path went with the path, a bare one is replaced here
        foreach ($n in $identity) {
            $tag = if ($n -eq $env:COMPUTERNAME -or $n -eq $env:HOSTNAME) { '<host>' } else { '<user>' }
            $s = $s.Replace($n, $tag)
        }
        if ($s -match $script:Residue) { return '<line withheld: a path the sanitizer could not strip>' }
        return $s
    }
}

Export-ModuleMember -Function Protect-CiLine
