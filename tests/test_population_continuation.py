"""Execute the actual PowerShell wrapper offline with fake controller responses."""
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which('powershell'), 'Windows PowerShell required')
class ContinuationTests(unittest.TestCase):
    def run_case(self, scenario):
        wrapper = Path('run_population_continuous.ps1').resolve()
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'check.ps1'
            script.write_text(f"""
$ErrorActionPreference='Stop'
. '{str(wrapper).replace("'", "''")}'
$script:statuses=0; $script:sessions=0
$script:now=[DateTimeOffset]::Parse('2026-10-18T00:00:00Z')
$scenario='{scenario}'
$runner={{ param($a)
 if ($a[0] -eq 'session') {{
  $script:sessions++
  if (($a -join ' ') -notmatch '--max-active 4') {{throw 'Wrong active limit'}}
  $script:now=$script:now.AddMinutes(15)
  $code=0; $output='{{"actions":[]}}'
  if ($scenario -eq 'error') {{$code=1}}
  if ($scenario -eq 'intervention') {{$output='{{"action":"NEEDS_REVIEW"}}'}}
  return [pscustomobject]@{{ExitCode=$code;Lines=@($output)}}
 }}
 $script:statuses++
 $done=16; $pending=2084; $failed=0
 if ($scenario -eq 'failed') {{$failed=1}}
 if ($scenario -eq 'complete') {{$done=2100;$pending=0}}
 if ($scenario -eq 'cap' -and $script:sessions -gt 0) {{$pending=2080}}
 if ($scenario -eq 'restart' -and $script:sessions -eq 1) {{$done=20;$pending=2080}}
 if ($scenario -eq 'restart' -and $script:sessions -eq 2) {{$done=2100;$pending=0}}
 $s=@{{production_version='phase3_ca175_tile1_v1';expected_exports=2100;completed=$done;pending=$pending;failed=$failed;running_status=0}}
 return [pscustomobject]@{{ExitCode=0;Lines=@(($s|ConvertTo-Json -Compress))}}
}}
$caught=$false
try {{Invoke-ProductionContinuation -Hours 1 -NewLimit $(if($scenario -eq 'cap'){{4}}else{{2088}}) -SliceMinutes 15 -StallMinutes $(if($scenario -eq 'runtime'){{120}}else{{30}}) -Runner $runner -Clock {{$script:now}} -LogDirectory '{directory}'}}
catch {{$caught=$true;Write-Host $_}}
$expectedError=$scenario -in @('failed','error','intervention','stall')
if ($caught -ne $expectedError) {{throw 'Wrong error behavior'}}
$expected=@{{complete=0;failed=0;cap=1;restart=2;error=1;intervention=1;stall=2;runtime=4}}
if ($script:sessions -ne $expected[$scenario]) {{throw "Wrong session count $script:sessions"}}
if (Test-Path '{directory}/phase3_ca175_tile1_v1_continuation.lock') {{throw 'Lock not cleaned'}}
""")
            result=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(script)],capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_complete_and_failed_do_not_launch(self):
        for case in ('complete','failed'): self.run_case(case)

    def test_sessions_continue_without_overlap_and_total_cap_stops(self):
        for case in ('restart','cap'): self.run_case(case)

    def test_error_intervention_and_stall_stop(self):
        for case in ('error','intervention','stall'): self.run_case(case)

    def test_total_runtime_stops_across_restarted_sessions(self):
        self.run_case('runtime')
