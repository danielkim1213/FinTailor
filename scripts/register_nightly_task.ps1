# Registers a Windows Task Scheduler job for the nightly loop at the given LOCAL time.
# Pick ~1.5h after the US close (16:00 ET):
#   Toronto/New York (ET):  17:30 Mon-Fri   (default)
#   Vancouver (PT):         14:30 Mon-Fri
#   Korea (KST):            07:30 Tue-Sat   (06:30 during US daylight saving time)  -> -Korea
# Usage:  .\scripts\register_nightly_task.ps1                 (17:30 Mon-Fri)
#         .\scripts\register_nightly_task.ps1 -Time "07:30" -Korea
param([string]$Time = "22:00", [switch]$Korea, [string]$Name = "FinTailorNightly")
$script = Join-Path $PSScriptRoot "nightly.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$script`""
$days = if ($Korea) { @("Tuesday", "Wednesday", "Thursday", "Friday", "Saturday") } else { @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") }
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $Time
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 3)
Register-ScheduledTask -TaskName $Name -Action $action -Trigger $trigger -Settings $settings -Force
Write-Host "Registered $Name at $Time (local) on $($days -join ','). Remove: Unregister-ScheduledTask -TaskName $Name"
