# Cleanup tree_runner + experiments artifacts
$repoRoot = Get-Location
$worktreesDir = Join-Path $repoRoot "agentic_experimentation/worktrees"
$experimentsDir = Join-Path $repoRoot "agentic_experimentation/experiments"

$targets = @($worktreesDir, $experimentsDir) | Where-Object { Test-Path $_ }

Write-Host "About to delete ALL contents of:"
$targets | ForEach-Object { Write-Host " - $_" }

$confirm = Read-Host "Type YES to proceed"
if ($confirm -ne "YES") {
  Write-Host "Aborted."
  return
}

foreach ($t in $targets) {
  Get-ChildItem -Path $t -Force | Remove-Item -Recurse -Force
}

# Delete tree branches
$branches = git -C $repoRoot branch --list "tree/*"
foreach ($b in $branches) {
  $name = $b.Trim()
  if ($name) { git -C $repoRoot branch -D $name }
}

# Prune stale git worktree metadata
git -C $repoRoot worktree prune

Write-Host "Cleanup complete."
