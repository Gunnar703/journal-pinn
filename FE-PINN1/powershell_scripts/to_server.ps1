# Define variables
$sourcePath = "./*"
$destinationPath = "~/Projects/journal-pinn"
$serverAddress = Get-Content -Path ".\ip.txt"
$username = "anthony"

# Construct the full command
scp -r ""$sourcePath"" $username@${serverAddress}:""$destinationPath""