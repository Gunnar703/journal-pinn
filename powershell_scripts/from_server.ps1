# Define variables
$sourcePath = "outputs"
$destinationPath = "outputs_ssh"
$serverAddress = Get-Content -Path ".\ip.txt"
$username = "anthony"

# Construct the full command
scp -r $username@${serverAddress}:""$sourcePath"" $destinationPath