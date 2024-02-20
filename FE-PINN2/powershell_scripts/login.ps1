$username = "anthony"
$serverAddress = Get-Content -Path ".\ip.txt"

ssh ${username}@$serverAddress