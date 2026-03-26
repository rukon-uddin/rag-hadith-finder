
Copy

#!/bin/bash
 
# Start server and SSH tunnel together
# Kills both on exit (Ctrl+C)
 
cleanup() {
    echo ""
    echo "Shutting down..."
    kill "$SERVER_PID" "$SSH_PID" 2>/dev/null
    wait "$SERVER_PID" "$SSH_PID" 2>/dev/null
    echo "Done."
    exit 0
}
 
trap cleanup SIGINT SIGTERM
 
# Start Python server in background
echo "Starting Python server..."
python server.py &
SERVER_PID=$!
 
# Start SSH tunnel in background
echo "Starting SSH tunnel..."
ssh -R 8082:localhost:5123 root@172.245.107.85 -N \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 &
SSH_PID=$!
 
echo "Both services running. Press Ctrl+C to stop."
echo "  Python server PID : $SERVER_PID"
echo "  SSH tunnel PID    : $SSH_PID"
 
# Wait for either process to exit
wait -n "$SERVER_PID" "$SSH_PID"
EXIT_CODE=$?
 
echo "One of the processes exited (code: $EXIT_CODE). Stopping all..."
cleanup