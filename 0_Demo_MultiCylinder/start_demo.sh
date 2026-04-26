
# whenever want to work on or show off the demo, 
# only need to open one single terminal and run these two commands:
# conda activate ModularDT
# ./start_demo.sh

# A safety check to remind you about your Conda environment
if [[ "$CONDA_DEFAULT_ENV" != "ModularDT" ]]; then
    echo "⚠️  Warning: Conda environment 'ModularDT' does not appear to be active."
    echo "For the backend to work, you may need to run 'conda activate ModularDT' first."
    echo ""
fi

# 1. Define the cleanup function (The "Trap")
# This ensures that when you press Ctrl+C, it kills both the frontend and backend.
cleanup() {
    echo -e "\n🛑 Shutting down the demo..."
    # kill all background jobs started by this script
    kill $(jobs -p) 2>/dev/null
    wait $(jobs -p) 2>/dev/null
    echo "✅ Backend and Frontend stopped cleanly."
    exit 0
}

# Bind the cleanup function to Ctrl+C (SIGINT)
trap cleanup SIGINT

echo "🚀 Starting ModularDT Web Demo in a single terminal..."

# Navigate to your demo folder
cd web_demo || exit

# 2. Start Backend in the background (notice the & at the end)
echo "▶️  Starting Backend..."
./run_backend.sh &

# 3. Start Frontend in the background (notice the & at the end)
echo "▶️  Starting Frontend..."
./run_frontend.sh &

# 4. Wait a few seconds for Uvicorn and Vite to boot up
echo "⏳ Waiting 4 seconds for servers to initialize..."
sleep 4

# 5. Open the browser automatically
echo "🌐 Opening dashboard..."
if command -v xdg-open &> /dev/null; then
    xdg-open http://127.0.0.1:5173
elif command -v open &> /dev/null; then
    open http://127.0.0.1:5173 # For macOS
elif command -v start &> /dev/null; then
    start http://127.0.0.1:5173 # For Windows/WSL
else
    echo "👉 Please manually open: http://127.0.0.1:5173"
fi

echo "✨ Demo is live! Both servers are running in the background."
echo "👇 Press Ctrl+C in this terminal at any time to stop everything."

# 6. Wait indefinitely so the script doesn't exit (which would kill the servers)
wait