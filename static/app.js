// Getting the page elements so the code can update them
const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const statusMessage = document.getElementById("status-message");
const cameraFeed = document.getElementById("camera-feed");
const cameraPlaceholder = document.getElementById("camera-placeholder");
let wasRunning = false;

// Putting the latest kick results on the page
function showMetrics(metrics) {
    if (!metrics) return;

    // Putting the values from Python into the matching spots on the page
    document.getElementById("metric-side").textContent = metrics.side;
    document.getElementById("metric-knee").textContent = `${metrics.max_knee_angle.toFixed(2)}°`;
    document.getElementById("metric-height").textContent = metrics.max_kick_height.toFixed(3);
    document.getElementById("metric-duration").textContent =
        metrics.duration === null ? "n/a" : `${metrics.duration.toFixed(2)}s`;
    document.getElementById("feedback-height").textContent = metrics.feedback.height;
    document.getElementById("feedback-knee").textContent = metrics.feedback.knee;
    document.getElementById("feedback-speed").textContent = metrics.feedback.speed;
}

// Updating the saved session table
async function refreshHistory() {
    // Asking Flask for the saved sessions; await means wait for its reply
    const response = await fetch("/history");
    const sessions = await response.json();
    const rows = document.getElementById("history-rows");
    rows.replaceChildren();

    if (sessions.length === 0) {
        const row = rows.insertRow();
        const cell = row.insertCell();
        cell.colSpan = 5;
        cell.textContent = "No sessions yet.";
        return;
    }

    for (const session of sessions) {
        // Making a table row for each saved session
        const row = rows.insertRow();
        for (const value of [
            session.ended_at,
            session.kick_count,
            session.right_kicks,
            session.left_kicks,
        ]) {
            row.insertCell().textContent = value;
        }
        const recordings = row.insertCell();
        const link = document.createElement("a");
        link.className = "btn btn-sm btn-outline-primary";
        link.href = `/sessions?date=${encodeURIComponent(session.ended_at.slice(0, 10))}`;
        link.textContent = "View";
        recordings.appendChild(link);
    }
}

// Checking if the camera is running and updating the page
async function refreshStatus() {
    // Getting the latest camera status from Flask once a second
    const response = await fetch("/status");
    const status = await response.json();
    startButton.disabled = status.running;
    stopButton.disabled = !status.running;
    document.getElementById("kick-count").textContent = status.kick_count || 0;
    showMetrics(status.latest_metrics);

    statusMessage.className = "badge align-self-center";
    if (status.running) {
        statusMessage.classList.add("text-bg-success");
        statusMessage.textContent = "Analyzing";
    } else if (status.error) {
        statusMessage.classList.add("text-bg-danger");
        statusMessage.textContent = status.error;
    } else {
        statusMessage.classList.add("text-bg-secondary");
        statusMessage.textContent = "Ready";
    }

    if (wasRunning && !status.running) {
        // Refresh the table after a camera session has just finished
        await refreshHistory();
    }
    wasRunning = status.running;
}

// Starting the camera when Start is clicked
startButton.addEventListener("click", async () => {
    // This function runs when the user clicks Start
    statusMessage.textContent = "Starting camera...";
    const response = await fetch("/start", { method: "POST" });
    const result = await response.json();
    if (!response.ok) {
        statusMessage.textContent = result.error;
        return;
    }

    cameraFeed.src = `/video_feed?session_id=${result.session_id}`;
    cameraFeed.hidden = false;
    cameraPlaceholder.hidden = true;
    wasRunning = true;
    await refreshStatus();
});

// Stopping the camera when Stop is clicked
stopButton.addEventListener("click", async () => {
    // This function runs when the user clicks Stop
    statusMessage.textContent = "Stopping...";
    const response = await fetch("/stop", { method: "POST" });
    const result = await response.json();
    if (!response.ok && response.status !== 202) {
        statusMessage.textContent = result.error;
        return;
    }
    await refreshStatus();
});

cameraFeed.addEventListener("error", () => {
    // Show the placeholder if the browser can't load the feed
    cameraFeed.hidden = true;
    cameraPlaceholder.hidden = false;
});

// Check for status updates every 1000 milliseconds (one second)
setInterval(refreshStatus, 1000);
