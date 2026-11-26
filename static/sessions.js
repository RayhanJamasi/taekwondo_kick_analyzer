// Asking before deleting a saved kick
for (const button of document.querySelectorAll(".delete-recording")) {
    button.addEventListener("click", async () => {
        if (!window.confirm("Delete this kick and its saved video?")) return;

        const response = await fetch(button.dataset.url, { method: "DELETE" });
        if (!response.ok) {
            window.alert("Could not delete this kick.");
            return;
        }

        window.location.reload();
    });
}
