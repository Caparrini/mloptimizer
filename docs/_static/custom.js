document.addEventListener("DOMContentLoaded", function() {
    // Select the toc-item and its ul (if exists)
    const tocItem = document.querySelector('.bd-toc-item ul');

    // Check if the ul doesn't exist
    if (!tocItem) {
        // Hide the entire sidebar if the toc ul is not found
        const sidebar = document.querySelector('.bd-sidebar');
        if (sidebar) {
            sidebar.style.display = 'none';
        }
    }
});
