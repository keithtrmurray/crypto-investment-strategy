// Main JavaScript file for Crypto Investment Strategy
console.log("Crypto Investment Strategy loaded");

// Function to format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,');
}

// Function to update market data
function updateMarketData() {
    console.log("Updating market data...");
    // This would normally fetch data from the server
    // For now, we'll just log that it was called
}

// Initialize any interactive elements when the DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM fully loaded");
    
    // Set up any event listeners or initialize UI components
    const refreshButtons = document.querySelectorAll('.refresh-data');
    if (refreshButtons) {
        refreshButtons.forEach(button => {
            button.addEventListener('click', function() {
                updateMarketData();
            });
        });
    }
    
    // Auto-refresh market data every 5 minutes
    setInterval(updateMarketData, 300000);
});
