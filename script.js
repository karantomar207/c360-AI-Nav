// Store search history
let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];

// Update the search history display
function updateHistoryDisplay() {
  const historyList = document.getElementById('searchHistory');
  historyList.innerHTML = '';
  
  if (searchHistory.length === 0) {
    historyList.innerHTML = '<li class="history-item">No recent searches</li>';
    return;
  }
  
  // Display the 5 most recent searches
  searchHistory.slice(0, 5).forEach((query, index) => {
    const historyItem = document.createElement('li');
    historyItem.className = 'history-item';
    historyItem.innerHTML = `
      <div class="history-query">
        <i class="fas fa-history"></i>
        <span>${query}</span>
      </div>
      <div class="history-actions">
        <button onclick="setKeyword('${query}')">
          <i class="fas fa-search"></i> Search Again
        </button>
        <button onclick="removeFromHistory(${index})">
          <i class="fas fa-times"></i> Remove
        </button>
      </div>
    `;
    historyList.appendChild(historyItem);
  });
}

// Set a keyword to the search input
function setKeyword(keyword) {
  document.getElementById('userInput').value = keyword;
  fetchData();
}

// Remove an item from search history
function removeFromHistory(index) {
  searchHistory.splice(index, 1);
  localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  updateHistoryDisplay();
}

// Add search query to history
function addToHistory(query) {
  // Remove the query if it already exists to avoid duplicates
  searchHistory = searchHistory.filter(item => item !== query);
  
  // Add the new query to the beginning
  searchHistory.unshift(query);
  
  // Keep only the 10 most recent searches
  if (searchHistory.length > 10) {
    searchHistory.pop();
  }
  
  // Update localStorage
  localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  
  // Update the display
  updateHistoryDisplay();
}

// Clear results and reset the page
function clearResults() {
  const userInput = document.getElementById("userInput");
  const errorMsg = document.getElementById("errorMsg");
  const resultsSection = document.getElementById("resultsSection");
  const noResults = document.getElementById("noResults");
  
  // Clear input field
  userInput.value = "";
  
  // Hide all sections
  errorMsg.style.display = "none";
  resultsSection.style.display = "none";
  noResults.style.display = "none";
  
  // Clear focus from input field
  userInput.blur();
}

// Fetch data from the API
async function fetchData() {
  const userInput = document.getElementById("userInput").value.trim();
  const loader = document.getElementById("loader");
  const errorMsg = document.getElementById("errorMsg");
  const resultsSection = document.getElementById("resultsSection");
  const resultsContainer = document.getElementById("resultsContainer");
  const noResults = document.getElementById("noResults");
  const searchButton = document.getElementById("searchButton");
  const queryDisplay = document.getElementById("queryDisplay");
  
  if (!userInput) {
    errorMsg.textContent = "Please enter a skill or interest.";
    errorMsg.style.display = "block";
    return;
  }
  
  // Disable button and show loader
  searchButton.disabled = true;
  loader.style.display = "block";
  errorMsg.style.display = "none";
  resultsSection.style.display = "none";
  noResults.style.display = "none";
  resultsContainer.innerHTML = "";
  
  try {
    const response = await fetch(`http://localhost:8000/search?query=${encodeURIComponent(userInput)}`);
    
    if (!response.ok) {
      throw new Error("Server error or invalid response.");
    }
    
    const data = await response.json();
    
    // Add the query to search history
    addToHistory(userInput);
    
    // Check if we have any results
    const hasResults = Object.values(data).some(arr => arr.length > 0);
    
    if (!hasResults) {
      noResults.style.display = "block";
    } else {
      // Display the query text
      queryDisplay.textContent = userInput;
      
      // Create result cards
      createResultCard("Jobs", data.jobs, "fa-briefcase");
      createResultCard("Certifications", data.certifications, "fa-certificate");
      createResultCard("YouTube Channels", data.youtube_channels, "fa-youtube");
      createResultCard("eBooks", data.ebooks, "fa-book");
      createResultCard("Websites", data.websites, "fa-globe");
      
      resultsSection.style.display = "block";
    }
  } catch (err) {
    errorMsg.textContent = err.message;
    errorMsg.style.display = "block";
  } finally {
    loader.style.display = "none";
    searchButton.disabled = false;
  }
}

// Create a result card
function createResultCard(title, items, iconClass) {
  const resultsContainer = document.getElementById("resultsContainer");
  
  const card = document.createElement("div");
  card.className = "result-card";
  
  let cardContent = `<h3><i class="fas ${iconClass}"></i> ${title}</h3>`;
  
  if (items && items.length > 0) {
    cardContent += '<ul>';
    items.forEach(item => {
      cardContent += `<li>${item}</li>`;
    });
    cardContent += '</ul>';
  } else {
    cardContent += `<p class="empty">No ${title.toLowerCase()} found for this query.</p>`;
  }
  
  card.innerHTML = cardContent;
  resultsContainer.appendChild(card);
}

// Initialize the history display
updateHistoryDisplay();

// Add event listener for Enter key
document.getElementById("userInput").addEventListener("keypress", function(event) {
  if (event.key === "Enter") {
    fetchData();
  }
});