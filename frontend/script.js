// Store search history with mode
let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];

// Current user mode (student or professional)
let currentMode = "student";

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  // Set up navigation
  setupNavigation();
  
  // Set up mode selection
  setupModeSelection();
  
  // Update the history display
  updateHistoryDisplay();
});

// Setup navigation functionality
function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  
  navItems.forEach(item => {
    item.addEventListener('click', function() {
      // Remove active class from all nav items
      navItems.forEach(navItem => navItem.classList.remove('active'));
      
      // Add active class to clicked nav item
      this.classList.add('active');
      
      // Get the page to show
      const page = this.getAttribute('data-page');
      
      // Hide all pages
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      
      // Show the selected page
      document.getElementById(page + 'Page').classList.add('active');
    });
  });
}

// Setup mode selection functionality
function setupModeSelection() {
  const modeOptions = document.querySelectorAll('.mode-option');
  
  modeOptions.forEach(option => {
    option.addEventListener('click', function() {
      // Remove active class from all mode options
      modeOptions.forEach(opt => opt.classList.remove('active'));
      
      // Add active class to clicked mode option
      this.classList.add('active');
      
      // Set current mode
      currentMode = this.getAttribute('data-mode');
      
      // Update mode indicator
      const modeIndicator = document.getElementById('modeIndicator');
      modeIndicator.textContent = currentMode === "student" ? "Student Mode" : "Professional Mode";
      modeIndicator.className = currentMode === "student" ? 
        "mode-indicator" : "mode-indicator professional";
      
      // If we have results displayed, refresh the search to show mode-specific results
      const resultsSection = document.getElementById("resultsSection");
      if (resultsSection.style.display === "block") {
        fetchData();
      }
    });
  });
}

// Update the search history display
function updateHistoryDisplay() {
  const historyList = document.getElementById('searchHistory');
  historyList.innerHTML = '';
  
  if (searchHistory.length === 0) {
    historyList.innerHTML = '<li class="history-item">No recent searches</li>';
    return;
  }
  
  // Display all history items
  searchHistory.forEach((item, index) => {
    const historyItem = document.createElement('li');
    historyItem.className = 'history-item';
    
    const modeClass = item.mode === "student" ? "student" : "professional";
    const modeText = item.mode === "student" ? "Student" : "Professional";
    
    historyItem.innerHTML = `
      <div class="history-query">
        <i class="fas fa-history"></i>
        <span>${item.query}</span>
        <span class="history-mode ${modeClass}">${modeText}</span>
      </div>
      <div class="history-actions">
        <button onclick="setKeywordWithMode('${item.query}', '${item.mode}')">
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

// Set a keyword to the search input with mode
function setKeywordWithMode(keyword, mode) {
  document.getElementById('userInput').value = keyword;
  
  // Switch to explore page
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  document.querySelector('.nav-item[data-page="explore"]').classList.add('active');
  
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('explorePage').classList.add('active');
  
  // Set the mode
  const modeOptions = document.querySelectorAll('.mode-option');
  modeOptions.forEach(option => {
    if (option.getAttribute('data-mode') === mode) {
      option.click();
    }
  });
  
  // Execute search
  fetchData();
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

// Clear all search history
function clearAllHistory() {
  searchHistory = [];
  localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  updateHistoryDisplay();
}

// Add search query to history
function addToHistory(query) {
  // Create history item with query and current mode
  const historyItem = {
    query: query,
    mode: currentMode,
    timestamp: new Date().getTime()
  };
  
  // Remove the query if it already exists to avoid duplicates
  searchHistory = searchHistory.filter(item => item.query !== query);
  
  // Add the new query to the beginning
  searchHistory.unshift(historyItem);
  
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
    // Construct the URL for the API request
    const apiUrl = `http://localhost:8000/search?query=${encodeURIComponent(userInput)}&mode=${currentMode}`;
    
    // Fetch the data from the API
    const response = await fetch(apiUrl);
    console.log(response)

    // Check if the response is OK
    if (!response.ok) {
      throw new Error("Server error or invalid response.");
    }

    // Parse the response body as JSON
    const data = await response.json();

    // Log the response data to inspect it
    console.log("API Response:", data);

    // Add the query to search history only if there are results
    const hasResults = Object.values(data).some(arr => arr && arr.length > 0);

    if (hasResults) {
      addToHistory(userInput);  // Add to search history
    }

    // Check if there are results and display "No Results" if needed
    if (!hasResults) {
      noResults.style.display = "block";
    } else {
      // Display the query text
      queryDisplay.textContent = userInput;

      // Set mode-specific class on results container
      resultsContainer.classList.remove("student-mode", "professional-mode");
      resultsContainer.classList.add(currentMode === "student" ? "student-mode" : "professional-mode");

      // Create result cards based on user mode
      if (currentMode === "student") {
        createResultCard("Jobs", data.jobs, "fa-briefcase");
        createResultCard("Certifications", data.certifications, "fa-certificate");
        createResultCard("YouTube Channels", data.youtube_channels, "fa-youtube");
        createResultCard("Ebooks", data.ebooks, "fa-book");
        createResultCard("Websites", data.websites, "fa-globe");
      } else {
        createResultCard("Jobs", data.jobs, "fa-briefcase");
        createResultCard("Certifications", data.certifications, "fa-certificate");
        createResultCard("YouTube Channels", data.youtube_channels, "fa-youtube");
        createResultCard("Ebooks", data.ebooks, "fa-book");
        createResultCard("Websites", data.websites, "fa-globe");
      }

      resultsSection.style.display = "block";
    }
  } catch (err) {
    // Handle any errors
    errorMsg.textContent = `Error: ${err.message}`;
    errorMsg.style.display = "block";
  } finally {
    // Always reset loader and search button
    loader.style.display = "none";
    searchButton.disabled = false;
  }
}

// Create a result card
function createResultCard(title, items, iconClass) {
  const resultsContainer = document.getElementById("resultsContainer");
  const card = document.createElement("div");
  card.className = "result-card";
  
  let cardContent = `
    <h4><i class="fas ${iconClass}"></i> ${title}</h4>
    <ul>`;
  
  if (items && items.length > 0) {
    items.forEach(item => {
      cardContent += `<li>${item}</li>`;
    });
  } else {
    cardContent += `<p class="empty">No ${title.toLowerCase()} found for this query.</p>`;
  }
  
  cardContent += `</ul>`;
  card.innerHTML = cardContent;
  resultsContainer.appendChild(card);
}
