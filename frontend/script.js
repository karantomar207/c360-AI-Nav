    // Store search history with mode
    let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];

    // Current user mode (student or professional)
    let currentMode = "student";

    // Initialize the application
    document.addEventListener('DOMContentLoaded', function() {
      setupNavigation();
      setupModeSelection();
      updateHistoryDisplay();
      
      document.getElementById('userInput').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
          fetchDataFromAPI();
        }
      });
    });

    function setupNavigation() {
      const navItems = document.querySelectorAll('.nav-item');
      navItems.forEach(item => {
        item.addEventListener('click', function() {
          navItems.forEach(navItem => navItem.classList.remove('active'));
          this.classList.add('active');
          const page = this.getAttribute('data-page');
          document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
          document.getElementById(page + 'Page').classList.add('active');
        });
      });
    }

    function setupModeSelection() {
      const modeOptions = document.querySelectorAll('.mode-option');
      modeOptions.forEach(option => {
        option.addEventListener('click', function() {
          modeOptions.forEach(opt => opt.classList.remove('active'));
          this.classList.add('active');
          currentMode = this.getAttribute('data-mode');
          const modeIndicator = document.getElementById('modeIndicator');
          modeIndicator.textContent = currentMode === "student" ? "Student Mode" : "Professional Mode";
          modeIndicator.className = currentMode === "student" ? "mode-indicator" : "mode-indicator professional";
          const resultsSection = document.getElementById("resultsSection");
          if (resultsSection.style.display === "block") {
            fetchDataFromAPI();
          }
        });
      });
    }

    function updateHistoryDisplay() {
      const historyList = document.getElementById('searchHistory');
      historyList.innerHTML = '';
      if (searchHistory.length === 0) {
        historyList.innerHTML = '<li class="history-item">No recent searches</li>';
        return;
      }
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
    function setKeyword(keyword) {
      document.getElementById('userInput').value = keyword;
      fetchData();
    }

    function setKeywordWithMode(keyword, mode) {
      document.getElementById('userInput').value = keyword;
      document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
      document.querySelector('.nav-item[data-page="explore"]').classList.add('active');
      document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
      document.getElementById('explorePage').classList.add('active');
      const modeOptions = document.querySelectorAll('.mode-option');
      modeOptions.forEach(option => {
        if (option.getAttribute('data-mode') === mode) {
          option.click();
        }
      });
      fetchDataFromAPI();
    }

    function removeFromHistory(index) {
      searchHistory.splice(index, 1);
      localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
      updateHistoryDisplay();
    }

    function clearAllHistory() {
      searchHistory = [];
      localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
      updateHistoryDisplay();
    }

    function addToHistory(query) {
      const historyItem = {
        query: query,
        mode: currentMode,
        timestamp: new Date().getTime()
      };
      searchHistory = searchHistory.filter(item => item.query !== query);
      searchHistory.unshift(historyItem);
      if (searchHistory.length > 10) {
        searchHistory.pop();
      }
      localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
      updateHistoryDisplay();
    }

    function clearResults() {
      const userInput = document.getElementById("userInput");
      const errorMsg = document.getElementById("errorMsg");
      const resultsSection = document.getElementById("resultsSection");
      const noResults = document.getElementById("noResults");
      userInput.value = "";
      errorMsg.style.display = "none";
      resultsSection.style.display = "none";
      noResults.style.display = "none";
      userInput.blur();
    }

    function displayResultsLineByLine(data, query) {
      const resultsContainer = document.getElementById("resultsContainer");
      const queryDisplay = document.getElementById("queryDisplay");
      queryDisplay.textContent = query;
      resultsContainer.innerHTML = "";
      resultsContainer.classList.remove("student-mode", "professional-mode");
      resultsContainer.classList.add(currentMode === "student" ? "student-mode" : "professional-mode");

      const sections = [
        { title: "Jobs", data: data.jobs, icon: "fa-briefcase" },
        { title: "Certifications", data: data.certifications, icon: "fa-certificate" },
        { title: "YouTube Channels", data: data.youtube_channels, icon: "fa-youtube" },
        { title: "Ebooks", data: data.ebooks, icon: "fa-book" },
        { title: "Websites", data: data.websites, icon: "fa-globe" }
      ];

      sections.forEach((section, sectionIndex) => {
        setTimeout(() => {
          createHorizontalSection(section.title, section.data, section.icon);
        }, sectionIndex * 100);
      });
    }

    function createHorizontalSection(title, items, iconClass) {
      const resultsContainer = document.getElementById("resultsContainer");
      const section = document.createElement("div");
      section.className = "result-section";
      section.innerHTML = `
        <h3><i class="fas ${iconClass}"></i> ${title}</h3>
        <div class="result-content" id="content-${title.toLowerCase().replace(' ', '-')}">
        </div>
      `;
      resultsContainer.appendChild(section);
      const contentDiv = section.querySelector('.result-content');

      if (items && items.length > 0) {
        items.forEach((item, index) => {
          setTimeout(() => {
            const itemSpan = document.createElement('span');
            itemSpan.className = 'result-item';
            itemSpan.textContent = item;
            itemSpan.style.animationDelay = `${index * 0.1}s`;
            contentDiv.appendChild(itemSpan);
          }, index * 50);
        });
      } else {
        contentDiv.innerHTML = `<span class="empty-result">No ${title.toLowerCase()} found for this query.</span>`;
      }
    }

    async function fetchDataFromAPI() {
      const userInput = document.getElementById("userInput").value.trim();
      const loader = document.getElementById("loader");
      const errorMsg = document.getElementById("errorMsg");
      const resultsSection = document.getElementById("resultsSection");
      const noResults = document.getElementById("noResults");
      const searchButton = document.getElementById("searchButton");

      if (!userInput) {
        errorMsg.textContent = "Please enter a skill or interest.";
        errorMsg.style.display = "block";
        return;
      }

      searchButton.disabled = true;
      loader.style.display = "block";
      errorMsg.style.display = "none";
      resultsSection.style.display = "none";
      noResults.style.display = "none";

      try {
        const apiUrl = `http://localhost:8000/search?query=${encodeURIComponent(userInput)}&mode=${currentMode}`;
        const response = await fetch(apiUrl);

        if (!response.ok) {
          throw new Error("Server error or invalid response.");
        }

        const data = await response.json();
        console.log("API Response:", data);

        const hasResults = Object.values(data).some(arr => arr && arr.length > 0);

        if (hasResults) {
          addToHistory(userInput);
          resultsSection.style.display = "block";
          displayResultsLineByLine(data, userInput);
        } else {
          noResults.style.display = "block";
        }
      } catch (err) {
        errorMsg.textContent = `Error: ${err.message}`;
        errorMsg.style.display = "block";
      } finally {
        loader.style.display = "none";
        searchButton.disabled = false;
      }
    }
