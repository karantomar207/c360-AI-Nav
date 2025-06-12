// Updated script.js to handle structured response data

let searchHistory = JSON.parse(localStorage.getItem('searchHistory')) || [];
let currentMode = "student";

document.addEventListener('DOMContentLoaded', function () {
  setupNavigation();
  setupModeSelection();
  updateHistoryDisplay();
});

function setupNavigation() {
  const navItems = document.querySelectorAll('.nav-item');
  navItems.forEach(item => {
    item.addEventListener('click', function () {
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
    option.addEventListener('click', function () {
      modeOptions.forEach(opt => opt.classList.remove('active'));
      this.classList.add('active');
      currentMode = this.getAttribute('data-mode');
      const modeIndicator = document.getElementById('modeIndicator');
      modeIndicator.textContent = currentMode === "student" ? "Student Mode" : "Professional Mode";
      modeIndicator.className = currentMode === "student" ? "mode-indicator" : "mode-indicator professional";
      const resultsSection = document.getElementById("resultsSection");
      if (resultsSection.style.display === "block") fetchData();
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

function setKeywordWithMode(keyword, mode) {
  document.getElementById('userInput').value = keyword;
  document.querySelectorAll('.nav-item').forEach(item => item.classList.remove('active'));
  document.querySelector('.nav-item[data-page="explore"]').classList.add('active');
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.getElementById('explorePage').classList.add('active');
  const modeOptions = document.querySelectorAll('.mode-option');
  modeOptions.forEach(option => {
    if (option.getAttribute('data-mode') === mode) option.click();
  });
  fetchData();
}

function setKeyword(keyword) {
  document.getElementById('userInput').value = keyword;
  fetchData();
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
  const historyItem = { query: query, mode: currentMode, timestamp: new Date().getTime() };
  searchHistory = searchHistory.filter(item => item.query !== query);
  searchHistory.unshift(historyItem);
  if (searchHistory.length > 10) searchHistory.pop();
  localStorage.setItem('searchHistory', JSON.stringify(searchHistory));
  updateHistoryDisplay();
}

function clearResults() {
  document.getElementById("userInput").value = "";
  document.getElementById("errorMsg").style.display = "none";
  document.getElementById("resultsSection").style.display = "none";
  document.getElementById("noResults").style.display = "none";
}

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

  searchButton.disabled = true;
  loader.style.display = "block";
  errorMsg.style.display = "none";
  resultsSection.style.display = "none";
  noResults.style.display = "none";
  resultsContainer.innerHTML = "";

  try {
      const apiUrl = `http://localhost:8000/search?prompt=${encodeURIComponent(userInput)}&mode=${currentMode}`;
    const response = await fetch(apiUrl);
    if (!response.ok) throw new Error("Server error or invalid response.");
    const data = await response.json();
    console.log("data", data.results.ebook)
    const hasResults = Object.values(data).some(arr => arr && arr.length > 0);
    if (hasResults) addToHistory(userInput);

    if (!hasResults) {
      noResults.style.display = "block";
    } else {
      queryDisplay.textContent = userInput;
      resultsContainer.className = currentMode === "student" ? "results-container student-mode" : "results-container professional-mode";

      if (data.results.jobs && data.results.jobs.length > 0) {
        createObjectCard("Jobs", data.results.jobs, "fa-briefcase", item => `
          <ul>
            ${item.title ? `<li><strong>${item.title}</strong></li>` : ''}
            ${item.description ? `<li>${item.description}</li>` : ''}
            ${item.location ? `<li>(${item.location})</li>` : ''}
            ${item.salary_description ? `<li><strong>Salary:</strong> ${item.salary_description}</li>` : ''}
            ${item.requirements ? `<li><strong>Requirements:</strong> ${item.requirements}</li>` : ''}
          </ul><hr />
        `);
      }
      
      if (data.results.certificates && data.results.certificates.length > 0) {
        createObjectCard("Certificates", data.results.certificates, "fa-certificate", item => `
          <ul>
            ${item.page_title ? `<li><strong>${item.page_title}</strong></li>` : ''}
            ${item.page_description ? `<li>${item.page_description}</li>` : ''}
            ${item.fee_detail ? `<li><strong>Fee:</strong> ${item.fee_detail}</li>` : ''}
            ${item.learning_term_details ? `<li><strong>Learning Terms & Detail:</strong> ${item.learning_term_details}</li>` : ''}
            ${item.job_details ? `<li><strong>Career:</strong> ${item.job_details}</li>` : ''}
          </ul><hr />
        `);
      }
      
      if (data.results.courses && data.results.courses.length > 0) {
        createObjectCard("Courses", data.results.courses, "fa-graduation-cap", item => `
          <ul>
            ${item.course_name ? `<li><strong>${item.course_name}</strong></li>` : ''}
            ${item.page_description ? `<li>${item.page_description}</li>` : ''}
            ${item.fee_detail ? `<li><strong>Fee Details:</strong> ${item.fee_detail}</li>` : ''}
            ${item.learning_term_details ? `<li><strong>Topics:</strong> ${item.learning_term_details}</li>` : ''}
            ${item.course_highlight ? `<li><strong>Course Highlights:</strong> ${item.course_highlight}</li>` : ''}
          </ul><hr />
        `);
      }
      
      if (data.results.ebooks && data.results.ebooks.length > 0) {
        createObjectCard("Ebooks", data.results.ebooks, "fa-book", item => `
          <ul>
            ${item.title ? `<li><strong>${item.title}</strong></li>` : ''}
            ${item.author ? `<li>by ${item.author}</li>` : ''}
            ${item.description ? `<li>${item.description}</li>` : ''}
            ${item.topics ? `<li><strong>Topics:</strong> ${item.topics}</li>` : ''}
            ${item.pdf_upload ? `<li><a href="https://cache.careers360.mobi/media/${item.pdf_upload}" target="blank" >View PDF</a></li>` : ''}
          </ul><hr />
        `);
      }

      resultsSection.style.display = "block";
    }
  } catch (err) {
    errorMsg.textContent = `Error: ${err.message}`;
    errorMsg.style.display = "block";
  } finally {
    loader.style.display = "none";
    searchButton.disabled = false;
  }
}

function createObjectCard(title, items, iconClass, renderItem) {
  const resultsContainer = document.getElementById("resultsContainer");
  const card = document.createElement("div");
  card.className = "result-card";

  let cardContent = `<h3><i class="fas ${iconClass}"></i> ${title}</h3>`;

  if (items && items.length > 0) {
    items.forEach(item => {
      cardContent += renderItem(item);
    });
  } else {
    cardContent += `<p class="empty">No ${title.toLowerCase()} found for this query.</p>`;
  }

  card.innerHTML = cardContent;
  resultsContainer.appendChild(card);
}

function createObjectCard(title, items, iconClass, renderItem) {
  const resultsContainer = document.getElementById("resultsContainer");
  const card = document.createElement("div");
  card.className = "result-card";

  let cardContent = `<h3><i class="fas ${iconClass}"></i> ${title}</h3>`;

  if (items && items.length > 0) {
    items.forEach(item => {
      cardContent += renderItem(item);
    });
  } else {
    cardContent += `<p class="empty">No ${title.toLowerCase()} found for this query.</p>`;
  }

  card.innerHTML = cardContent;
  resultsContainer.appendChild(card);

  // Animate list items line-by-line
  const listItems = card.querySelectorAll('li');
  listItems.forEach((li, index) => {
    li.classList.add('fade-in-line');
    li.style.animationDelay = `${index * 0.15}s`;
  });
}