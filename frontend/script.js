const API_URL = "http://127.0.0.1:8000/api/v1/summarize";

// DOM Elements
const textInput = document.getElementById('textInput');
const lengthSelect = document.getElementById('lengthSelect');
const modelSelect = document.getElementById('modelSelect');
const summarizeBtn = document.getElementById('summarizeBtn');
const summaryOutput = document.getElementById('summaryOutput');
const copyBtn = document.getElementById('copyBtn');
const themeToggle = document.getElementById('themeToggle');
const toast = document.getElementById('toast');
const statsContainer = document.getElementById('statsContainer');
const statOriginal = document.getElementById('statOriginal');
const statSummary = document.getElementById('statSummary');
const statRatio = document.getElementById('statRatio');
const modelWarning = document.getElementById('modelWarning');

// State
let isDark = true;

// Theme Handling
function toggleTheme() {
    isDark = !isDark;
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    themeToggle.textContent = isDark ? 'Light Mode' : 'Dark Mode';
}

themeToggle.addEventListener('click', toggleTheme);

// Model Selection Handling
modelSelect.addEventListener('change', () => {
    const selectedModel = modelSelect.value;
    // Show warning for slow models
    if (selectedModel === 'pegasus' || selectedModel === 'bart-large') {
        modelWarning.style.display = 'flex';
    } else {
        modelWarning.style.display = 'none';
    }
});

// Summarize Function
async function handleSummarize() {
    const text = textInput.value.trim();
    const length = lengthSelect.value;
    const model = modelSelect.value;

    // Reset UI
    summaryOutput.innerHTML = '';
    summaryOutput.classList.remove('error-msg');
    statsContainer.style.display = 'none';

    // Validation
    if (!text) {
        showError("Please enter some text to summarize.");
        return;
    }

    if (text.split(/\s+/).length < 5) {
        showError("Text is too short. Please enter at least 5 words.");
        return;
    }

    // Set Loading State
    setLoading(true);

    // Create AbortController for timeout
    const controller = new AbortController();
    // 90s timeout to allow for slow models if user explicitly chose them
    // The backend has its own 5s timeout for fast models
    const timeoutId = setTimeout(() => controller.abort(), 90000);

    try {
        const response = await fetch(API_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text, length, model }),
            signal: controller.signal
        });

        clearTimeout(timeoutId); // Clear timeout if successful

        if (!response.ok) {
            // Try to parse JSON error from backend
            const errorData = await response.json().catch(() => null);
            const errorMessage = errorData?.detail || `Server Error: ${response.status}`;
            throw new Error(errorMessage);
        }

        const data = await response.json();

        // Validate Response
        if (!data.summary) {
            throw new Error("Received empty summary from server.");
        }

        // Display Result
        displayResult(data);

    } catch (error) {
        console.error("Summarization failed:", error);
        let msg = "Failed to connect to the server.";

        if (error.name === 'AbortError') {
            msg = "Request timed out. The server might be busy loading the model. Please try again.";
        } else if (error.message.includes("Failed to fetch")) {
            msg = "Failed to connect to backend. Is the server running on port 8000?";
        } else if (error.message) {
            msg = error.message;
        }

        showError(msg);
    } finally {
        setLoading(false);
    }
}

function displayResult(data) {
    summaryOutput.textContent = data.summary;

    // Update Stats
    statOriginal.textContent = data.original_word_count;
    statSummary.textContent = data.summary_word_count;
    statRatio.textContent = `${data.compression_ratio}x`;
    statsContainer.style.display = 'flex';

    // Scroll into view
    document.querySelector('.output-section').scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    summaryOutput.textContent = message;
    summaryOutput.classList.add('error-msg');
}

function setLoading(isLoading) {
    summarizeBtn.disabled = isLoading;
    if (isLoading) {
        summarizeBtn.classList.add('loading');
        summarizeBtn.innerHTML = '<div class="spinner"></div> Generating...';
    } else {
        summarizeBtn.classList.remove('loading');
        summarizeBtn.innerHTML = 'Summarize';
    }
}

// Copy Functionality
copyBtn.addEventListener('click', () => {
    const text = summaryOutput.textContent;
    if (!text || summaryOutput.classList.contains('error-msg')) return;

    navigator.clipboard.writeText(text).then(() => {
        showToast("Copied");
    }).catch(err => {
        console.error('Failed to copy:', err);
        showToast("Failed to copy");
    });
});

function showToast(message) {
    toast.textContent = message;
    toast.classList.add('show');
    setTimeout(() => {
        toast.classList.remove('show');
    }, 2000);
}

// Event Listeners
summarizeBtn.addEventListener('click', handleSummarize);

// Allow Enter to submit (without Shift)
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault(); // Prevent new line
        handleSummarize();
    }
});
