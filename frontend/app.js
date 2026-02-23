// API Base URL - update when deploying backend
const API_BASE = 'http://localhost:8000/api';

// DOM Elements
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadPrompt = document.getElementById('upload-prompt');
const imagePreview = document.getElementById('image-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const stateSelect = document.getElementById('state-select');

// State views
const welcomeState = document.getElementById('welcome-state');
const loadingState = document.getElementById('loading-state');
const resultsDashboard = document.getElementById('results-dashboard');

// Result elements
const resClass = document.getElementById('res-class');
const resConf = document.getElementById('res-conf');
const resConfBar = document.getElementById('res-conf-bar');
const resPrice = document.getElementById('res-price');
const resTargetState = document.getElementById('res-target-state');
const heatmapImg = document.getElementById('heatmap-img');
const heatmapLoader = document.getElementById('heatmap-loader');

let selectedFile = null;
let shapChartInstance = null;

// --- Drag & Drop Handlers ---
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-green-500', 'bg-green-50');
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-green-500', 'bg-green-50');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-green-500', 'bg-green-50');
    if (e.dataTransfer.files.length > 0) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file.type.match('image.*')) {
        alert("Please upload an image file (JPG/PNG).");
        return;
    }
    selectedFile = file;

    // Preview image
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        imagePreview.classList.remove('hidden');
        uploadPrompt.classList.add('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

// --- Analysis Logic ---
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    const selectedState = stateSelect.value;

    // Show loading
    welcomeState.classList.add('hidden');
    resultsDashboard.classList.add('hidden');
    loadingState.classList.remove('hidden');
    analyzeBtn.disabled = true;

    try {
        // 1. Image Classification (/api/predict)
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('state', selectedState);

        const predictRes = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!predictRes.ok) throw new Error("Classification failed");
        const predictData = await predictRes.json();

        const wasteType = predictData.waste_type;
        const confidence = predictData.confidence;
        const classIndex = predictData.class_index;

        // Populate classification UI
        resClass.textContent = wasteType.replace(/_/g, ' ');
        resConf.textContent = confidence;
        resConfBar.style.width = `${confidence}%`;
        resTargetState.textContent = selectedState;

        // Show dashboard placeholder
        loadingState.classList.add('hidden');
        resultsDashboard.classList.remove('hidden');

        // Request parallel data (Price, Shap, Grad-CAM)
        heatmapLoader.classList.remove('hidden'); // Show gradcam loader

        Promise.allSettled([
            fetchPrice(selectedState, wasteType),
            fetchShap(selectedState, wasteType),
            fetchGradCam(selectedFile, classIndex)
        ]);

    } catch (err) {
        console.error(err);
        alert("Error analyzing image: " + err.message);
        loadingState.classList.add('hidden');
        welcomeState.classList.remove('hidden');
    } finally {
        analyzeBtn.disabled = false;
    }
});

async function fetchPrice(state, wasteType) {
    try {
        const res = await fetch(`${API_BASE}/price?state=${encodeURIComponent(state)}&waste_type=${encodeURIComponent(wasteType)}`);
        const data = await res.json();
        if (data.price_per_kg != null) {
            resPrice.textContent = `₹${data.price_per_kg.toFixed(2)}`;
        } else {
            resPrice.textContent = "N/A";
        }
    } catch (err) {
        console.warn("Pricing error", err);
        resPrice.textContent = "Error";
    }
}

async function fetchGradCam(file, classIndex) {
    try {
        console.log(`Fetching Grad-CAM for class ${classIndex}...`);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('class_index', classIndex);

        const res = await fetch(`${API_BASE}/gradcam`, {
            method: 'POST',
            body: formData
        });

        console.log(`Grad-CAM response status: ${res.status}`);
        const data = await res.json();
        console.log(`Grad-CAM data received:`, data.heatmap_b64 ? "Success (base64 length " + data.heatmap_b64.length + ")" : "No base64 data");

        if (data.heatmap_b64) {
            heatmapImg.src = `data:image/png;base64,${data.heatmap_b64}`;
        }
    } catch (err) {
        console.error("Grad-CAM error strictly UI:", err);
    } finally {
        heatmapLoader.classList.add('hidden');
    }
}

async function fetchShap(state, wasteType) {
    try {
        const res = await fetch(`${API_BASE}/shap?state=${encodeURIComponent(state)}&waste_type=${encodeURIComponent(wasteType)}`);
        const data = await res.json();
        renderShapChart(data.shap_contributions || []);
    } catch (err) {
        console.warn("SHAP error", err);
    }
}

function renderShapChart(contributions) {
    const ctx = document.getElementById('shapChart').getContext('2d');

    // Destroy previous chart if exists
    if (shapChartInstance) {
        shapChartInstance.destroy();
    }

    // Format data for Chart.js
    const labels = contributions.map(c => c.feature);
    const values = contributions.map(c => c.shap_value);
    const bgColors = values.map(v => v > 0 ? 'rgba(34, 197, 94, 0.6)' : 'rgba(239, 68, 68, 0.6)');
    const borderColors = values.map(v => v > 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)');

    shapChartInstance = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Price Impact (₹)',
                data: values,
                backgroundColor: bgColors,
                borderColor: borderColors,
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y', // Horizontal bars
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            let val = context.raw || 0;
                            let prefix = val > 0 ? '+' : '';
                            return `Impact: ${prefix}₹${val.toFixed(2)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: '#f3f4f6' },
                    title: { display: true, text: 'Effect on Price (₹)' }
                },
                y: {
                    grid: { display: false },
                    ticks: {
                        callback: function (value, index, values) {
                            let label = this.getLabelForValue(value);
                            // Shorten very long labels
                            if (label.length > 20) {
                                return label.substr(0, 17) + '...';
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}
