let map = null;
let overlayLayer = null;

// --- REGION ANALYSIS ---
async function runRegionAnalysis() {
    const region = document.getElementById('regionSelect').value;
    const btn = document.querySelector('#region-tool .btn-primary');
    
    btn.innerText = "Analyzing...";
    btn.disabled = true;

    try {
        const response = await fetch('/analyze_region', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ region: region })
        });
        const data = await response.json();
        
        if(data.error) throw new Error(data.error);

        // Show Results
        document.getElementById('regionResults').classList.remove('hidden');
        document.getElementById('regionGraph').src = data.plot_image;
        document.getElementById('regionPercent').innerText = data.stats.percentage + "%";

        // Render Map
        initMap('regionMap', data.coords, data.mask_image);

    } catch (e) {
        alert("Error: " + e.message);
    } finally {
        btn.innerText = "Analyze Region";
        btn.disabled = false;
    }
}

// --- UPLOAD ANALYSIS ---
async function runUploadAnalysis() {
    const pre = document.getElementById('preFile').files[0];
    const post = document.getElementById('postFile').files[0];
    const btn = document.querySelector('#upload-tool .btn-primary');

    if(!pre || !post) { alert("Please upload both images."); return; }

    btn.innerText = "Processing...";
    btn.disabled = true;

    const formData = new FormData();
    formData.append('pre_image', pre);
    formData.append('post_image', post);

    try {
        const response = await fetch('/analyze_upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if(data.error) throw new Error(data.error);

        // Show Results
        document.getElementById('uploadResults').classList.remove('hidden');
        document.getElementById('uploadResultImg').src = data.result_image;
        document.getElementById('uploadGraph').src = data.plot_image;
        document.getElementById('uploadPercent').innerText = data.stats.percentage + "%";

    } catch (e) {
        alert("Error: " + e.message);
    } finally {
        btn.innerText = "Detect Flood";
        btn.disabled = false;
    }
}

// --- MAP HELPER ---
function initMap(divId, coords, maskUrl) {
    // Destroy previous map instance if exists
    if (map) { map.off(); map.remove(); map = null; }

    const lat = parseFloat(coords[0]);
    const lng = parseFloat(coords[1]);

    map = L.map(divId).setView([lat, lng], 10);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap'
    }).addTo(map);

    if (maskUrl) {
        const bounds = [[lat - 0.1, lng - 0.1], [lat + 0.1, lng + 0.1]];
        overlayLayer = L.imageOverlay(maskUrl, bounds, { opacity: 0.7 }).addTo(map);
        map.fitBounds(bounds);
    }
    
    // Fix Map Size Glitch
    setTimeout(() => { map.invalidateSize(); }, 500);
}