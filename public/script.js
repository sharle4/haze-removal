document.addEventListener('DOMContentLoaded', () => {
    // --- Références aux éléments du DOM ---
    const imageUpload = document.getElementById('image-upload');
    const paramsContainer = document.getElementById('params-container');
    const processButton = document.getElementById('process-button');
    const logConsole = document.getElementById('log-console');
    const placeholder = document.getElementById('placeholder');
    const mainDisplay = document.getElementById('main-display');
    const originalImageComp = document.getElementById('original-image-comp');
    const resultWrapper = document.getElementById('result-wrapper');
    const resultImageComp = document.getElementById('result-image-comp');
    const comparisonContainer = document.getElementById('comparison-container');
    const comparisonSlider = document.getElementById('comparison-slider');
    
    const sliders = [
        { id: 'patch_size', valueId: 'patch_size_value' },
        { id: 'omega', valueId: 'omega_value' },
        { id: 'atmospheric_light_percentile', valueId: 'atmospheric_light_percentile_value' },
        { id: 't0', valueId: 't0_value' },
        { id: 'gf_radius', valueId: 'gf_radius_value' },
        { id: 'gf_epsilon', valueId: 'gf_epsilon_value' }
    ];
    let imageFile = null;
    let eventSource = null;
    // --- Fonctions utilitaires ---
    function addLog(message, type = 'info') {
        if (logConsole.querySelector('.text-gray-500')) {
            logConsole.innerHTML = '';
        }
        const p = document.createElement('p');
        p.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        if (type === 'error') {
            p.className = 'text-red-400';
        } else if (type === 'success') {
             p.className = 'text-blue-400';
        }
        logConsole.appendChild(p);
        logConsole.scrollTop = logConsole.scrollHeight;
    }
    function setIntermediateImage(name, base64Data) {
        const imgElement = document.getElementById(`img-${name}`);
        if (imgElement) {
            imgElement.src = base64Data;
            imgElement.classList.remove('hidden');
        }
         if (name === 'final_result') {
            resultImageComp.src = base64Data;
            resultWrapper.style.width = '50%';
        }
    }
    
    function resetUI() {
        processButton.disabled = true;
        processButton.textContent = 'Lancer le Traitement';
        
        ['dark_channel', 'initial_transmission', 'refined_transmission', 'final_result'].forEach(name => {
            const img = document.getElementById(`img-${name}`);
            if (img) {
                img.src = "";
                img.classList.add('hidden');
            }
        });
        mainDisplay.classList.add('hidden');
        placeholder.classList.remove('hidden');
    }
    // --- Gestion des événements ---
    
    sliders.forEach(slider => {
        const input = document.getElementById(slider.id);
        const valueSpan = document.getElementById(slider.valueId);
        input.addEventListener('input', () => {
            const decimals =
                slider.id === 'atmospheric_light_percentile' ? 5 :
                slider.id === 'gf_epsilon' ? 4 :
                (slider.id.includes('omega') || slider.id.includes('t0') ? 2 : 0);
            valueSpan.textContent = parseFloat(input.value).toFixed(decimals);
        });
    });
    imageUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            imageFile = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                originalImageComp.src = event.target.result;
                placeholder.classList.add('hidden');
                mainDisplay.classList.remove('hidden');
                resultImageComp.src = "";
                resultWrapper.style.width = '0%';
            };
            reader.readAsDataURL(imageFile);
            paramsContainer.classList.remove('opacity-50', 'pointer-events-none');
            processButton.disabled = false;
            addLog('Image chargée. Prêt à traiter.', 'success');
        }
    });
    
    processButton.addEventListener('click', async () => {
        if (!imageFile) {
            addLog("Erreur: Aucune image n'est chargée.", 'error');
            return;
        }
        processButton.disabled = true;
        processButton.textContent = 'Traitement en cours...';
        logConsole.innerHTML = '';
        addLog('Initialisation du traitement...');
        const formData = new FormData();
        formData.append('image', imageFile);
        sliders.forEach(slider => {
             formData.append(slider.id, document.getElementById(slider.id).value);
        });
        
        try {
            // Étape 1: Envoyer la requête pour démarrer le job
            const response = await fetch('/process-image/', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Erreur du serveur.');
            }
            const data = await response.json();
            const jobId = data.job_id;
            addLog(`Tâche démarrée avec l'ID: ${jobId}`, 'success');
            // Étape 2: Se connecter au flux SSE pour les logs
            connectToLogStream(jobId);
        } catch (error) {
            addLog(`Erreur lors du lancement du traitement: ${error.message}`, 'error');
            processButton.disabled = false;
            processButton.textContent = 'Lancer le Traitement';
        }
    });
    function connectToLogStream(jobId) {
        if(eventSource) {
            eventSource.close();
        }
        eventSource = new EventSource(`/stream-logs/${jobId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data.replace(/'/g, '"'));
            
            if(data.type === 'log') {
                addLog(data.message);
            } else if(data.type === 'result') {
                addLog(`Résultat intermédiaire reçu: ${data.name}`, 'success');
                setIntermediateImage(data.name, data.image);
            } else if(data.type === 'done') {
                addLog(data.message, 'success');
                processButton.disabled = false;
                processButton.textContent = 'Lancer un nouveau traitement';
                eventSource.close();
            } else if (data.type === 'error'){
                addLog(`Erreur: ${data.message}`, 'error');
                processButton.disabled = false;
                processButton.textContent = 'Réessayer le traitement';
                eventSource.close();
            }
        };
        
        eventSource.onerror = () => {
            addLog('Connexion au serveur perdue.', 'error');
            processButton.disabled = false;
            processButton.textContent = 'Lancer le Traitement';
            eventSource.close();
        };
    }
    // --- Gestion du slider de comparaison ---
    let isDragging = false;
    comparisonSlider.addEventListener('mousedown', () => { isDragging = true; });
    document.addEventListener('mouseup', () => { isDragging = false; });
    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const rect = comparisonContainer.getBoundingClientRect();
        let offsetX = e.clientX - rect.left;
        let width = Math.max(0, Math.min(offsetX, rect.width));
        let percentage = (width / rect.width) * 100;
        resultWrapper.style.width = `${percentage}%`;
        comparisonSlider.style.left = `${percentage}%`;
    });
    // --- Initialisation ---
    resetUI();
    
    fetch('/default-config')
        .then(res => res.json())
        .then(config => {
             document.getElementById('patch_size').value = config.algorithm.patch_size;
             document.getElementById('omega').value = config.algorithm.omega;
             document.getElementById('atmospheric_light_percentile').value = config.algorithm.atmospheric_light_percentile;
             document.getElementById('t0').value = config.algorithm.t0;
             document.getElementById('gf_radius').value = config.refinement.guided_filter.radius;
             document.getElementById('gf_epsilon').value = config.refinement.guided_filter.epsilon;
             
             sliders.forEach(s => document.getElementById(s.id).dispatchEvent(new Event('input')));
        }).catch(e => console.error("Impossible de charger la config par défaut", e));
});
