document.addEventListener('DOMContentLoaded', () => {
    // --- Références aux éléments du DOM ---
    const imageUpload = document.getElementById('image-upload');
    const paramsContainer = document.getElementById('params-container');
    const processButton = document.getElementById('process-button');
    const defaultButton = document.getElementById('default-button');
    const logConsole = document.getElementById('log-console');
    const placeholder = document.getElementById('placeholder');
    const mainDisplay = document.getElementById('main-display');
    const originalImageComp = document.getElementById('original-image-comp');
    const resultWrapper = document.getElementById('result-wrapper');
    const resultImageComp = document.getElementById('result-image-comp');
    const comparisonContainer = document.getElementById('comparison-container');
    const comparisonSlider = document.getElementById('comparison-slider');
    
    const controls = [
        { sliderId: 'patch_size', inputId: 'patch_size_value', configPath: 'algorithm.patch_size' },
        { sliderId: 'omega', inputId: 'omega_value', configPath: 'algorithm.omega' },
        { sliderId: 'atmospheric_light_percentile', inputId: 'atmospheric_light_percentile_value', configPath: 'algorithm.atmospheric_light_percentile' },
        { sliderId: 't0', inputId: 't0_value', configPath: 'algorithm.t0' },
        { sliderId: 'gf_radius', inputId: 'gf_radius_value', configPath: 'refinement.guided_filter.radius' },
        { sliderId: 'gf_epsilon', inputId: 'gf_epsilon_value', configPath: 'refinement.guided_filter.epsilon' }
    ];
    let imageFile = null;
    let eventSource = null;
    let defaultConfig = null;

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
    function updateControlsFromConfig(config) {
        controls.forEach(control => {
            const slider = document.getElementById(control.sliderId);
            const input = document.getElementById(control.inputId);
            const path = control.configPath.split('.');
            const value = path.reduce((obj, key) => obj && obj[key], config);
            if (value !== undefined) {
                slider.value = value;
                input.value = formatValue(input, value);
            }
        });
    }
    
    function formatValue(inputElement, value) {
        const step = parseFloat(inputElement.step) || 1;
        if (step < 1) {
            const decimals = step.toString().split('.')[1]?.length || 0;
            return parseFloat(value).toFixed(decimals);
        }
        return parseInt(value);
    }

    // --- Gestion des événements ---
    
    controls.forEach(control => {
        const slider = document.getElementById(control.sliderId);
        const input = document.getElementById(control.inputId);
        
        slider.addEventListener('input', () => {
            input.value = formatValue(input, slider.value);
        });
        input.addEventListener('change', () => {
            let value = parseFloat(input.value);
            const min = parseFloat(slider.min);
            const max = parseFloat(slider.max);
            if (isNaN(value)) value = min;
            value = Math.max(min, Math.min(max, value));
            
            slider.value = value;
            input.value = formatValue(input, value);
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
    
    defaultButton.addEventListener('click', () => {
        if (defaultConfig) {
            updateControlsFromConfig(defaultConfig);
            addLog("Paramètres réinitialisés aux valeurs par défaut.", "info");
        } else {
            addLog("Erreur: Configuration par défaut non chargée.", "error");
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
        controls.forEach(c => {
             const key = c.sliderId;
             const value = document.getElementById(c.sliderId).value;
             formData.append(key, value);
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
            addLog(`Erreur lors du lancement: ${error.message}`, 'error');
            processButton.disabled = false;
            processButton.textContent = 'Lancer le Traitement';
        }
    });
    // Connexion au flux de logs SSE
    function connectToLogStream(jobId) {
        addLog(`Tâche démarrée avec l'ID: ${jobId}`, 'success');
        if(eventSource) eventSource.close();
        eventSource = new EventSource(`/stream-logs/${jobId}`);
        
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data.replace(/'/g, '"'));
            
            if(data.type === 'log') addLog(data.message);
            else if(data.type === 'result') {
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
            if(eventSource) eventSource.close();
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
        let percentage = (offsetX / rect.width) * 100;
        percentage = Math.max(0, Math.min(100, percentage));
        resultWrapper.style.width = `${percentage}%`;
        comparisonSlider.style.left = `${percentage}%`;
    });
    
    // --- Initialisation ---
    resetUI();
    
    fetch('/default-config')
        .then(res => res.json())
        .then(config => {
             defaultConfig = config;
             updateControlsFromConfig(defaultConfig);
        }).catch(e => {
            console.error("Impossible de charger la config par défaut", e);
            addLog("Impossible de charger la config par défaut.", "error");
        });
});
