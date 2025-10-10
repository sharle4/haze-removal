/**
 * Script principal pour l'interface de démonstration de l'algorithme "Dark Channel Prior".
 * Gère les interactions utilisateur, la communication avec l'API, et l'affichage des résultats.
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- État global de l'application ---
    let imageFile = null;
    let eventSource = null;
    let defaultConfig = null;
    let currentMode = 'single'; // 'single' ou 'experiment'
    let experimentParams = {};
    let selectedForComparison = [];

    // --- Références aux éléments du DOM ---
    const dom = {
        imageUpload: document.getElementById('image-upload'),
        paramsContainer: document.getElementById('params-container'),
        modeSelectionContainer: document.getElementById('mode-selection-container'),
        processButton: document.getElementById('process-button'),
        defaultButton: document.getElementById('default-button'),
        logConsole: document.getElementById('log-console'),
        logContainer: document.getElementById('log-container'),
        placeholder: document.getElementById('placeholder'),
        
        singleRunView: document.getElementById('single-run-view'),
        experimentView: document.getElementById('experiment-view'),

        // Vue "Analyse Unique"
        comparisonContainer: document.getElementById('comparison-container'),
        comparisonSlider: document.getElementById('comparison-slider'),
        originalImageComp: document.getElementById('original-image-comp'),
        resultWrapper: document.getElementById('result-wrapper'),
        resultImageComp: document.getElementById('result-image-comp'),
        intermediateResults: document.getElementById('intermediate-results'),

        // Vue "Expérimentale"
        experimentGrid: document.getElementById('experiment-grid'),
        originalImageExp: document.getElementById('original-image-exp'),
        resultsCount: document.getElementById('results-count'),
        totalRuns: document.getElementById('total-runs'),
        
        // Panneau de comparaison (Mode Expérimental)
        expComparisonPanel: document.getElementById('experiment-comparison-panel'),
        clearComparisonBtn: document.getElementById('clear-comparison-btn'),
        compImageA: document.getElementById('comp-image-a'),
        compParamsA: document.getElementById('comp-params-a'),
        compImageB: document.getElementById('comp-image-b'),
        compParamsB: document.getElementById('comp-params-b'),
        expComparisonContainer: document.getElementById('exp-comparison-container'),
        expCompImgA: document.getElementById('exp-comp-img-a'),
        expCompImgB: document.getElementById('exp-comp-img-b'),
        expCompWrapperB: document.getElementById('exp-comp-wrapper-b'),
        expComparisonSlider: document.getElementById('exp-comparison-slider'),
        
        // Sélecteurs de mode et infos
        modeSingleBtn: document.getElementById('mode-single'),
        modeExperimentBtn: document.getElementById('mode-experiment'),
        singleRunInfo: document.getElementById('single-run-info'),
        experimentRunInfo: document.getElementById('experiment-run-info'),

        // Paramètres dynamiques
        dynamicParams: document.getElementById('dynamic-params'),
        dynamicParamsGf: document.getElementById('dynamic-params-gf'),
    };

    // --- Configuration des paramètres ---
    const PARAM_CONFIG = {
        'algorithm': {
            label: 'Algorithme',
            container: dom.dynamicParams,
            params: {
                patch_size: { label: 'Taille du Patch', type: 'range', min: 3, max: 51, step: 2, default: 15 },
                omega: { label: 'Omega (Force)', type: 'range', min: 0.5, max: 1.0, step: 0.01, default: 0.95 },
                atmospheric_light_percentile: { label: '% Lumière Atmosphérique', type: 'range', min: 0.0001, max: 0.01, step: 0.0001, default: 0.001 },
                t0: { label: 'Transmission Min (t₀)', type: 'range', min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
            }
        },
        'guided_filter': {
            label: 'Filtre Guidé',
            container: dom.dynamicParamsGf,
            params: {
                gf_radius: { label: 'Rayon', type: 'range', min: 1, max: 150, step: 1, default: 60 },
                gf_epsilon: { label: 'Epsilon', type: 'range', min: 0.0001, max: 0.1, step: 0.0001, default: 0.001 },
            }
        }
    };
    
    // --- Fonctions de Logging et UI ---

    /**
     * Affiche un message dans la console de log de l'interface.
     * @param {string} message - Le message à afficher.
     * @param {'info'|'error'|'success'} type - Le type de message pour la coloration.
     */
    function addLog(message, type = 'info') {
        if (dom.logConsole.querySelector('.text-gray-500')) {
            dom.logConsole.innerHTML = '';
        }
        const p = document.createElement('p');
        p.innerHTML = `[<span class="text-gray-500">${new Date().toLocaleTimeString()}</span>] ${message}`;
        const typeClasses = {
            error: 'text-red-400',
            success: 'text-blue-400',
            info: 'text-green-400'
        };
        p.className = typeClasses[type] || 'text-green-400';
        dom.logConsole.appendChild(p);
        dom.logConsole.scrollTop = dom.logConsole.scrollHeight;
    }
    
    /**
     * Réinitialise l'interface à son état initial.
     */
    function resetUI() {
        dom.processButton.disabled = true;
        dom.processButton.textContent = 'Lancer le Traitement';
        
        dom.placeholder.classList.remove('hidden');
        dom.singleRunView.classList.add('hidden');
        dom.experimentView.classList.add('hidden');
        dom.logContainer.classList.add('hidden');
        
        dom.paramsContainer.classList.add('opacity-50', 'pointer-events-none');
        dom.modeSelectionContainer.classList.add('opacity-50', 'pointer-events-none');

        dom.experimentGrid.innerHTML = '';
        dom.resultsCount.textContent = '0';
        dom.totalRuns.textContent = '0';

        selectedForComparison = [];
        updateComparisonUI();


        if (eventSource) {
            eventSource.close();
        }
    }

    /**
     * Met à jour le mode de l'application et ajuste l'interface en conséquence.
     * @param {'single'|'experiment'} newMode - Le nouveau mode à activer.
     */
    function setMode(newMode) {
        currentMode = newMode;
        if (newMode === 'single') {
            dom.modeSingleBtn.classList.add('bg-blue-600', 'text-white');
            dom.modeSingleBtn.classList.remove('text-gray-400');
            dom.modeExperimentBtn.classList.remove('bg-blue-600', 'text-white');
            dom.modeExperimentBtn.classList.add('text-gray-400');
            dom.singleRunInfo.classList.remove('hidden');
            dom.experimentRunInfo.classList.add('hidden');
        } else {
            dom.modeExperimentBtn.classList.add('bg-blue-600', 'text-white');
            dom.modeExperimentBtn.classList.remove('text-gray-400');
            dom.modeSingleBtn.classList.remove('bg-blue-600', 'text-white');
            dom.modeSingleBtn.classList.add('text-gray-400');
            dom.singleRunInfo.classList.add('hidden');
            dom.experimentRunInfo.classList.remove('hidden');
        }
        renderParams();
    }
    
    // --- Génération dynamique des contrôles de paramètres ---

    /**
     * Crée et affiche les contrôles de paramètres en fonction du mode actuel.
     */
    function renderParams() {
        Object.values(PARAM_CONFIG).forEach(group => {
            group.container.innerHTML = '';
            Object.entries(group.params).forEach(([key, config]) => {
                const paramElement = createParamControl(key, config);
                group.container.appendChild(paramElement);
            });
        });
        if(defaultConfig) updateControlsFromConfig(defaultConfig);
    }

    /**
     * Crée un élément de contrôle pour un paramètre.
     * @param {string} key - La clé du paramètre (ex: 'patch_size').
     * @param {object} config - La configuration du paramètre.
     * @returns {HTMLElement} L'élément HTML du contrôle.
     */
    function createParamControl(key, config) {
        const wrapper = document.createElement('div');
        wrapper.className = 'param-control space-y-2';

        const header = document.createElement('div');
        header.className = 'flex justify-between items-center mb-1';
        header.innerHTML = `<label for="${key}" class="text-sm">${config.label}</label>`;

        if (currentMode === 'single') {
            const valueInput = document.createElement('input');
            valueInput.type = 'number';
            valueInput.id = `${key}_value`;
            valueInput.className = 'w-20 text-center text-sm font-mono bg-gray-700 px-2 py-1 rounded';
            valueInput.min = config.min;
            valueInput.max = config.max;
            valueInput.step = config.step;
            header.appendChild(valueInput);
            
            const slider = document.createElement('input');
            slider.type = 'range';
            slider.id = key;
            slider.min = config.min;
            slider.max = config.max;
            slider.step = config.step;
            slider.className = 'w-full';
            
            slider.addEventListener('input', () => valueInput.value = slider.value);
            valueInput.addEventListener('change', () => slider.value = valueInput.value);
            
            wrapper.append(header, slider);
        } else {
            const inputGroup = document.createElement('div');
            inputGroup.className = 'flex gap-2';
            
            const valueInput = document.createElement('input');
            valueInput.type = 'number';
            valueInput.id = `${key}_exp_value`;
            valueInput.className = 'flex-grow text-sm font-mono bg-gray-700 px-2 py-1 rounded';
            valueInput.step = config.step;
            valueInput.placeholder = `ex: ${config.default}`;
            
            const addButton = document.createElement('button');
            addButton.textContent = 'Ajouter';
            addButton.className = 'text-xs bg-blue-600 hover:bg-blue-700 text-white font-semibold py-1 px-3 rounded-md transition-colors';
            addButton.onclick = () => addExperimentValue(key, valueInput);

            inputGroup.append(valueInput, addButton);
            
            const tagsContainer = document.createElement('div');
            tagsContainer.id = `tags_${key}`;
            tagsContainer.className = 'flex flex-wrap -m-1 pt-2';

            wrapper.append(header, inputGroup, tagsContainer);
        }
        return wrapper;
    }
    
    /**
     * Ajoute une valeur à la liste d'un paramètre pour le mode expérimental.
     * @param {string} key - La clé du paramètre.
     * @param {HTMLInputElement} inputElement - L'élément input contenant la valeur.
     */
    function addExperimentValue(key, inputElement) {
        const value = parseFloat(inputElement.value);
        if (isNaN(value)) return;
        
        if (!experimentParams[key]) {
            experimentParams[key] = [];
        }
        if (!experimentParams[key].includes(value)) {
            experimentParams[key].push(value);
            experimentParams[key].sort((a, b) => a - b);
        }
        
        inputElement.value = '';
        renderExperimentTags(key);
    }

    /**
     * Affiche les "tags" pour les valeurs d'un paramètre en mode expérimental.
     * @param {string} key - La clé du paramètre.
     */
    function renderExperimentTags(key) {
        const container = document.getElementById(`tags_${key}`);
        container.innerHTML = '';
        (experimentParams[key] || []).forEach(value => {
            const tag = document.createElement('div');
            tag.className = 'param-tag';
            tag.innerHTML = `
                ${value}
                <span class="param-tag-remove" data-key="${key}" data-value="${value}">&times;</span>
            `;
            container.appendChild(tag);
        });
        
        container.querySelectorAll('.param-tag-remove').forEach(btn => {
            btn.onclick = (e) => {
                const { key, value } = e.currentTarget.dataset;
                experimentParams[key] = experimentParams[key].filter(v => v != value);
                renderExperimentTags(key);
            };
        });
    }

    /**
     * Met à jour les contrôles de l'UI à partir d'un objet de configuration.
     * @param {object} config - L'objet de configuration.
     */
    function updateControlsFromConfig(config) {
        Object.entries(PARAM_CONFIG).forEach(([groupKey, group]) => {
            Object.entries(group.params).forEach(([paramKey, paramConfig]) => {
                const path = (groupKey === 'guided_filter') 
                    ? `refinement.guided_filter.${paramKey.replace('gf_', '')}` 
                    : `algorithm.${paramKey}`;
                
                const value = path.split('.').reduce((obj, key) => obj && obj[key], config);

                if (value !== undefined) {
                    if (currentMode === 'single') {
                        document.getElementById(paramKey).value = value;
                        document.getElementById(`${paramKey}_value`).value = value;
                    } else {
                        const input = document.getElementById(`${paramKey}_exp_value`);
                        if (input) input.placeholder = `ex: ${value}`;
                        
                        if (!experimentParams[paramKey] || experimentParams[paramKey].length === 0) {
                           experimentParams[paramKey] = [value];
                           renderExperimentTags(paramKey);
                        }
                    }
                }
            });
        });
    }


    // --- Logique de traitement et communication API ---

    /**
     * Démarre le traitement en contactant le backend.
     */
    async function startProcessing() {
        if (!imageFile) {
            addLog("Erreur: Aucune image n'est chargée.", 'error');
            return;
        }

        dom.processButton.disabled = true;
        dom.processButton.textContent = 'Traitement en cours...';
        dom.logConsole.innerHTML = '';
        dom.experimentGrid.innerHTML = '';
        dom.resultsCount.textContent = '0';
        dom.totalRuns.textContent = '0';
        selectedForComparison = [];
        updateComparisonUI();

        addLog('Initialisation du traitement...');

        const formData = new FormData();
        formData.append('image', imageFile);
        
        let endpoint = '';
        
        if (currentMode === 'single') {
            endpoint = '/process-image/';
            Object.values(PARAM_CONFIG).forEach(group => {
                Object.keys(group.params).forEach(key => {
                    formData.append(key, document.getElementById(key).value);
                });
            });
        } else {
            endpoint = '/process-experiment/';
            for (const group of Object.values(PARAM_CONFIG)) {
                for (const key in group.params) {
                    if (!experimentParams[key] || experimentParams[key].length === 0) {
                        addLog(`Erreur: Veuillez ajouter au moins une valeur pour "${group.params[key].label}".`, 'error');
                        dom.processButton.disabled = false;
                        dom.processButton.textContent = 'Lancer le Traitement';
                        return;
                    }
                }
            }
            formData.append('parameter_grid', JSON.stringify(experimentParams));
        }

        try {
            const response = await fetch(endpoint, { method: 'POST', body: formData });
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Erreur du serveur.');
            }
            const data = await response.json();
            connectToLogStream(data.job_id);
        } catch (error) {
            addLog(`Erreur lors du lancement: ${error.message}`, 'error');
            dom.processButton.disabled = false;
            dom.processButton.textContent = 'Lancer le Traitement';
        }
    }
    
    /**
     * Se connecte au flux SSE pour recevoir les logs et résultats.
     * @param {string} jobId - L'ID de la tâche de traitement.
     */
    function connectToLogStream(jobId) {
        addLog(`Tâche démarrée avec l'ID: <span class="text-yellow-400">${jobId}</span>`, 'success');
        if(eventSource) eventSource.close();
        
        eventSource = new EventSource(`/stream-logs/${jobId}`);
        
        eventSource.onmessage = (event) => {
            const cleanData = event.data.replace(/'/g, '"');
            const data = JSON.parse(cleanData);
            
            handleSSEMessage(data);
        };
        
        eventSource.onerror = () => {
            addLog('Connexion au serveur perdue.', 'error');
            dom.processButton.disabled = false;
            dom.processButton.textContent = 'Relancer le Traitement';
            if(eventSource) eventSource.close();
        };
    }

    /**
     * Traite un message reçu via SSE et met à jour l'UI.
     * @param {object} data - L'objet de données parsé du message SSE.
     */
    function handleSSEMessage(data) {
        switch (data.type) {
            case 'log':
                addLog(data.message);
                break;
            case 'result_intermediate':
                setIntermediateImage(data.name, data.image);
                break;
            case 'experiment_start':
                dom.totalRuns.textContent = data.total_runs;
                break;
            case 'run_result':
                createExperimentResultCard(data);
                dom.resultsCount.textContent = dom.experimentGrid.children.length;
                break;
            case 'done':
            case 'experiment_done':
                addLog(data.message, 'success');
                dom.processButton.disabled = false;
                dom.processButton.textContent = 'Lancer un nouveau traitement';
                if(eventSource) eventSource.close();
                break;
            case 'error':
                addLog(`Erreur: ${data.message}`, 'error');
                dom.processButton.disabled = false;
                dom.processButton.textContent = 'Réessayer le traitement';
                if(eventSource) eventSource.close();
                break;
        }
    }

    /**
     * Affiche une image intermédiaire dans la vue "Analyse Unique".
     * @param {string} name - Le nom de l'image (ex: 'dark_channel').
     * @param {string} base64Data - L'image encodée en base64.
     */
    function setIntermediateImage(name, base64Data) {
        const imgElement = document.getElementById(`img-vis-${name}`);
        if (imgElement) {
            imgElement.src = base64Data;
            imgElement.classList.remove('hidden');
        }
         if (name === 'final_result') {
            dom.resultImageComp.src = base64Data;
            dom.resultWrapper.style.width = '50%';
            dom.comparisonSlider.style.left = '50%';
        }
    }

    /**
     * Crée une carte de résultat pour le mode expérimental.
     * @param {object} data - Les données du résultat (image, paramètres, index).
     */
    function createExperimentResultCard(data) {
        const card = document.createElement('div');
        card.className = 'result-card bg-gray-700/50 p-4 rounded-lg shadow-md flex flex-col gap-3 cursor-pointer transition-all duration-200 border-2 border-transparent';
        card.dataset.runId = data.run_index;
        card.dataset.imageData = data.image;
        card.dataset.paramsData = JSON.stringify(data.params);
        
        const paramList = Object.entries(data.params)
            .map(([key, value]) => `<li><span class="font-semibold text-gray-300">${key.replace('_', ' ')}:</span> <span class="font-mono text-blue-400">${value}</span></li>`)
            .join('');

        card.innerHTML = `
            <img src="${data.image}" class="w-full h-auto object-contain rounded-md bg-black pointer-events-none">
            <ul class="text-xs space-y-1 text-gray-400 mt-2 pointer-events-none">
                ${paramList}
            </ul>
        `;
        
        card.addEventListener('click', handleComparisonSelection);
        dom.experimentGrid.appendChild(card);
    }
    
    // --- Logique de Comparaison (Mode Expérimental) ---

    /**
     * Gère la sélection/désélection d'une carte pour la comparaison.
     * @param {MouseEvent} event - L'événement de clic.
     */
    function handleComparisonSelection(event) {
        const card = event.currentTarget;
        const runId = card.dataset.runId;

        const isSelected = selectedForComparison.some(item => item.id === runId);

        if (isSelected) {
            selectedForComparison = selectedForComparison.filter(item => item.id !== runId);
        } else {
            if (selectedForComparison.length >= 2) {
                selectedForComparison.shift();
            }
            selectedForComparison.push({
                id: runId,
                image: card.dataset.imageData,
                params: JSON.parse(card.dataset.paramsData)
            });
        }
        updateComparisonUI();
    }

    /**
     * Met à jour toute l'interface de comparaison en fonction de l'état `selectedForComparison`.
     */
    function updateComparisonUI() {
        const selectedIds = selectedForComparison.map(item => item.id);
        document.querySelectorAll('.result-card').forEach(card => {
            if (selectedIds.includes(card.dataset.runId)) {
                card.classList.add('selected');
            } else {
                card.classList.remove('selected');
            }
        });

        if (selectedForComparison.length === 2) {
            const [itemA, itemB] = selectedForComparison;
            
            dom.compImageA.src = itemA.image;
            dom.compParamsA.innerHTML = formatParamsToList(itemA.params);
            dom.compImageB.src = itemB.image;
            dom.compParamsB.innerHTML = formatParamsToList(itemB.params);

            dom.expCompImgA.src = itemA.image;
            dom.expCompImgB.src = itemB.image;

            dom.expCompWrapperB.style.width = `50%`;
            dom.expComparisonSlider.style.left = `50%`;

            dom.expComparisonPanel.classList.remove('hidden');
        } else {
            dom.expComparisonPanel.classList.add('hidden');
        }
    }
    
    /**
     * Formate un objet de paramètres en une chaîne de list items HTML.
     * @param {object} params - L'objet de paramètres.
     * @returns {string} La chaîne HTML.
     */
    function formatParamsToList(params) {
        return Object.entries(params)
            .map(([key, value]) => `<li><span class="font-semibold text-gray-300">${key.replace('_', ' ')}:</span> <span class="font-mono text-blue-400">${value}</span></li>`)
            .join('');
    }

    /**
     * Réinitialise la sélection de comparaison.
     */
    function clearComparison() {
        selectedForComparison = [];
        updateComparisonUI();
    }

    // --- Initialisation et Écouteurs d'événements ---

    /**
     * Initialise l'application au chargement de la page.
     */
    function initialize() {
        resetUI();
        setMode('single');

        fetch('/default-config')
            .then(res => res.json())
            .then(config => {
                 defaultConfig = config;
                 renderParams();
            }).catch(e => {
                console.error("Impossible de charger la config par défaut", e);
                addLog("Impossible de charger la config par défaut.", "error");
            });
        
        const visNames = {
            'dark_channel': 'Dark Channel',
            'initial_transmission': 'Transmission Initiale',
            'refined_transmission': 'Transmission Affinée',
            'final_result': 'Résultat Final',
        };
        dom.intermediateResults.innerHTML = Object.entries(visNames).map(([key, label]) => `
            <div class="space-y-2">
                <div class="aspect-square bg-gray-700 rounded-lg flex items-center justify-center">
                    <img id="img-vis-${key}" class="w-full h-full object-contain rounded-lg hidden">
                </div>
                <p class="text-sm text-gray-400">${label}</p>
            </div>
        `).join('');
    }

    dom.imageUpload.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            imageFile = e.target.files[0];
            const reader = new FileReader();
            reader.onload = (event) => {
                const imageUrl = event.target.result;
                dom.originalImageComp.src = imageUrl;
                dom.originalImageExp.src = imageUrl;

                resetUI();
                dom.placeholder.classList.add('hidden');
                dom.logContainer.classList.remove('hidden');
                dom.paramsContainer.classList.remove('opacity-50', 'pointer-events-none');
                dom.modeSelectionContainer.classList.remove('opacity-50', 'pointer-events-none');
                dom.processButton.disabled = false;
                
                if (currentMode === 'single') {
                    dom.singleRunView.classList.remove('hidden');
                } else {
                    dom.experimentView.classList.remove('hidden');
                }
            };
            reader.readAsDataURL(imageFile);
            addLog('Image chargée. Prêt à traiter.', 'success');
        }
    });

    dom.modeSingleBtn.addEventListener('click', () => {
        if (currentMode === 'experiment') {
            setMode('single');
            if(imageFile) dom.singleRunView.classList.remove('hidden');
            dom.experimentView.classList.add('hidden');
        }
    });

    dom.modeExperimentBtn.addEventListener('click', () => {
        if (currentMode === 'single') {
            setMode('experiment');
            dom.singleRunView.classList.add('hidden');
            if(imageFile) dom.experimentView.classList.remove('hidden');
        }
    });
    
    dom.defaultButton.addEventListener('click', () => {
        if (defaultConfig) {
            experimentParams = {};
            renderParams();
            addLog("Paramètres réinitialisés aux valeurs par défaut.", "info");
        } else {
            addLog("Erreur: Configuration par défaut non chargée.", "error");
        }
    });

    dom.processButton.addEventListener('click', startProcessing);
    dom.clearComparisonBtn.addEventListener('click', clearComparison);

    // --- Gestionnaires des sliders de comparaison ---
    function setupComparisonSlider(container, slider, wrapper) {
        let isDragging = false;
        slider.addEventListener('mousedown', () => { isDragging = true; });
        document.addEventListener('mouseup', () => { isDragging = false; });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const rect = container.getBoundingClientRect();
            // S'assure que le mouvement est contraint à l'intérieur du conteneur
            const x = e.clientX - rect.left;
            const offsetX = Math.max(0, Math.min(rect.width, x));
            const percentage = (offsetX / rect.width) * 100;
            wrapper.style.width = `${percentage}%`;
            slider.style.left = `${percentage}%`;
        });
    }

    setupComparisonSlider(dom.comparisonContainer, dom.comparisonSlider, dom.resultWrapper);
    setupComparisonSlider(dom.expComparisonContainer, dom.expComparisonSlider, dom.expCompWrapperB);


    initialize();
});
