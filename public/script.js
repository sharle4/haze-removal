/**
 * Script principal pour l'interface de démonstration de l'algorithme "Dark Channel Prior".
 * Gère les interactions utilisateur, la communication avec l'API, et l'affichage des résultats,
 * y compris une fonctionnalité avancée de comparaison en mode expérimental.
 * @author Charles
 * @version 2.2.0
 */
document.addEventListener('DOMContentLoaded', () => {
    // --- État global de l'application ---
    let imageFile = null;
    let eventSource = null;
    let defaultConfig = null;
    let currentMode = 'single'; // 'single' ou 'experiment'
    let experimentParams = {};
    /** @type {Array<object>} Contient les données des cartes sélectionnées pour la comparaison. */
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
        singleFullscreenBtn: document.getElementById('single-fullscreen-btn'),

        // Vue "Expérimentale"
        experimentGrid: document.getElementById('experiment-grid'),
        originalImageExp: document.getElementById('original-image-exp'),
        resultsCount: document.getElementById('results-count'),
        totalRuns: document.getElementById('total-runs'),
        
        // Panneaux de visionnage/comparaison (Mode Expérimental)
        expViewerPanel: document.getElementById('experiment-viewer-panel'),
        viewerContent: document.getElementById('viewer-content'),
        viewerImage: document.getElementById('viewer-image'),
        viewerParams: document.getElementById('viewer-params'),
        clearViewerBtn: document.getElementById('clear-viewer-btn'),

        expComparisonPanel: document.getElementById('experiment-comparison-panel'),
        comparatorContent: document.getElementById('comparator-content'),
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
        
        // Plein écran
        fullscreenModal: document.getElementById('fullscreen-modal'),
        fullscreenContent: document.getElementById('fullscreen-content'),
        fullscreenCloseBtn: document.getElementById('fullscreen-close-btn'),
        viewerFullscreenBtn: document.getElementById('viewer-fullscreen-btn'),
        comparatorFullscreenBtn: document.getElementById('comparator-fullscreen-btn'),

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

    function addLog(message, type = 'info') {
        if (dom.logConsole.querySelector('.text-gray-500')) dom.logConsole.innerHTML = '';
        const p = document.createElement('p');
        p.innerHTML = `[<span class="text-gray-500">${new Date().toLocaleTimeString()}</span>] ${message}`;
        const typeClasses = { error: 'text-red-400', success: 'text-blue-400', info: 'text-green-400' };
        p.className = typeClasses[type] || 'text-green-400';
        dom.logConsole.appendChild(p);
        dom.logConsole.scrollTop = dom.logConsole.scrollHeight;
    }
    
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
        clearComparison();
        if (eventSource) eventSource.close();
    }

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

    function renderParams() {
        Object.values(PARAM_CONFIG).forEach(group => {
            group.container.innerHTML = '';
            Object.entries(group.params).forEach(([key, config]) => {
                group.container.appendChild(createParamControl(key, config));
            });
        });
        if(defaultConfig) updateControlsFromConfig(defaultConfig);
    }

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
            valueInput.min = config.min; valueInput.max = config.max; valueInput.step = config.step;
            header.appendChild(valueInput);
            const slider = document.createElement('input');
            slider.type = 'range'; slider.id = key; slider.min = config.min; slider.max = config.max; slider.step = config.step;
            slider.className = 'w-full';
            slider.addEventListener('input', () => valueInput.value = slider.value);
            valueInput.addEventListener('change', () => slider.value = valueInput.value);
            wrapper.append(header, slider);
        } else {
            const inputGroup = document.createElement('div');
            inputGroup.className = 'flex gap-2';
            const valueInput = document.createElement('input');
            valueInput.type = 'number'; valueInput.id = `${key}_exp_value`;
            valueInput.className = 'flex-grow text-sm font-mono bg-gray-700 px-2 py-1 rounded';
            valueInput.step = config.step; valueInput.placeholder = `ex: ${config.default}`;
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
    
    function addExperimentValue(key, inputElement) {
        const value = parseFloat(inputElement.value);
        if (isNaN(value)) return;
        if (!experimentParams[key]) experimentParams[key] = [];
        if (!experimentParams[key].includes(value)) {
            experimentParams[key].push(value);
            experimentParams[key].sort((a, b) => a - b);
        }
        inputElement.value = '';
        renderExperimentTags(key);
    }

    function renderExperimentTags(key) {
        const container = document.getElementById(`tags_${key}`);
        container.innerHTML = '';
        (experimentParams[key] || []).forEach(value => {
            const tag = document.createElement('div');
            tag.className = 'param-tag';
            tag.innerHTML = `${value}<span class="param-tag-remove" data-key="${key}" data-value="${value}">&times;</span>`;
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

    function updateControlsFromConfig(config) {
        Object.entries(PARAM_CONFIG).forEach(([groupKey, group]) => {
            Object.entries(group.params).forEach(([paramKey, paramConfig]) => {
                const path = (groupKey === 'guided_filter') ? `refinement.guided_filter.${paramKey.replace('gf_', '')}` : `algorithm.${paramKey}`;
                const value = path.split('.').reduce((obj, key) => obj && obj[key], config);
                if (value === undefined) return;
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
            });
        });
    }

    // --- Logique de traitement et communication API ---

    async function startProcessing() {
        if (!imageFile) { addLog("Erreur: Aucune image n'est chargée.", 'error'); return; }
        dom.processButton.disabled = true;
        dom.processButton.textContent = 'Traitement en cours...';
        dom.logConsole.innerHTML = '';
        dom.experimentGrid.innerHTML = '';
        dom.resultsCount.textContent = '0';
        dom.totalRuns.textContent = '0';
        clearComparison();
        addLog('Initialisation du traitement...');
        const formData = new FormData();
        formData.append('image', imageFile);
        let endpoint = '';
        if (currentMode === 'single') {
            endpoint = '/process-image/';
            Object.values(PARAM_CONFIG).forEach(g => Object.keys(g.params).forEach(k => formData.append(k, document.getElementById(k).value)));
        } else {
            endpoint = '/process-experiment/';
            for (const group of Object.values(PARAM_CONFIG)) {
                for (const key in group.params) {
                    if (!experimentParams[key] || experimentParams[key].length === 0) {
                        addLog(`Erreur: Ajoutez au moins une valeur pour "${group.params[key].label}".`, 'error');
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
            if (!response.ok) { const error = await response.json(); throw new Error(error.detail || 'Erreur du serveur.'); }
            const data = await response.json();
            connectToLogStream(data.job_id);
        } catch (error) {
            addLog(`Erreur lors du lancement: ${error.message}`, 'error');
            dom.processButton.disabled = false;
            dom.processButton.textContent = 'Lancer le Traitement';
        }
    }
    
    function connectToLogStream(jobId) {
        addLog(`Tâche démarrée avec l'ID: <span class="text-yellow-400">${jobId}</span>`, 'success');
        if(eventSource) eventSource.close();
        eventSource = new EventSource(`/stream-logs/${jobId}`);
        eventSource.onmessage = (event) => handleSSEMessage(JSON.parse(event.data.replace(/'/g, '"')));
        eventSource.onerror = () => {
            addLog('Connexion au serveur perdue.', 'error');
            dom.processButton.disabled = false;
            dom.processButton.textContent = 'Relancer le Traitement';
            if(eventSource) eventSource.close();
        };
    }

    function handleSSEMessage(data) {
        switch (data.type) {
            case 'log': addLog(data.message); break;
            case 'result_intermediate': setIntermediateImage(data.name, data.image); break;
            case 'experiment_start': dom.totalRuns.textContent = data.total_runs; break;
            case 'run_result':
                createExperimentResultCard(data);
                dom.resultsCount.textContent = dom.experimentGrid.children.length;
                break;
            case 'done': case 'experiment_done':
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

    function setIntermediateImage(name, base64Data) {
        const imgElement = document.getElementById(`img-vis-${name}`);
        if (imgElement) { imgElement.src = base64Data; imgElement.classList.remove('hidden'); }
        if (name === 'final_result') {
            dom.resultImageComp.src = base64Data;
            dom.resultWrapper.style.width = '50%';
            dom.comparisonSlider.style.left = '50%';
        }
    }

    function createExperimentResultCard(data) {
        const card = document.createElement('div');
        card.className = 'result-card bg-gray-700/50 p-4 rounded-lg shadow-md flex flex-col gap-3 cursor-pointer transition-all duration-200 border-2 border-transparent';
        card.dataset.runId = data.run_index;
        card.dataset.imageData = data.image;
        card.dataset.paramsData = JSON.stringify(data.params);
        const paramList = Object.entries(data.params).map(([key, value]) => `<li><span class="font-semibold text-gray-300">${key.replace('_', ' ')}:</span> <span class="font-mono text-blue-400">${value}</span></li>`).join('');
        card.innerHTML = `<img src="${data.image}" class="w-full h-auto object-contain rounded-md bg-black pointer-events-none"><ul class="text-xs space-y-1 text-gray-400 mt-2 pointer-events-none">${paramList}</ul>`;
        card.addEventListener('click', handleComparisonSelection);
        dom.experimentGrid.appendChild(card);
    }
    
    // --- Logique de Comparaison & Visionnage (Mode Expérimental) ---

    function handleComparisonSelection(event) {
        const card = event.currentTarget;
        const runId = card.dataset.runId;
        const isSelected = selectedForComparison.some(item => item.id === runId);
        if (isSelected) {
            selectedForComparison = selectedForComparison.filter(item => item.id !== runId);
        } else {
            if (selectedForComparison.length >= 2) selectedForComparison.shift();
            selectedForComparison.push({
                id: runId,
                image: card.dataset.imageData,
                params: JSON.parse(card.dataset.paramsData)
            });
        }
        updateInspectionUI();
    }

    function updateInspectionUI() {
        const selectedIds = selectedForComparison.map(item => item.id);
        document.querySelectorAll('.result-card').forEach(card => {
            card.classList.toggle('selected', selectedIds.includes(card.dataset.runId));
        });

        const selectionCount = selectedForComparison.length;
        dom.expViewerPanel.classList.toggle('hidden', selectionCount !== 1);
        dom.expComparisonPanel.classList.toggle('hidden', selectionCount !== 2);

        if (selectionCount === 1) {
            const [item] = selectedForComparison;
            dom.viewerImage.src = item.image;
            dom.viewerParams.innerHTML = formatParamsToList(item.params);
        } else if (selectionCount === 2) {
            const [itemA, itemB] = selectedForComparison;
            dom.compImageA.src = itemA.image;
            dom.compParamsA.innerHTML = formatParamsToList(itemA.params);
            dom.compImageB.src = itemB.image;
            dom.compParamsB.innerHTML = formatParamsToList(itemB.params);
            dom.expCompImgA.src = itemA.image;
            dom.expCompImgB.src = itemB.image;
            dom.expCompWrapperB.style.width = `50%`;
            dom.expComparisonSlider.style.left = `50%`;
        }
    }
    
    function formatParamsToList(params) {
        return Object.entries(params).map(([key, value]) => `<li><span class="font-semibold text-gray-300">${key.replace('_', ' ')}:</span> <span class="font-mono text-blue-400">${value}</span></li>`).join('');
    }

    function clearComparison() {
        selectedForComparison = [];
        updateInspectionUI();
    }

    // --- Logique du mode plein écran ---
    function openFullscreen(contentElement) {
        dom.fullscreenContent.innerHTML = ''; // Vider le contenu précédent
        const contentClone = contentElement.cloneNode(true);
        dom.fullscreenContent.appendChild(contentClone);
        dom.fullscreenModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';

        // Ré-attacher les écouteurs du slider sur le contenu cloné
        const clonedComparisonContainer = dom.fullscreenContent.querySelector('.comparison-container');
        if (clonedComparisonContainer) {
            const clonedSlider = clonedComparisonContainer.querySelector('.comparison-slider');
            const clonedResultWrapper = clonedComparisonContainer.querySelector('[id^="result-wrapper"], [id^="exp-comp-wrapper-b"]');
            if (clonedSlider && clonedResultWrapper) {
                setupComparisonSlider(clonedComparisonContainer, clonedSlider, clonedResultWrapper);
            }
        }
    }
    
    function closeFullscreen() {
        dom.fullscreenModal.classList.add('hidden');
        dom.fullscreenContent.innerHTML = '';
        document.body.style.overflow = '';
    }

    // --- Initialisation et Écouteurs d'événements ---

    function initialize() {
        resetUI();
        setMode('single');
        fetch('/default-config').then(res => res.json()).then(config => {
            defaultConfig = config; renderParams();
        }).catch(e => { console.error("Impossible de charger la config par défaut", e); addLog("Impossible de charger la config par défaut.", "error"); });
        
        const visNames = { 'dark_channel': 'Dark Channel', 'initial_transmission': 'Transmission Initiale', 'refined_transmission': 'Transmission Affinée', 'final_result': 'Résultat Final' };
        dom.intermediateResults.innerHTML = Object.entries(visNames).map(([key, label]) => `
            <div class="space-y-2"><div class="aspect-square bg-gray-700 rounded-lg flex items-center justify-center"><img id="img-vis-${key}" class="w-full h-full object-contain rounded-lg hidden"></div><p class="text-sm text-gray-400">${label}</p></div>`).join('');
    }

    dom.imageUpload.addEventListener('change', (e) => {
        if (!e.target.files || !e.target.files[0]) return;
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
            dom.singleRunView.classList.toggle('hidden', currentMode !== 'single');
            dom.experimentView.classList.toggle('hidden', currentMode !== 'experiment');
        };
        reader.readAsDataURL(imageFile);
        addLog('Image chargée. Prêt à traiter.', 'success');
    });

    dom.modeSingleBtn.addEventListener('click', () => { if (currentMode === 'experiment') { setMode('single'); if(imageFile) { dom.singleRunView.classList.remove('hidden'); dom.experimentView.classList.add('hidden'); } } });
    dom.modeExperimentBtn.addEventListener('click', () => { if (currentMode === 'single') { setMode('experiment'); if(imageFile) { dom.singleRunView.classList.add('hidden'); dom.experimentView.classList.remove('hidden'); } } });
    dom.defaultButton.addEventListener('click', () => { if (defaultConfig) { experimentParams = {}; renderParams(); addLog("Paramètres réinitialisés.", "info"); } else { addLog("Erreur: Config par défaut non chargée.", "error"); } });
    dom.processButton.addEventListener('click', startProcessing);
    dom.clearViewerBtn.addEventListener('click', clearComparison);
    dom.clearComparisonBtn.addEventListener('click', clearComparison);

    // Écouteurs pour le plein écran
    dom.singleFullscreenBtn.addEventListener('click', () => openFullscreen(dom.comparisonContainer));
    dom.viewerFullscreenBtn.addEventListener('click', () => openFullscreen(dom.viewerContent));
    dom.comparatorFullscreenBtn.addEventListener('click', () => openFullscreen(dom.comparatorContent));
    dom.fullscreenCloseBtn.addEventListener('click', closeFullscreen);
    dom.fullscreenModal.addEventListener('click', (e) => { if (e.target === dom.fullscreenModal || e.target.id === 'fullscreen-overlay') closeFullscreen(); });
    document.addEventListener('keydown', (e) => { if (e.key === "Escape" && !dom.fullscreenModal.classList.contains('hidden')) closeFullscreen(); });

    function setupComparisonSlider(container, slider, wrapper) {
        let isDragging = false;
        slider.addEventListener('mousedown', (e) => { isDragging = true; e.preventDefault(); });
        document.addEventListener('mouseup', () => { isDragging = false; });
        document.addEventListener('mousemove', (e) => {
            if (!isDragging) return;
            const rect = container.getBoundingClientRect();
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

