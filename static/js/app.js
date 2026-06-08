/* ================================================
   BananaAI — App Logic
   ================================================ */

const $ = (id) => document.getElementById(id);

// Views
const viewUpload = $('view-upload');
const viewResult = $('view-result');

// Upload elements
const dropZone    = $('drop-zone');
const fileInput   = $('file-input');
const dropIdle    = $('drop-idle');
const dropPreview = $('drop-preview');
const previewImg  = $('preview-img');
const btnBrowse   = $('btn-browse');
const btnChange   = $('btn-change');
const btnAnalyze  = $('btn-analyze');
const btnSpinner  = $('btn-spinner');
const btnArrow    = $('btn-arrow');

// Result elements
const resultImg     = $('result-img');
const resultImgBadge= $('result-img-badge');
const resultEmoji   = $('result-emoji');
const resultKlass   = $('result-klass');
const resultDesc    = $('result-desc');
const confPct       = $('conf-pct');
const confFill      = $('conf-fill');
const probaBlock    = $('proba-block');
const recText       = $('rec-text');
const btnBack       = $('btn-back');

let selectedFile = null;

// --- Category meta ---
const CATS = {
    matang:       { label:'Matang',        emoji:'🍌', color:'#eab308', desc:'Pisang dalam kondisi matang sempurna dan siap dikonsumsi.', rec:'Waktu terbaik untuk dimakan! Nikmati pisang segar ini.' },
    mentah:       { label:'Mentah',        emoji:'🍃', color:'#15803d', desc:'Pisang masih mentah dan belum siap untuk dikonsumsi.', rec:'Tunggu beberapa hari lagi agar pisang matang sempurna.' },
    busuk:        { label:'Busuk',         emoji:'🍂', color:'#854d0e', desc:'Pisang ini sudah membusuk dan tidak layak untuk dikonsumsi.', rec:'Segera buang pisang ini. Jangan dikonsumsi!' }
};

const LABEL_ID = {
    matang:'Matang', mentah:'Mentah', busuk:'Busuk'
};


// ================================================
// EVENTS
// ================================================

btnBrowse.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener('click', () => { if (!selectedFile) fileInput.click(); });
fileInput.addEventListener('change', (e) => e.target.files[0] && setFile(e.target.files[0]));
btnChange.addEventListener('click', (e) => { e.stopPropagation(); fileInput.click(); });
btnAnalyze.addEventListener('click', doAnalyze);
btnBack.addEventListener('click', reset);

// Drag & Drop
dropZone.addEventListener('dragenter', (e) => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', (e) => { if (!dropZone.contains(e.relatedTarget)) dropZone.classList.remove('over'); });
dropZone.addEventListener('dragover',  (e) => e.preventDefault());
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) setFile(f);
});

// Keyboard
document.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && selectedFile && !btnAnalyze.disabled) doAnalyze();
    if (e.key === 'Escape' && viewResult.style.display !== 'none') reset();
});


// ================================================
// FILE HANDLING
// ================================================

function setFile(file) {
    const valid = ['image/jpeg','image/jpg','image/png','image/webp'];
    if (!valid.includes(file.type)) return toast('Format tidak didukung. Gunakan JPG, PNG, atau WEBP.');
    if (file.size > 16 * 1024 * 1024) return toast('File terlalu besar. Maksimum 16 MB.');

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = (e) => {
        previewImg.src = e.target.result;
        dropIdle.style.display = 'none';
        dropPreview.style.display = 'block';
        btnAnalyze.disabled = false;
        viewResult.style.display = 'none';
    };
    reader.readAsDataURL(file);
}


// ================================================
// ANALYZE
// ================================================

async function doAnalyze() {
    if (!selectedFile) return;

    // Loading state
    btnAnalyze.disabled = true;
    btnSpinner.style.display = 'flex';
    btnArrow.style.display = 'none';

    const fd = new FormData();
    fd.append('file', selectedFile);

    try {
        const res = await fetch('/predict', { method: 'POST', body: fd });
        const data = await res.json();

        if (data.success) {
            showResult(data);
        } else {
            toast(data.error || 'Gagal memproses gambar.');
        }
    } catch {
        toast('Tidak dapat terhubung ke server. Pastikan server berjalan.');
    } finally {
        btnAnalyze.disabled = false;
        btnSpinner.style.display = 'none';
        btnArrow.style.display = '';
    }
}


// ================================================
// SHOW RESULT
// ================================================

function showResult({ result, image_url }) {
    const { predicted_class, confidence, confidence_scores } = result;
    const cat = CATS[predicted_class] || {};

    // Image
    resultImg.src = image_url;
    resultImg.alt = cat.label || predicted_class;

    // Badge
    const col = cat.color || '#E8B84B';
    resultImgBadge.textContent = cat.label || predicted_class;
    resultImgBadge.style.background = '#ffffff';
    resultImgBadge.style.color = col;
    resultImgBadge.style.border = `2px solid ${col}`;
    resultImgBadge.style.boxShadow = `0 4px 16px ${hexAlpha(col, 0.3)}`;

    // Hero
    resultEmoji.textContent = '';
    resultKlass.textContent  = cat.label || predicted_class;
    resultKlass.style.color  = col;
    
    const heroBox = document.querySelector('.hero-emoji-box');
    heroBox.style.background = col;
    heroBox.style.borderColor = col;
    heroBox.style.boxShadow = `0 8px 24px ${hexAlpha(col, 0.3)}`;

    // Description
    resultDesc.textContent = cat.desc || '';

    // Confidence
    const pct = confidence.toFixed(1);
    confPct.textContent = `${pct}%`;
    confPct.style.color = col;

    // Color conf bar by level
    confFill.style.background =
        confidence >= 80 ? '#5DB35D' :
        confidence >= 60 ? '#E8B84B' : '#C0832A';

    // Animate conf bar
    confFill.style.width = '0%';
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            confFill.style.width = `${confidence}%`;
        });
    });

    // Probabilities
    probaBlock.innerHTML = '';
    const sorted = Object.entries(confidence_scores).sort(([,a],[,b]) => b - a);

    sorted.forEach(([cls, score]) => {
        const isWinner = cls === predicted_class;
        const name = LABEL_ID[cls] || cls;

        const row = document.createElement('div');
        row.className = 'proba-row';
        row.innerHTML = `
            <span class="proba-name ${isWinner ? 'is-winner' : ''}">${name}</span>
            <div class="proba-track">
                <div class="proba-fill ${isWinner ? 'is-winner' : ''}" data-w="${score}"></div>
            </div>
            <span class="proba-pct ${isWinner ? 'is-winner' : ''}">${score.toFixed(1)}%</span>
        `;
        probaBlock.appendChild(row);
    });

    // Animate proba bars
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            probaBlock.querySelectorAll('.proba-fill').forEach(el => {
                el.style.width = el.dataset.w + '%';
            });
        });
    });

    // Recommendation
    recText.textContent = cat.rec || '';

    // Switch view
    viewUpload.style.display = 'none';
    viewResult.style.display = 'flex';
    viewResult.style.animation = 'none';
    requestAnimationFrame(() => {
        viewResult.style.animation = '';
    });
}


// ================================================
// RESET
// ================================================

function reset() {
    selectedFile = null;
    fileInput.value = '';
    previewImg.src = '';

    dropIdle.style.display = 'flex';
    dropPreview.style.display = 'none';
    btnAnalyze.disabled = true;
    confFill.style.width = '0%';

    viewResult.style.display = 'none';
    viewUpload.style.display = 'flex';
    viewUpload.style.animation = 'none';
    requestAnimationFrame(() => { viewUpload.style.animation = ''; });
}


// ================================================
// TOAST
// ================================================

let toastTimer;
function toast(msg) {
    const el = $('toast');
    $('toast-msg').textContent = msg;
    el.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => el.classList.remove('show'), 3000);
}


// ================================================
// UTILS
// ================================================

function hexAlpha(hex, a) {
    if (!hex?.startsWith('#')) return `rgba(232,184,75,${a})`;
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${r},${g},${b},${a})`;
}
