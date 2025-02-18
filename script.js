
const dropArea = document.getElementById('file-drop-area');
const uploadBtn = document.getElementById('upload-btn');

// Handle file drop
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = '#e0e0e0';
});

dropArea.addEventListener('dragleave', () => {
    dropArea.style.backgroundColor = '#f0f0f0';
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.style.backgroundColor = '#f0f0f0';
    const files = e.dataTransfer.files;
    handleFiles(files);
});

// Handle file selection via the button
uploadBtn.addEventListener('click', () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv, .xls, .xlsx'; // Specify accepted file types
    input.addEventListener('change', () => {
        const files = input.files;
        handleFiles(files);
    });
    input.click();
});

// Handle the files
function handleFiles(files) {
    if (files.length > 0) {
        alert('File uploaded: ' + files[0].name);
    }
}
