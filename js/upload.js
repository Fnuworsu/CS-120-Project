document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');
    const leftEyeInput = document.getElementById('leftEye');
    const rightEyeInput = document.getElementById('rightEye');
    const leftEyeFilename = document.getElementById('leftEyeFilename');
    const rightEyeFilename = document.getElementById('rightEyeFilename');
    const leftLens = document.getElementById('leftLens');
    const rightLens = document.getElementById('rightLens');

    // Handle file selection and filename display
    function handleFileSelect(input, filenameDisplay, lens) {
        return (e) => {
            const file = e.target.files[0];
            if (file) {
                // Show filename
                filenameDisplay.textContent = file.name;
                filenameDisplay.classList.remove('hidden');

                // Highlight lens
                lens.classList.remove('fill-gray-200');
                lens.classList.add('fill-blue-400');
            }
        };
    }

    // Handle drag and drop
    function handleDragDrop(dropZone, input) {
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-400');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-400');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-400');
            input.files = e.dataTransfer.files;
            input.dispatchEvent(new Event('change'));
        });
    }

    // Set up event listeners for file inputs
    leftEyeInput.addEventListener('change', handleFileSelect(leftEyeInput, leftEyeFilename, leftLens));
    rightEyeInput.addEventListener('change', handleFileSelect(rightEyeInput, rightEyeFilename, rightLens));

    // Set up click handlers for the upload buttons
    document.querySelectorAll('.upload-button').forEach((button, index) => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            if (index === 0) {
                leftEyeInput.click();
            } else {
                rightEyeInput.click();
            }
        });
    });

    // Set up drag and drop
    const dropZones = document.querySelectorAll('.border-dashed');
    dropZones.forEach((zone, index) => {
        handleDragDrop(zone, index === 0 ? leftEyeInput : rightEyeInput);
    });

    // Form submission
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        const leftEyeFile = leftEyeInput.files[0];
        const rightEyeFile = rightEyeInput.files[0];

        if (!leftEyeFile || !rightEyeFile) {
            alert('Please upload both eye scans');
            return;
        }

        formData.append('leftEye', leftEyeFile);
        formData.append('rightEye', rightEyeFile);

        try {
            // TODO: Replace with actual Flask endpoint URL
            const response = await fetch('/api/process-retina', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            
            // Save case ID and redirect to results page
            window.location.href = `result.html?caseId=${data.caseId}`;
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while processing the scans. Please try again.');
        }
    });
}); 