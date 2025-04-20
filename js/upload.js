document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('uploadForm');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        const formData = new FormData();
        const leftEyeFile = document.getElementById('leftEye').files[0];
        const rightEyeFile = document.getElementById('rightEye').files[0];

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