document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    const caseId = urlParams.get('caseId');

    if (!caseId) {
        alert('No case ID provided');
        return;
    }

    try {
        // TODO: Replace with actual Flask endpoint URL
        const response = await fetch(`/api/get-diagnosis?caseId=${caseId}`);
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Update the DOM with the diagnosis results
        document.getElementById('od-optic-disc').textContent = data.od.opticDisc || 'N/A';
        document.getElementById('os-optic-disc').textContent = data.os.opticDisc || 'N/A';
        
        document.getElementById('od-cd-ratio').textContent = data.od.cdRatio || 'N/A';
        document.getElementById('os-cd-ratio').textContent = data.os.cdRatio || 'N/A';
        
        document.getElementById('od-macula').textContent = data.od.macula || 'N/A';
        document.getElementById('os-macula').textContent = data.os.macula || 'N/A';
        
        document.getElementById('od-blood-vessels').textContent = data.od.bloodVessels || 'N/A';
        document.getElementById('os-blood-vessels').textContent = data.os.bloodVessels || 'N/A';
        
        document.getElementById('od-posterior-pole').textContent = data.od.posteriorPole || 'N/A';
        document.getElementById('os-posterior-pole').textContent = data.os.posteriorPole || 'N/A';

    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while fetching the diagnosis results. Please try again.');
    }
}); 