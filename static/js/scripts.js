document.addEventListener("DOMContentLoaded", function () {
    fetch('static/data/data.json')
        .then(response => response.json())
        .then(artPieces => {
            const container = document.getElementById("artCardsContainer");
            if (!container) {
                console.error("Container element not found!");
                return;
            }

            container.innerHTML = ""; // Clear existing content

            artPieces.forEach(art => {
                const card = document.createElement("div");
                card.classList.add("card");

                card.innerHTML = `
                    <img src="${art.imageUrl}" alt="${art.title}">
                    <div class="card-description">
                        <h3>${art.title}</h3>
                        <p>${art.description}</p>
                        <a href="/3d">
                            <button class="button">3D view</button>
                        </a>
                    </div>
                `;

                // Add click event to each card to open the modal
                card.addEventListener('click', () => {
                    openModal(art.imageUrl, art.title, art.description);
                });

                container.appendChild(card);
            });
        })
        .catch(error => console.error("Error loading art data:", error));

    /* Modal-related functions */
    const modal = document.getElementById('popupModal');
    const closeModalBtn = document.getElementById('closeModal');
    const modalImage = document.getElementById('modalImage');
    const modalTitle = document.getElementById('modalTitle');
    const modalDescription = document.getElementById('modalDescription');

    if (!modal || !closeModalBtn || !modalImage || !modalTitle || !modalDescription) {
        console.error("Modal elements not found!");
        return;
    }

    // Function to open the modal and populate it with data
    function openModal(imageSrc, title, description) {
        modalImage.src = imageSrc;
        modalTitle.textContent = title;
        modalDescription.textContent = description;
        modal.style.display = 'flex'; // Show the modal
    }

    // Function to close the modal
    function closeModal() {
        modal.style.display = 'none'; // Hide the modal
    }

    // Event listener to close the modal when the close button is clicked
    closeModalBtn.addEventListener('click', closeModal);

    // Event listener to close the modal if the user clicks outside the modal content
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });
});
