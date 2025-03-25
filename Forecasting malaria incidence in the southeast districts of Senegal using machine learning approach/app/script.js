document.addEventListener("DOMContentLoaded", function () {
    const dateSelect = document.querySelector("#date-select");
    const mapContainer = document.querySelector("#map-container");

    if (dateSelect) {
        dateSelect.addEventListener("change", function () {
            if (dateSelect.value) {
                // Affiche la carte avec une transition fluide
                mapContainer.classList.remove("hidden");
                mapContainer.classList.add("visible");
            } else {
                // Masque la carte avec une transition fluide
                mapContainer.classList.remove("visible");
                mapContainer.classList.add("hidden");
            }
        });
    }
});

document.addEventListener("DOMContentLoaded", function () {
    const visualizeButton = document.querySelector("#visualize_button");

    if (visualizeButton) {
        visualizeButton.addEventListener("click", function () {
            visualizeButton.style.backgroundColor = "#1b9bff"; // Changer la couleur du bouton
            visualizeButton.style.color = "white"; // Optionnel, pour changer la couleur du texte du bouton
        });
    }
});

