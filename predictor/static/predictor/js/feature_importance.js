/* ==========================================================================
   Objective 4 Feature Importance Scaler
   Path: static/predictor/js/feature_importance.js
   ========================================================================== */

document.addEventListener("DOMContentLoaded", function() {
    document.querySelectorAll(".chart-section").forEach(function(section) {
        var barRows = section.querySelectorAll(".bar-row");
        var maxVal = 0;

        // Step 1: Find the maximum feature importance value
        barRows.forEach(function(row) {
            var valElem = row.querySelector(".bar-value");
            if (valElem) {
                var val = parseFloat(valElem.textContent.trim()) || 0;
                if (val > maxVal) maxVal = val;
            }
        });

        // Step 2: Scale each bar so the highest feature fills 100% card width
        if (maxVal > 0) {
            barRows.forEach(function(row) {
                var valElem = row.querySelector(".bar-value");
                var fillElem = row.querySelector(".bar-fill");
                if (valElem && fillElem) {
                    var val = parseFloat(valElem.textContent.trim()) || 0;
                    var pct = (val / maxVal) * 100;
                    fillElem.style.width = pct + "%";
                }
            });
        }
    });
});