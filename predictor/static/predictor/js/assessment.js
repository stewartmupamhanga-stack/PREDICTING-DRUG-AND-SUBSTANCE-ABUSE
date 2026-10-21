// --- predictor/static/predictor/js/bulk_upload_bar.js -------------------
// Assumes the upload form on the page looks like:
//   <form id="bulk-upload-form" action="{% url 'bulk-upload' %}" method="post" enctype="multipart/form-data">
//     {% csrf_token %}
//     <input type="file" name="bulk_file">
//     <button type="submit">Upload</button>
//   </form>
// Adjust the selectors below if your form's id/input name differ.

(function () {
    const form = document.getElementById("bulk-upload-form");
    if (!form) return;

    const uploadUrl = form.getAttribute("action");

    function getCsrfToken() {
        const input = form.querySelector("input[name=csrfmiddlewaretoken]");
        return input ? input.value : "";
    }

    function getBar() {
        let bar = document.getElementById("bulk-upload-bar");
        if (!bar) {
            bar = document.createElement("div");
            bar.id = "bulk-upload-bar";
            document.body.appendChild(bar);
        }
        return bar;
    }

    function hideBar() {
        const bar = document.getElementById("bulk-upload-bar");
        if (bar) bar.classList.remove("visible");
    }

    function renderBar(innerHtml) {
        const bar = getBar();
        bar.innerHTML = innerHtml;
        bar.classList.add("visible");
        const closeBtn = bar.querySelector(".bar-close");
        if (closeBtn) closeBtn.addEventListener("click", hideBar);
    }

    function showMessage(text, type) {
        renderBar(
            `<span class="bar-message ${type}">${text}</span>` +
            `<button type="button" class="bar-close" aria-label="Dismiss">&times;</button>`
        );
        if (type === "success") setTimeout(hideBar, 5000);
    }

    function showChoice(data) {
        renderBar(`
            <span class="bar-message">${data.original_name} — ${data.row_count} row(s) validated.</span>
            <div class="bar-actions">
                <button type="button" class="btn bar-btn" data-action="save_only">Save only</button>
                <button type="button" class="btn btn-primary bar-btn" data-action="predict_save">Predict &amp; Save</button>
            </div>
            <button type="button" class="bar-close" aria-label="Dismiss">&times;</button>
        `);

        document.querySelectorAll("#bulk-upload-bar .bar-btn").forEach((btn) => {
            btn.addEventListener("click", () => submitConfirm(data, btn.dataset.action));
        });
    }

    function submitConfirm(data, action) {
        renderBar(`<span class="bar-message">Working…</span>`);

        const body = new FormData();
        body.append("stage", "confirm");
        body.append("token", data.token);
        body.append("original_name", data.original_name);
        body.append("action", action);

        fetch(uploadUrl, {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
            body,
        })
            .then((r) => r.json())
            .then((result) => {
                if (result.status === "ok") {
                    if (result.redirect_url) {
                        window.location = result.redirect_url;
                    } else {
                        showMessage(result.message, "success");
                    }
                } else {
                    showMessage(result.message, "error");
                }
            })
            .catch(() => showMessage("Something went wrong. Please try again.", "error"));
    }

    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const fileInput = form.querySelector('input[name="bulk_file"]');
        if (!fileInput || !fileInput.files.length) {
            showMessage("Please choose a file to upload.", "error");
            return;
        }

        renderBar(`<span class="bar-message">Validating…</span>`);

        const body = new FormData();
        body.append("stage", "upload");
        body.append("bulk_file", fileInput.files[0]);

        fetch(uploadUrl, {
            method: "POST",
            headers: { "X-CSRFToken": getCsrfToken() },
            body,
        })
            .then((r) => r.json())
            .then((data) => {
                if (data.status === "ok") {
                    showChoice(data);
                } else {
                    showMessage(data.message, "error");
                }
            })
            .catch(() => showMessage("Something went wrong. Please try again.", "error"));
    });
})();