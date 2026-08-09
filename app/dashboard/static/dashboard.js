/*
 * Dashboard refresh.
 *
 * Deliberately tiny and framework-free: it only re-reads the public overview
 * endpoint and writes numbers into existing nodes via textContent, so no
 * untrusted string is ever parsed as HTML.
 */
(function () {
  "use strict";

  var REFRESH_MS = 60000;

  function setMetric(name, value) {
    var node = document.querySelector('[data-metric="' + name + '"]');
    if (node) {
      node.textContent = String(value);
    }
  }

  function refresh() {
    fetch("/api/v1/analytics/overview", {
      headers: { Accept: "application/json" },
      credentials: "same-origin",
    })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("overview request failed: " + response.status);
        }
        return response.json();
      })
      .then(function (data) {
        setMetric("total_articles", data.total_articles);
        setMetric("articles_24h", data.articles_24h);
      })
      .catch(function (error) {
        // A failed refresh must never blank the server-rendered values.
        if (window.console) {
          window.console.warn("dashboard refresh failed", error.message);
        }
      });
  }

  var button = document.getElementById("refresh");
  if (button) {
    button.addEventListener("click", function () {
      window.location.reload();
    });
  }

  window.setInterval(refresh, REFRESH_MS);
})();
