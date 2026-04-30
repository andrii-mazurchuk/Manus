/**
 * Manus shared navigation.
 * Each page calls: renderNav("dashboard") or renderNav("studio")
 */
(function () {
  const PAGES = [
    { id: "dashboard", label: "Dashboard", href: "/static/dashboard.html" },
    { id: "studio",    label: "Studio",    href: "/static/studio.html" },
    { id: "statistics", label: "Statistics", href: null, soon: true },
    { id: "health",    label: "Health",    href: null, soon: true },
  ];

  window.renderNav = function renderNav(activePage) {
    const nav = document.createElement("nav");
    nav.className = "manus-nav";

    const brand = document.createElement("a");
    brand.className = "nav-brand";
    brand.href = "/static/dashboard.html";
    brand.innerHTML = 'Manus <span>Gesture</span>';
    nav.appendChild(brand);

    const links = document.createElement("div");
    links.className = "nav-links";

    PAGES.forEach(page => {
      if (page.soon) {
        const span = document.createElement("span");
        span.className = "nav-link nav-link-soon";
        span.textContent = page.label;
        span.title = "Coming soon";
        links.appendChild(span);
      } else {
        const a = document.createElement("a");
        a.className = "nav-link" + (page.id === activePage ? " active" : "");
        a.href = page.href;
        a.textContent = page.label;
        links.appendChild(a);
      }
    });

    nav.appendChild(links);

    // Insert as first child of <body>
    document.body.insertBefore(nav, document.body.firstChild);
  };

  /**
   * Tab switcher utility.
   * Usage: initTabs(".tab-bar", ".tab-panel")
   * Tabs must have data-tab="<id>", panels must have id="panel-<id>"
   */
  window.initTabs = function initTabs(barSelector, panelSelector, onChange) {
    const bars = document.querySelectorAll(barSelector);
    bars.forEach(bar => {
      bar.querySelectorAll(".tab-btn[data-tab]").forEach(btn => {
        btn.addEventListener("click", () => {
          const target = btn.dataset.tab;
          // deactivate siblings in same bar
          bar.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
          btn.classList.add("active");
          // hide panels scoped to bar's parent
          const scope = bar.closest("[data-tab-scope]") || document;
          scope.querySelectorAll(panelSelector).forEach(p => p.classList.remove("active"));
          const panel = scope.querySelector(`#panel-${target}`);
          if (panel) panel.classList.add("active");
          if (onChange) onChange(target);
        });
      });
    });
  };
})();
