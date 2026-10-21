document.addEventListener('DOMContentLoaded', function () {
  var toggle = document.getElementById('sidebarToggle');
  var sidebar = document.getElementById('sidebar');
  var backdrop = document.getElementById('sidebarBackdrop');

  if (!toggle || !sidebar || !backdrop) return;

  function openSidebar() {
    sidebar.classList.add('open');
    backdrop.classList.add('show');
    toggle.setAttribute('aria-expanded', 'true');
  }

  function closeSidebar() {
    sidebar.classList.remove('open');
    backdrop.classList.remove('show');
    toggle.setAttribute('aria-expanded', 'false');
  }

  toggle.addEventListener('click', function () {
    var isOpen = sidebar.classList.contains('open');
    if (isOpen) { closeSidebar(); } else { openSidebar(); }
  });

  backdrop.addEventListener('click', closeSidebar);

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') closeSidebar();
  });

  // Close automatically if the viewport grows past the mobile breakpoint
  window.addEventListener('resize', function () {
    if (window.innerWidth > 900) closeSidebar();
  });
});









/**
 * Moves the .bg-graph-dot / .bg-graph-dot-glow elements along #riskPath.
 *
 * The background line lives inside an SVG that is stretched non-uniformly
 * (preserveAspectRatio="none") so it always reaches the page's true
 * corners. That non-uniform stretch would turn an SVG <circle> into an
 * ellipse, so instead the dot is a plain HTML div (always a perfect
 * circle via border-radius) whose position is computed every frame from
 * the path's real on-screen geometry via getPointAtLength + getScreenCTM.
 */
(function () {
  var TRAVEL = 6000; // ms spent moving bottom-left -> top-right
  var PAUSE = 1000;  // ms held at the top-right before looping
  var CYCLE = TRAVEL + PAUSE;

  var COLOR_STOPS = [
    { t: 0,     c: [63, 177, 152] },  // teal  (low)
    { t: 0.45,  c: [220, 154, 79] },  // amber (moderate)
    { t: 0.857, c: [217, 122, 94] }   // clay  (elevated) — reached exactly as travel ends
  ];

  function colorAt(progress) {
    if (progress <= COLOR_STOPS[0].t) return COLOR_STOPS[0].c;
    for (var i = 1; i < COLOR_STOPS.length; i++) {
      if (progress <= COLOR_STOPS[i].t) {
        var a = COLOR_STOPS[i - 1], b = COLOR_STOPS[i];
        var span = b.t - a.t;
        var localT = span > 0 ? (progress - a.t) / span : 1;
        return [
          Math.round(a.c[0] + (b.c[0] - a.c[0]) * localT),
          Math.round(a.c[1] + (b.c[1] - a.c[1]) * localT),
          Math.round(a.c[2] + (b.c[2] - a.c[2]) * localT)
        ];
      }
    }
    return COLOR_STOPS[COLOR_STOPS.length - 1].c;
  }

  function init() {
    var path = document.getElementById('riskPath');
    var dot = document.querySelector('.bg-graph-dot');
    var glow = document.querySelector('.bg-graph-dot-glow');
    if (!path || !dot || !glow) return;

    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return; // .bg-motion is display:none in this case anyway
    }

    var totalLength = path.getTotalLength();
    var dotSize = 14;
    var glowSize = 34;

    function place(el, screenPt, size) {
      el.style.transform =
        'translate3d(' + (screenPt.x - size / 2) + 'px, ' + (screenPt.y - size / 2) + 'px, 0)';
    }

    function frame(ts) {
      var elapsed = ts % CYCLE;
      var progress = elapsed <= TRAVEL ? elapsed / TRAVEL : 1;

      var pt = path.getPointAtLength(progress * totalLength);
      var ctm = path.getScreenCTM();
      if (ctm) {
        var svgPoint = path.ownerSVGElement.createSVGPoint();
        svgPoint.x = pt.x;
        svgPoint.y = pt.y;
        var screenPt = svgPoint.matrixTransform(ctm);

        place(dot, screenPt, dotSize);
        place(glow, screenPt, glowSize);
      }

      var rgb = colorAt(progress);
      var color = 'rgb(' + rgb[0] + ', ' + rgb[1] + ', ' + rgb[2] + ')';
      dot.style.background = color;
      glow.style.background = color;

      requestAnimationFrame(frame);
    }

    requestAnimationFrame(frame);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();